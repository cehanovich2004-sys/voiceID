"""Deterministic audio preprocessing for Phase 3."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Final, cast

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from voiceid.audio.models import AudioValidationResult
from voiceid.audio.wav_reader import PCM16_ABS_MAX, WavDecodeError, WavHeader

TARGET_PREPROCESSING_SAMPLE_RATE_HZ = 16000
TARGET_PREPROCESSING_CHANNELS = 1
PUBLIC_FLOAT_DECIMALS = 6
PREPROCESSING_CONTRACT_VERSION: Final = "phase3-v1"

Float32Waveform = npt.NDArray[np.float32]


class PreprocessingStatus(StrEnum):
    """Public status of a Phase 3 preprocessing result."""

    VALID = "VALID"
    INVALID = "INVALID"


class PreprocessingErrorCode(StrEnum):
    """Stable machine-readable preprocessing error codes."""

    INVALID_INPUT = "INVALID_INPUT"
    DECODE_ERROR = "DECODE_ERROR"
    PREPROCESSING_ERROR = "PREPROCESSING_ERROR"


@dataclass(frozen=True, slots=True)
class PreprocessingIssue:
    """User-safe preprocessing issue with a stable code."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable issue representation."""

        return {
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class PreprocessedAudioMetadata:
    """Non-sensitive metadata for a preprocessed waveform."""

    source_sample_rate_hz: int
    source_channels: int
    source_duration_seconds: float
    output_sample_rate_hz: int
    output_channels: int
    output_samples: int
    output_duration_seconds: float
    downmixed_to_mono: bool
    dc_offset_removed: bool
    resampled: bool
    resample_up: int
    resample_down: int
    safety_clipped: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return public metadata without paths or waveform data."""

        return {
            "source_sample_rate_hz": self.source_sample_rate_hz,
            "source_channels": self.source_channels,
            "source_duration_seconds": self.source_duration_seconds,
            "output_sample_rate_hz": self.output_sample_rate_hz,
            "output_channels": self.output_channels,
            "output_samples": self.output_samples,
            "output_duration_seconds": self.output_duration_seconds,
            "downmixed_to_mono": self.downmixed_to_mono,
            "dc_offset_removed": self.dc_offset_removed,
            "resampled": self.resampled,
            "resample_up": self.resample_up,
            "resample_down": self.resample_down,
            "safety_clipped": self.safety_clipped,
        }


@dataclass(frozen=True, slots=True, repr=False)
class PreprocessedAudioResult:
    """Structured Phase 3 preprocessing result.

    The waveform is intentionally excluded from repr() and to_dict() because it
    is biometric signal data, not public metadata.
    """

    status: PreprocessingStatus
    file_name: str
    waveform: Float32Waveform | None
    metadata: PreprocessedAudioMetadata | None
    errors: tuple[PreprocessingIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Return whether preprocessing produced a usable waveform."""

        return self.status == PreprocessingStatus.VALID

    def __repr__(self) -> str:
        """Return a safe representation without waveform values."""

        return (
            "PreprocessedAudioResult("
            f"status={self.status!r}, "
            f"file_name={self.file_name!r}, "
            f"metadata={self.metadata!r}, "
            f"errors={self.errors!r})"
        )

    def to_dict(self) -> dict[str, object]:
        """Return the public dictionary contract without waveform data."""

        return {
            "status": self.status.value,
            "file_name": self.file_name,
            "metadata": self.metadata.to_dict() if self.metadata else {},
            "errors": [error.to_dict() for error in self.errors],
        }


def build_invalid_preprocessing_result(
    *,
    file_name: str,
    code: PreprocessingErrorCode,
    message: str,
) -> PreprocessedAudioResult:
    """Build an invalid preprocessing result without partial waveform data."""

    return PreprocessedAudioResult(
        status=PreprocessingStatus.INVALID,
        file_name=file_name,
        waveform=None,
        metadata=None,
        errors=(PreprocessingIssue(code=code.value, message=message),),
    )


def build_preprocessed_audio_result(
    *,
    file_name: str,
    validation_result: AudioValidationResult,
    header: WavHeader,
    waveform: Float32Waveform,
    downmixed_to_mono: bool,
    resampled: bool,
    resample_up: int,
    resample_down: int,
    safety_clipped: bool,
) -> PreprocessedAudioResult:
    """Build a valid preprocessing result after invariant checks."""

    metadata = PreprocessedAudioMetadata(
        source_sample_rate_hz=header.sample_rate_hz,
        source_channels=header.channels,
        source_duration_seconds=_duration_from_validation(validation_result),
        output_sample_rate_hz=TARGET_PREPROCESSING_SAMPLE_RATE_HZ,
        output_channels=TARGET_PREPROCESSING_CHANNELS,
        output_samples=int(waveform.shape[0]),
        output_duration_seconds=_round_seconds(
            waveform.shape[0] / TARGET_PREPROCESSING_SAMPLE_RATE_HZ
        ),
        downmixed_to_mono=downmixed_to_mono,
        dc_offset_removed=True,
        resampled=resampled,
        resample_up=resample_up,
        resample_down=resample_down,
        safety_clipped=safety_clipped,
    )
    return PreprocessedAudioResult(
        status=PreprocessingStatus.VALID,
        file_name=file_name,
        waveform=waveform,
        metadata=metadata,
        errors=(),
    )


def decode_pcm16_to_float32(path: Path, header: WavHeader) -> Float32Waveform:
    """Decode a validated PCM16 WAV into normalized float32 frames."""

    try:
        with wave.open(str(path), "rb") as wav_file:
            return decode_pcm16_to_float32_from_wave_file(wav_file, header)
    except (EOFError, OSError, ValueError, wave.Error) as exc:
        raise WavDecodeError from exc


def decode_pcm16_to_float32_from_file(
    wav_file: BinaryIO,
    header: WavHeader,
) -> Float32Waveform:
    """Decode PCM16 from an already opened WAV snapshot."""

    try:
        wav_file.seek(0)
        with wave.open(wav_file, "rb") as wave_reader:
            return decode_pcm16_to_float32_from_wave_file(wave_reader, header)
    except (EOFError, OSError, ValueError, wave.Error) as exc:
        raise WavDecodeError from exc


def decode_pcm16_to_float32_from_wave_file(
    wav_file: wave.Wave_read,
    header: WavHeader,
) -> Float32Waveform:
    """Decode PCM16 from an open wave reader into normalized float32 frames."""

    _assert_wave_matches_header(wav_file, header)
    frame_bytes = wav_file.readframes(header.total_frames)

    expected_bytes = header.total_frames * header.channels * 2
    if len(frame_bytes) != expected_bytes:
        raise WavDecodeError

    pcm16 = np.frombuffer(frame_bytes, dtype="<i2")
    if pcm16.size != header.total_frames * header.channels:
        raise WavDecodeError

    normalized = pcm16.astype(np.float32) / np.float32(PCM16_ABS_MAX)
    if header.channels == TARGET_PREPROCESSING_CHANNELS:
        return normalized.copy()

    return normalized.reshape(header.total_frames, header.channels).copy()


def preprocess_validated_waveform(
    *,
    waveform: Float32Waveform,
    source_sample_rate_hz: int,
) -> tuple[Float32Waveform, bool, bool, int, int, bool]:
    """Run the deterministic Phase 3 signal pipeline."""

    mono_waveform, downmixed_to_mono = _downmix_to_mono(waveform)
    dc_corrected = _remove_dc_offset(mono_waveform)
    resampled_waveform, resampled, up, down = _resample_to_target(
        dc_corrected,
        source_sample_rate_hz,
    )
    clipped_waveform, safety_clipped = _safety_clip(resampled_waveform)
    checked_waveform = _validate_output_invariants(clipped_waveform)
    return checked_waveform, downmixed_to_mono, resampled, up, down, safety_clipped


def _downmix_to_mono(waveform: Float32Waveform) -> tuple[Float32Waveform, bool]:
    if waveform.ndim == 1:
        return waveform.astype(np.float32, copy=False), False
    if waveform.ndim != 2 or waveform.shape[1] != 2:
        raise WavDecodeError
    mono = np.asarray(np.mean(waveform, axis=1, dtype=np.float32), dtype=np.float32)
    return mono, True


def _remove_dc_offset(waveform: Float32Waveform) -> Float32Waveform:
    if waveform.size == 0:
        raise WavDecodeError
    mean_value = np.mean(waveform, dtype=np.float64)
    return (waveform.astype(np.float32, copy=False) - np.float32(mean_value)).astype(
        np.float32,
        copy=False,
    )


def _resample_to_target(
    waveform: Float32Waveform,
    source_sample_rate_hz: int,
) -> tuple[Float32Waveform, bool, int, int]:
    if source_sample_rate_hz == TARGET_PREPROCESSING_SAMPLE_RATE_HZ:
        return waveform.astype(np.float32, copy=False), False, 1, 1

    rate_gcd = math.gcd(source_sample_rate_hz, TARGET_PREPROCESSING_SAMPLE_RATE_HZ)
    up = TARGET_PREPROCESSING_SAMPLE_RATE_HZ // rate_gcd
    down = source_sample_rate_hz // rate_gcd
    resampled = resample_poly(waveform, up, down)
    return (
        cast(Float32Waveform, np.asarray(resampled, dtype=np.float32)),
        True,
        up,
        down,
    )


def _safety_clip(waveform: Float32Waveform) -> tuple[Float32Waveform, bool]:
    if not np.all(np.isfinite(waveform)):
        raise WavDecodeError
    safety_clipped = bool(np.any((waveform < -1.0) | (waveform > 1.0)))
    clipped = np.asarray(np.clip(waveform, -1.0, 1.0), dtype=np.float32)
    return clipped, safety_clipped


def _validate_output_invariants(waveform: Float32Waveform) -> Float32Waveform:
    checked = np.asarray(waveform, dtype=np.float32)
    if checked.ndim != 1:
        raise WavDecodeError
    if checked.size == 0:
        raise WavDecodeError
    if not np.all(np.isfinite(checked)):
        raise WavDecodeError
    if bool(np.any((checked < -1.0) | (checked > 1.0))):
        raise WavDecodeError
    return checked


def _assert_wave_matches_header(wav_file: wave.Wave_read, header: WavHeader) -> None:
    if wav_file.getcomptype() != "NONE":
        raise WavDecodeError
    if wav_file.getnchannels() != header.channels:
        raise WavDecodeError
    if wav_file.getframerate() != header.sample_rate_hz:
        raise WavDecodeError
    if wav_file.getsampwidth() != 2:
        raise WavDecodeError
    if wav_file.getnframes() != header.total_frames:
        raise WavDecodeError


def _duration_from_validation(validation_result: AudioValidationResult) -> float:
    if validation_result.metadata is None:
        raise WavDecodeError
    duration_seconds = validation_result.metadata.duration_seconds
    if duration_seconds is None:
        raise WavDecodeError
    return duration_seconds


def _round_seconds(value: float) -> float:
    return round(value, PUBLIC_FLOAT_DECIMALS)
