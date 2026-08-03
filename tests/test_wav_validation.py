"""Tests for Phase 2 WAV loading and technical validation."""

from __future__ import annotations

import os
import struct
import wave
from pathlib import Path

import pytest

from voiceid.audio.models import (
    AudioValidationResult,
    ValidationErrorCode,
    ValidationStatus,
    ValidationWarningCode,
)
from voiceid.audio.validation_policy import AudioValidationPolicy
from voiceid.audio.wav_reader import WavDecodeError
from voiceid.services.audio_validation import validate_wav_file


def test_valid_mono_pcm16_16khz_returns_metadata_without_warnings(
    tmp_path: Path,
) -> None:
    audio_path = _write_pcm16_wav(tmp_path / "valid.wav", sample_value=8192)

    result = validate_wav_file(audio_path)

    assert result.status == ValidationStatus.VALID
    assert result.is_valid is True
    assert result.file_name == "valid.wav"
    assert result.errors == ()
    assert result.warnings == ()
    assert result.metadata is not None
    assert result.metadata.container == "WAV"
    assert result.metadata.codec == "PCM_S16LE"
    assert result.metadata.sample_rate_hz == 16000
    assert result.metadata.channels == 1
    assert result.metadata.sample_width_bits == 16
    assert result.metadata.duration_seconds == 1.0
    assert result.metadata.total_samples == 16000
    assert result.metadata.peak_amplitude == 0.25
    assert result.metadata.rms_level == 0.25
    assert result.metadata.resampling_required is False
    assert result.metadata.mono_conversion_required is False


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
def test_supported_sample_rates_are_accepted(
    tmp_path: Path,
    sample_rate: int,
) -> None:
    audio_path = _write_pcm16_wav(
        tmp_path / f"rate-{sample_rate}.wav",
        sample_rate_hz=sample_rate,
        sample_value=8192,
    )

    result = validate_wav_file(audio_path)

    assert result.errors == ()
    assert result.metadata is not None
    assert result.metadata.sample_rate_hz == sample_rate
    if sample_rate == 16000:
        assert result.status == ValidationStatus.VALID
        assert result.warnings == ()
    else:
        assert result.status == ValidationStatus.VALID_WITH_WARNINGS
        assert _warning_codes(result) == {
            ValidationWarningCode.SAMPLE_RATE_NOT_TARGET.value
        }


def test_stereo_pcm16_wav_is_valid_with_warning(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(
        tmp_path / "stereo.wav",
        channels=2,
        sample_value=8192,
    )

    result = validate_wav_file(audio_path)

    assert result.status == ValidationStatus.VALID_WITH_WARNINGS
    assert result.errors == ()
    assert _warning_codes(result) == {ValidationWarningCode.STEREO_AUDIO.value}
    assert result.metadata is not None
    assert result.metadata.channels == 2
    assert result.metadata.total_samples == 16000
    assert result.metadata.peak_amplitude == 0.25
    assert result.metadata.rms_level == 0.25
    assert result.metadata.mono_conversion_required is True


@pytest.mark.parametrize("duration_seconds", [1.0, 60.0])
def test_duration_boundaries_are_accepted(
    tmp_path: Path,
    duration_seconds: float,
) -> None:
    audio_path = _write_pcm16_wav(
        tmp_path / f"duration-{duration_seconds}.wav",
        duration_seconds=duration_seconds,
        sample_value=8192,
    )

    result = validate_wav_file(audio_path)

    assert result.status == ValidationStatus.VALID
    assert result.metadata is not None
    assert result.metadata.duration_seconds == duration_seconds


def test_custom_max_duration_config_is_respected(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(
        tmp_path / "custom-max.wav",
        duration_seconds=2.0,
        sample_value=8192,
    )
    policy = AudioValidationPolicy(max_duration_seconds=2.0)

    result = validate_wav_file(audio_path, policy=policy)

    assert result.status == ValidationStatus.VALID
    assert result.errors == ()


def test_duration_total_samples_peak_and_rms_are_deterministic(
    tmp_path: Path,
) -> None:
    audio_path = _write_pcm16_wav(
        tmp_path / "stats.wav",
        sample_rate_hz=8000,
        duration_seconds=1.5,
        sample_value=4096,
    )

    result = validate_wav_file(audio_path)

    assert result.metadata is not None
    assert result.metadata.duration_seconds == 1.5
    assert result.metadata.total_samples == 12000
    assert result.metadata.peak_amplitude == 0.125
    assert result.metadata.rms_level == 0.125


def test_missing_file_is_invalid_without_path_leak(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.wav"

    result = validate_wav_file(missing_path)

    _assert_error(result, ValidationErrorCode.FILE_NOT_FOUND)
    assert result.metadata is None
    assert str(tmp_path) not in repr(result.to_dict())


def test_directory_path_is_invalid(tmp_path: Path) -> None:
    directory_path = tmp_path / "directory.wav"
    directory_path.mkdir()

    result = validate_wav_file(directory_path)

    _assert_error(result, ValidationErrorCode.FILE_NOT_READABLE)


def test_empty_file_is_invalid(tmp_path: Path) -> None:
    audio_path = tmp_path / "empty.wav"
    audio_path.touch()

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.FILE_EMPTY)


def test_unsupported_extension_is_invalid(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"not a wav")

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.UNSUPPORTED_EXTENSION)


def test_wav_extension_with_non_wav_content_is_invalid(tmp_path: Path) -> None:
    audio_path = tmp_path / "not-wav.wav"
    audio_path.write_bytes(b"not a wav")

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.INVALID_WAV_CONTAINER)


def test_truncated_wav_is_invalid_with_decode_error(tmp_path: Path) -> None:
    audio_path = tmp_path / "truncated.wav"
    _write_truncated_wav(audio_path)

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.DECODE_ERROR)


def test_unsupported_wav_codec_is_invalid(tmp_path: Path) -> None:
    audio_path = tmp_path / "float.wav"
    _write_ieee_float_wav(audio_path)

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.UNSUPPORTED_WAV_CODEC)


def test_non_pcm16_wav_is_invalid(tmp_path: Path) -> None:
    audio_path = _write_pcm8_wav(tmp_path / "pcm8.wav")

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.UNSUPPORTED_SAMPLE_WIDTH)


def test_unsupported_sample_rate_is_invalid(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(
        tmp_path / "unsupported-rate.wav",
        sample_rate_hz=11025,
        sample_value=8192,
    )

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.UNSUPPORTED_SAMPLE_RATE)


def test_unsupported_channel_count_is_invalid(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(
        tmp_path / "three-channels.wav",
        channels=3,
        sample_value=8192,
    )

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.UNSUPPORTED_CHANNEL_COUNT)


def test_zero_samples_are_invalid(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(
        tmp_path / "zero-samples.wav",
        duration_seconds=0.0,
    )

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.ZERO_SAMPLES)


def test_duration_shorter_than_minimum_is_invalid(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(
        tmp_path / "too-short.wav",
        duration_seconds=0.5,
        sample_value=8192,
    )

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.DURATION_TOO_SHORT)


def test_duration_longer_than_default_maximum_is_invalid(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(
        tmp_path / "too-long.wav",
        duration_seconds=61.0,
        sample_value=8192,
    )

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.DURATION_TOO_LONG)


def test_duration_longer_than_custom_maximum_is_invalid(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(
        tmp_path / "too-long-custom.wav",
        duration_seconds=3.0,
        sample_value=8192,
    )
    policy = AudioValidationPolicy(max_duration_seconds=2.0)

    result = validate_wav_file(audio_path, policy=policy)

    _assert_error(result, ValidationErrorCode.DURATION_TOO_LONG)


def test_fully_zero_audio_is_invalid_as_silent(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(tmp_path / "zero.wav", sample_value=0)

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.SILENT_AUDIO)


def test_practically_silent_audio_is_invalid(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(tmp_path / "near-silent.wav", sample_value=1)

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.SILENT_AUDIO)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission test")
def test_unreadable_file_is_invalid_when_platform_supports_it(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(tmp_path / "unreadable.wav", sample_value=8192)
    original_mode = audio_path.stat().st_mode
    audio_path.chmod(0)
    try:
        if os.access(audio_path, os.R_OK):
            pytest.skip("current user can still read chmod(0) file")

        result = validate_wav_file(audio_path)
    finally:
        audio_path.chmod(original_mode)

    _assert_error(result, ValidationErrorCode.FILE_NOT_READABLE)


def test_low_audio_level_is_warning_not_error(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(tmp_path / "low.wav", sample_value=100)

    result = validate_wav_file(audio_path)

    assert result.status == ValidationStatus.VALID_WITH_WARNINGS
    assert result.errors == ()
    assert _warning_codes(result) == {ValidationWarningCode.LOW_AUDIO_LEVEL.value}


def test_possible_clipping_is_warning_not_error(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(tmp_path / "clipped.wav", sample_value=32767)

    result = validate_wav_file(audio_path)

    assert result.status == ValidationStatus.VALID_WITH_WARNINGS
    assert result.errors == ()
    assert _warning_codes(result) == {ValidationWarningCode.POSSIBLE_CLIPPING.value}


def test_multiple_warnings_do_not_make_result_invalid(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(
        tmp_path / "multiple-warnings.wav",
        sample_rate_hz=8000,
        channels=2,
        sample_value=100,
    )

    result = validate_wav_file(audio_path)

    assert result.status == ValidationStatus.VALID_WITH_WARNINGS
    assert result.errors == ()
    assert _warning_codes(result) == {
        ValidationWarningCode.SAMPLE_RATE_NOT_TARGET.value,
        ValidationWarningCode.STEREO_AUDIO.value,
        ValidationWarningCode.LOW_AUDIO_LEVEL.value,
    }


def test_internal_decode_exception_is_controlled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = _write_pcm16_wav(tmp_path / "decode-error.wav", sample_value=8192)

    def raise_decode_error(*_args: object, **_kwargs: object) -> object:
        msg = f"internal decoder failed at {tmp_path}"
        raise WavDecodeError(msg)

    monkeypatch.setattr(
        "voiceid.services.audio_validation.decode_pcm16_signal_stats",
        raise_decode_error,
    )

    result = validate_wav_file(audio_path)

    _assert_error(result, ValidationErrorCode.DECODE_ERROR)
    assert str(tmp_path) not in repr(result.to_dict())


def test_public_result_dict_contains_stable_contract(tmp_path: Path) -> None:
    audio_path = _write_pcm16_wav(tmp_path / "contract.wav", sample_value=8192)

    result = validate_wav_file(audio_path)
    payload = result.to_dict()

    assert payload["status"] == "VALID"
    assert payload["file_name"] == "contract.wav"
    assert payload["warnings"] == []
    assert payload["errors"] == []
    assert payload["metadata"] == {
        "container": "WAV",
        "codec": "PCM_S16LE",
        "sample_rate_hz": 16000,
        "channels": 1,
        "sample_width_bits": 16,
        "duration_seconds": 1.0,
        "total_samples": 16000,
        "peak_amplitude": 0.25,
        "rms_level": 0.25,
        "resampling_required": False,
        "mono_conversion_required": False,
    }


def _assert_error(
    result: AudioValidationResult,
    code: ValidationErrorCode,
) -> None:
    assert result.status == ValidationStatus.INVALID
    assert result.is_valid is False
    assert code.value in _error_codes(result)


def _error_codes(result: AudioValidationResult) -> set[str]:
    return {error.code for error in result.errors}


def _warning_codes(result: AudioValidationResult) -> set[str]:
    return {warning.code for warning in result.warnings}


def _write_pcm16_wav(
    path: Path,
    *,
    sample_rate_hz: int = 16000,
    channels: int = 1,
    duration_seconds: float = 1.0,
    sample_value: int = 4096,
) -> Path:
    frame_count = int(round(sample_rate_hz * duration_seconds))
    frame = struct.pack("<h", sample_value) * channels
    return _write_wave_file(
        path=path,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        sample_width_bytes=2,
        frames=frame * frame_count,
    )


def _write_pcm8_wav(path: Path) -> Path:
    frame_count = 16000
    return _write_wave_file(
        path=path,
        sample_rate_hz=16000,
        channels=1,
        sample_width_bytes=1,
        frames=b"\xff" * frame_count,
    )


def _write_wave_file(
    *,
    path: Path,
    sample_rate_hz: int,
    channels: int,
    sample_width_bytes: int,
    frames: bytes,
) -> Path:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width_bytes)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(frames)
    return path


def _write_ieee_float_wav(path: Path) -> None:
    channels = 1
    sample_rate_hz = 16000
    sample_width_bytes = 4
    frame_count = 16000
    fmt_chunk = struct.pack(
        "<HHIIHH",
        3,
        channels,
        sample_rate_hz,
        sample_rate_hz * channels * sample_width_bytes,
        channels * sample_width_bytes,
        sample_width_bytes * 8,
    )
    data_chunk = struct.pack("<f", 0.5) * frame_count
    _write_manual_wav(path, fmt_chunk=fmt_chunk, data_chunk=data_chunk)


def _write_truncated_wav(path: Path) -> None:
    fmt_chunk = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    data_chunk = b"\x01\x00"
    claimed_data_size = 100
    chunks = (
        b"fmt "
        + struct.pack("<I", len(fmt_chunk))
        + fmt_chunk
        + b"data"
        + struct.pack("<I", claimed_data_size)
        + data_chunk
    )
    riff_size = 4 + len(chunks)
    path.write_bytes(b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + chunks)


def _write_manual_wav(path: Path, *, fmt_chunk: bytes, data_chunk: bytes) -> None:
    chunks = (
        b"fmt "
        + struct.pack("<I", len(fmt_chunk))
        + fmt_chunk
        + b"data"
        + struct.pack("<I", len(data_chunk))
        + data_chunk
    )
    riff_size = 4 + len(chunks)
    path.write_bytes(b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + chunks)
