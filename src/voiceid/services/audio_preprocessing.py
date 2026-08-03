"""Application use case for Phase 3 deterministic audio preprocessing."""

from __future__ import annotations

from pathlib import Path

from voiceid.audio.preprocessing import (
    PreprocessedAudioResult,
    PreprocessingErrorCode,
    build_invalid_preprocessing_result,
    build_preprocessed_audio_result,
    decode_pcm16_to_float32,
    preprocess_validated_waveform,
)
from voiceid.audio.validation_policy import AudioValidationPolicy
from voiceid.audio.wav_reader import (
    WavContainerError,
    WavDecodeError,
    WavFileNotReadableError,
    read_wav_header,
)
from voiceid.services.audio_validation import validate_wav_file


def preprocess_wav_file(
    file_path: str | Path,
    *,
    policy: AudioValidationPolicy | None = None,
) -> PreprocessedAudioResult:
    """Preprocess a local Phase 2-valid WAV into float32 mono 16 kHz audio."""

    try:
        return _preprocess_wav_file(file_path, policy=policy)
    except Exception:
        return _safe_preprocessing_error_result(file_path)


def _preprocess_wav_file(
    file_path: str | Path,
    *,
    policy: AudioValidationPolicy | None = None,
) -> PreprocessedAudioResult:
    path = Path(file_path)
    validation_result = validate_wav_file(path, policy=policy)
    file_name = validation_result.file_name
    if not validation_result.is_valid:
        return build_invalid_preprocessing_result(
            file_name=file_name,
            code=PreprocessingErrorCode.INVALID_INPUT,
            message="The WAV file failed Phase 2 technical validation.",
        )

    try:
        header = read_wav_header(path)
        decoded = decode_pcm16_to_float32(path, header)
        (
            waveform,
            downmixed_to_mono,
            resampled,
            up,
            down,
            safety_clipped,
        ) = preprocess_validated_waveform(
            waveform=decoded,
            source_sample_rate_hz=header.sample_rate_hz,
        )
    except (WavContainerError, WavDecodeError, WavFileNotReadableError):
        return build_invalid_preprocessing_result(
            file_name=file_name,
            code=PreprocessingErrorCode.DECODE_ERROR,
            message="The WAV file could not be preprocessed safely.",
        )

    return build_preprocessed_audio_result(
        file_name=file_name,
        validation_result=validation_result,
        header=header,
        waveform=waveform,
        downmixed_to_mono=downmixed_to_mono,
        resampled=resampled,
        resample_up=up,
        resample_down=down,
        safety_clipped=safety_clipped,
    )


def _safe_preprocessing_error_result(file_path: str | Path) -> PreprocessedAudioResult:
    return build_invalid_preprocessing_result(
        file_name=_safe_file_name_from_input(file_path),
        code=PreprocessingErrorCode.PREPROCESSING_ERROR,
        message="The WAV file could not be preprocessed safely.",
    )


def _safe_file_name_from_input(file_path: str | Path) -> str:
    try:
        name = Path(file_path).name or "audio.wav"
    except Exception:
        return "audio.wav"
    safe_name = "".join(
        character if character.isprintable() and character not in {"/", "\\"} else "_"
        for character in name
    ).strip()
    return safe_name or "audio.wav"
