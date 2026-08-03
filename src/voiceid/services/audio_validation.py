"""Application use case for Phase 2 WAV technical validation."""

from __future__ import annotations

import os
from pathlib import Path

from voiceid.audio.models import (
    AudioMetadata,
    AudioValidationResult,
    IssueValue,
    ValidationErrorCode,
    ValidationIssue,
    ValidationWarningCode,
    build_validation_result,
)
from voiceid.audio.validation_policy import AudioValidationPolicy
from voiceid.audio.wav_reader import (
    PCM_FORMAT_TAG,
    PcmSignalStats,
    WavContainerError,
    WavDecodeError,
    WavFileNotReadableError,
    WavHeader,
    calculate_duration_seconds,
    decode_pcm16_signal_stats,
    read_wav_header,
)


def validate_wav_file(
    file_path: str | Path,
    *,
    policy: AudioValidationPolicy | None = None,
) -> AudioValidationResult:
    """Validate a local WAV file against the Phase 2 technical contract.

    The returned result intentionally contains only a safe filename and never
    exposes the canonical or absolute local path.
    """

    try:
        return _validate_wav_file(file_path, policy=policy)
    except Exception:
        return _safe_decode_error_result(file_path)


def _validate_wav_file(
    file_path: str | Path,
    *,
    policy: AudioValidationPolicy | None = None,
) -> AudioValidationResult:
    """Internal implementation for the public WAV validation boundary."""

    active_policy = policy or AudioValidationPolicy()
    path = Path(file_path)
    file_name = _safe_file_name(path)
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    path_error = _validate_path(path)
    if path_error is not None:
        errors.append(path_error)
        return build_validation_result(
            file_name=file_name,
            metadata=None,
            warnings=warnings,
            errors=errors,
        )

    extension_error = _validate_extension(path)
    if extension_error is not None:
        errors.append(extension_error)
        return build_validation_result(
            file_name=file_name,
            metadata=None,
            warnings=warnings,
            errors=errors,
        )

    try:
        header = read_wav_header(path)
    except WavFileNotReadableError:
        errors.append(
            _error(
                ValidationErrorCode.FILE_NOT_READABLE,
                "The file cannot be read.",
            )
        )
        return build_validation_result(
            file_name=file_name,
            metadata=None,
            warnings=warnings,
            errors=errors,
        )
    except WavContainerError:
        errors.append(
            _error(
                ValidationErrorCode.INVALID_WAV_CONTAINER,
                "The file is not a valid RIFF/WAVE container.",
            )
        )
        return build_validation_result(
            file_name=file_name,
            metadata=None,
            warnings=warnings,
            errors=errors,
        )
    except WavDecodeError:
        errors.append(
            _error(
                ValidationErrorCode.DECODE_ERROR,
                "The WAV file could not be decoded safely.",
            )
        )
        return build_validation_result(
            file_name=file_name,
            metadata=None,
            warnings=warnings,
            errors=errors,
        )

    metadata = _metadata_from_header(header, active_policy)
    header_result = validate_wav_header_snapshot(
        file_name=file_name,
        header=header,
        policy=active_policy,
    )
    if not header_result.is_valid:
        return header_result

    warnings.extend(header_result.warnings)

    try:
        signal_stats = decode_pcm16_signal_stats(
            path,
            header,
            clipping_sample_level_threshold=(
                active_policy.clipping_sample_level_threshold
            ),
        )
    except WavFileNotReadableError:
        errors.append(
            _error(
                ValidationErrorCode.FILE_NOT_READABLE,
                "The file cannot be read.",
            )
        )
        return build_validation_result(
            file_name=file_name,
            metadata=metadata,
            warnings=warnings,
            errors=errors,
        )
    except WavDecodeError:
        errors.append(
            _error(
                ValidationErrorCode.DECODE_ERROR,
                "The WAV file could not be decoded safely.",
            )
        )
        return build_validation_result(
            file_name=file_name,
            metadata=metadata,
            warnings=warnings,
            errors=errors,
        )

    return validate_wav_signal_snapshot(
        file_name=file_name,
        header=header,
        signal_stats=signal_stats,
        policy=active_policy,
    )


def _validate_path(path: Path) -> ValidationIssue | None:
    try:
        if not path.exists():
            return _error(
                ValidationErrorCode.FILE_NOT_FOUND,
                "The file does not exist.",
            )
        if not path.is_file():
            return _error(
                ValidationErrorCode.FILE_NOT_READABLE,
                "The path does not point to a readable file.",
            )
        if not os.access(path, os.R_OK):
            return _error(
                ValidationErrorCode.FILE_NOT_READABLE,
                "The file cannot be read.",
            )
        if path.stat().st_size == 0:
            return _error(
                ValidationErrorCode.FILE_EMPTY,
                "The file is empty.",
            )
    except PermissionError:
        return _error(
            ValidationErrorCode.FILE_NOT_READABLE,
            "The file cannot be read.",
        )
    except OSError:
        return _error(
            ValidationErrorCode.FILE_NOT_READABLE,
            "The file cannot be inspected safely.",
        )
    return None


def validate_wav_header_snapshot(
    *,
    file_name: str,
    header: WavHeader,
    policy: AudioValidationPolicy | None = None,
) -> AudioValidationResult:
    """Apply Phase 2 header policy to already parsed WAV snapshot metadata."""

    active_policy = policy or AudioValidationPolicy()
    metadata = _metadata_from_header(header, active_policy)
    errors = _validate_header(header, active_policy, metadata)
    warnings: list[ValidationIssue] = []
    if not errors:
        warnings.extend(_structural_warnings(header, active_policy))

    return build_validation_result(
        file_name=file_name,
        metadata=metadata,
        warnings=warnings,
        errors=errors,
    )


def validate_wav_signal_snapshot(
    *,
    file_name: str,
    header: WavHeader,
    signal_stats: PcmSignalStats,
    policy: AudioValidationPolicy | None = None,
) -> AudioValidationResult:
    """Apply full Phase 2 policy to one parsed and decoded WAV snapshot."""

    active_policy = policy or AudioValidationPolicy()
    header_result = validate_wav_header_snapshot(
        file_name=file_name,
        header=header,
        policy=active_policy,
    )
    if not header_result.is_valid:
        return header_result
    if header_result.metadata is None:
        return build_validation_result(
            file_name=file_name,
            metadata=None,
            warnings=[],
            errors=[
                _error(
                    ValidationErrorCode.DECODE_ERROR,
                    "The WAV file could not be decoded safely.",
                )
            ],
        )

    metadata = _metadata_with_signal_stats(
        metadata=header_result.metadata,
        peak_amplitude=signal_stats.peak_amplitude,
        rms_level=signal_stats.rms_level,
    )
    warnings = list(header_result.warnings)
    errors: list[ValidationIssue] = []
    errors.extend(_signal_errors(signal_stats, active_policy))
    if not errors:
        warnings.extend(_signal_warnings(signal_stats, active_policy))

    return build_validation_result(
        file_name=file_name,
        metadata=metadata,
        warnings=warnings,
        errors=errors,
    )


def _validate_extension(path: Path) -> ValidationIssue | None:
    if path.suffix.lower() != ".wav":
        return _error(
            ValidationErrorCode.UNSUPPORTED_EXTENSION,
            "Only local .wav files are supported in Phase 2.",
            field="extension",
            measured_value=path.suffix.lower() or "<none>",
            expected=".wav",
        )
    return None


def _validate_header(
    header: WavHeader,
    policy: AudioValidationPolicy,
    metadata: AudioMetadata,
) -> list[ValidationIssue]:
    errors: list[ValidationIssue] = []

    if header.format_tag != PCM_FORMAT_TAG:
        errors.append(
            _error(
                ValidationErrorCode.UNSUPPORTED_WAV_CODEC,
                "Only uncompressed PCM WAV files are supported.",
                field="codec",
                measured_value=metadata.codec,
                expected="PCM_S16LE",
            )
        )
    if header.sample_width_bits != policy.required_sample_width_bits:
        errors.append(
            _error(
                ValidationErrorCode.UNSUPPORTED_SAMPLE_WIDTH,
                "Only 16-bit PCM WAV files are supported.",
                field="sample_width_bits",
                measured_value=header.sample_width_bits,
                expected=f"{policy.required_sample_width_bits}",
            )
        )
    if header.sample_rate_hz not in policy.supported_sample_rates_hz:
        errors.append(
            _error(
                ValidationErrorCode.UNSUPPORTED_SAMPLE_RATE,
                "The WAV sample rate is not supported in Phase 2.",
                field="sample_rate_hz",
                measured_value=header.sample_rate_hz,
                expected=_format_allowed_values(policy.supported_sample_rates_hz),
            )
        )
    if header.channels not in policy.supported_channel_counts:
        errors.append(
            _error(
                ValidationErrorCode.UNSUPPORTED_CHANNEL_COUNT,
                "Only mono and stereo WAV files are supported.",
                field="channels",
                measured_value=header.channels,
                expected=_format_allowed_values(policy.supported_channel_counts),
            )
        )
    if header.total_frames == 0:
        errors.append(
            _error(
                ValidationErrorCode.ZERO_SAMPLES,
                "The decoded waveform contains zero samples.",
                field="total_samples",
                measured_value=0,
                expected="> 0",
            )
        )

    duration = metadata.duration_seconds
    if duration is not None:
        if duration < policy.min_duration_seconds:
            errors.append(
                _error(
                    ValidationErrorCode.DURATION_TOO_SHORT,
                    "The WAV duration is shorter than the Phase 2 minimum.",
                    field="duration_seconds",
                    measured_value=duration,
                    expected=f">= {policy.min_duration_seconds}",
                )
            )
        if duration > policy.max_duration_seconds:
            errors.append(
                _error(
                    ValidationErrorCode.DURATION_TOO_LONG,
                    "The WAV duration is longer than the configured maximum.",
                    field="duration_seconds",
                    measured_value=duration,
                    expected=f"<= {policy.max_duration_seconds}",
                )
            )

    return errors


def _structural_warnings(
    header: WavHeader,
    policy: AudioValidationPolicy,
) -> list[ValidationIssue]:
    warnings: list[ValidationIssue] = []
    if header.sample_rate_hz != policy.target_sample_rate_hz:
        warnings.append(
            _warning(
                ValidationWarningCode.SAMPLE_RATE_NOT_TARGET,
                "The WAV sample rate differs from the future ML target.",
                field="sample_rate_hz",
                measured_value=header.sample_rate_hz,
                expected=f"{policy.target_sample_rate_hz}",
            )
        )
    if header.channels == 2:
        warnings.append(
            _warning(
                ValidationWarningCode.STEREO_AUDIO,
                "Stereo audio will require documented downmixing in a later phase.",
                field="channels",
                measured_value=header.channels,
                expected=f"{policy.target_channel_count}",
            )
        )
    return warnings


def _signal_errors(
    signal_stats: PcmSignalStats,
    policy: AudioValidationPolicy,
) -> list[ValidationIssue]:
    errors: list[ValidationIssue] = []
    if signal_stats.peak_amplitude == 0 or (
        signal_stats.rms_level <= policy.silent_rms_threshold
    ):
        errors.append(
            _error(
                ValidationErrorCode.SILENT_AUDIO,
                "The audio signal is silent or practically silent.",
                field="rms_level",
                measured_value=signal_stats.rms_level,
                expected=f"> {policy.silent_rms_threshold}",
            )
        )
    return errors


def _signal_warnings(
    signal_stats: PcmSignalStats,
    policy: AudioValidationPolicy,
) -> list[ValidationIssue]:
    warnings: list[ValidationIssue] = []
    if signal_stats.rms_level < policy.low_audio_rms_threshold:
        warnings.append(
            _warning(
                ValidationWarningCode.LOW_AUDIO_LEVEL,
                "The audio signal level is low.",
                field="rms_level",
                measured_value=signal_stats.rms_level,
                expected=f">= {policy.low_audio_rms_threshold}",
            )
        )

    if (
        signal_stats.peak_amplitude >= policy.clipping_peak_threshold
        or signal_stats.clipped_sample_fraction
        >= policy.clipping_sample_fraction_threshold
    ):
        warnings.append(
            _warning(
                ValidationWarningCode.POSSIBLE_CLIPPING,
                "The audio signal may contain clipped samples.",
                field="peak_amplitude",
                measured_value=signal_stats.peak_amplitude,
                expected=f"< {policy.clipping_peak_threshold}",
            )
        )
    return warnings


def _metadata_from_header(
    header: WavHeader,
    policy: AudioValidationPolicy,
) -> AudioMetadata:
    return AudioMetadata(
        container="WAV",
        codec=header.codec,
        sample_rate_hz=header.sample_rate_hz,
        channels=header.channels,
        sample_width_bits=header.sample_width_bits,
        duration_seconds=calculate_duration_seconds(header),
        total_samples=header.total_frames,
        peak_amplitude=None,
        rms_level=None,
        resampling_required=header.sample_rate_hz != policy.target_sample_rate_hz,
        mono_conversion_required=header.channels != policy.target_channel_count,
    )


def _metadata_with_signal_stats(
    *,
    metadata: AudioMetadata,
    peak_amplitude: float,
    rms_level: float,
) -> AudioMetadata:
    return AudioMetadata(
        container=metadata.container,
        codec=metadata.codec,
        sample_rate_hz=metadata.sample_rate_hz,
        channels=metadata.channels,
        sample_width_bits=metadata.sample_width_bits,
        duration_seconds=metadata.duration_seconds,
        total_samples=metadata.total_samples,
        peak_amplitude=peak_amplitude,
        rms_level=rms_level,
        resampling_required=metadata.resampling_required,
        mono_conversion_required=metadata.mono_conversion_required,
    )


def _error(
    code: ValidationErrorCode,
    message: str,
    *,
    field: str | None = None,
    measured_value: IssueValue | None = None,
    expected: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code.value,
        message=message,
        field=field,
        measured_value=measured_value,
        expected=expected,
    )


def _warning(
    code: ValidationWarningCode,
    message: str,
    *,
    field: str | None = None,
    measured_value: IssueValue | None = None,
    expected: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code.value,
        message=message,
        field=field,
        measured_value=measured_value,
        expected=expected,
    )


def _safe_file_name(path: Path) -> str:
    name = path.name or "audio.wav"
    safe_name = "".join(
        character if character.isprintable() and character not in {"/", "\\"} else "_"
        for character in name
    ).strip()
    return safe_name or "audio.wav"


def _safe_file_name_from_input(file_path: str | Path) -> str:
    try:
        return _safe_file_name(Path(file_path))
    except Exception:
        return "audio.wav"


def _safe_decode_error_result(file_path: str | Path) -> AudioValidationResult:
    return build_validation_result(
        file_name=_safe_file_name_from_input(file_path),
        metadata=None,
        warnings=[],
        errors=[
            _error(
                ValidationErrorCode.DECODE_ERROR,
                "The WAV file could not be decoded safely.",
            )
        ],
    )


def _format_allowed_values(values: frozenset[int]) -> str:
    return ", ".join(str(value) for value in sorted(values))
