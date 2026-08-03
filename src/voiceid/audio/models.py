"""Typed public validation models for audio files."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

IssueValue = str | int | float | bool


class ValidationStatus(StrEnum):
    """Public status of a technical audio validation result."""

    VALID = "VALID"
    VALID_WITH_WARNINGS = "VALID_WITH_WARNINGS"
    INVALID = "INVALID"


class ValidationErrorCode(StrEnum):
    """Stable machine-readable hard error codes."""

    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_NOT_READABLE = "FILE_NOT_READABLE"
    FILE_EMPTY = "FILE_EMPTY"
    UNSUPPORTED_EXTENSION = "UNSUPPORTED_EXTENSION"
    INVALID_WAV_CONTAINER = "INVALID_WAV_CONTAINER"
    UNSUPPORTED_WAV_CODEC = "UNSUPPORTED_WAV_CODEC"
    UNSUPPORTED_SAMPLE_WIDTH = "UNSUPPORTED_SAMPLE_WIDTH"
    UNSUPPORTED_SAMPLE_RATE = "UNSUPPORTED_SAMPLE_RATE"
    UNSUPPORTED_CHANNEL_COUNT = "UNSUPPORTED_CHANNEL_COUNT"
    ZERO_SAMPLES = "ZERO_SAMPLES"
    DURATION_TOO_SHORT = "DURATION_TOO_SHORT"
    DURATION_TOO_LONG = "DURATION_TOO_LONG"
    SILENT_AUDIO = "SILENT_AUDIO"
    DECODE_ERROR = "DECODE_ERROR"


class ValidationWarningCode(StrEnum):
    """Stable machine-readable non-blocking warning codes."""

    SAMPLE_RATE_NOT_TARGET = "SAMPLE_RATE_NOT_TARGET"
    STEREO_AUDIO = "STEREO_AUDIO"
    LOW_AUDIO_LEVEL = "LOW_AUDIO_LEVEL"
    POSSIBLE_CLIPPING = "POSSIBLE_CLIPPING"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """User-safe validation issue with a stable code."""

    code: str
    message: str
    field: str | None = None
    measured_value: IssueValue | None = None
    expected: str | None = None

    def to_dict(self) -> dict[str, IssueValue | None]:
        """Return a JSON-serializable issue representation."""

        issue: dict[str, IssueValue | None] = {
            "code": self.code,
            "message": self.message,
        }
        if self.field is not None:
            issue["field"] = self.field
        if self.measured_value is not None:
            issue["measured_value"] = self.measured_value
        if self.expected is not None:
            issue["expected"] = self.expected
        return issue


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """Technical WAV metadata exposed by Phase 2 validation."""

    container: str | None = None
    codec: str | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    sample_width_bits: int | None = None
    duration_seconds: float | None = None
    total_samples: int | None = None
    peak_amplitude: float | None = None
    rms_level: float | None = None
    resampling_required: bool | None = None
    mono_conversion_required: bool | None = None

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        """Return a JSON-serializable metadata representation."""

        return {
            "container": self.container,
            "codec": self.codec,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "sample_width_bits": self.sample_width_bits,
            "duration_seconds": self.duration_seconds,
            "total_samples": self.total_samples,
            "peak_amplitude": self.peak_amplitude,
            "rms_level": self.rms_level,
            "resampling_required": self.resampling_required,
            "mono_conversion_required": self.mono_conversion_required,
        }


@dataclass(frozen=True, slots=True)
class AudioValidationResult:
    """Structured result returned by the Phase 2 validation use case."""

    status: ValidationStatus
    file_name: str
    metadata: AudioMetadata | None
    warnings: tuple[ValidationIssue, ...] = ()
    errors: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Return whether the file can continue through the Phase 2 pipeline."""

        return self.status in {
            ValidationStatus.VALID,
            ValidationStatus.VALID_WITH_WARNINGS,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the public dictionary contract without filesystem paths."""

        return {
            "status": self.status.value,
            "file_name": self.file_name,
            "metadata": self.metadata.to_dict() if self.metadata else {},
            "warnings": [warning.to_dict() for warning in self.warnings],
            "errors": [error.to_dict() for error in self.errors],
        }


def build_validation_result(
    *,
    file_name: str,
    metadata: AudioMetadata | None,
    warnings: list[ValidationIssue],
    errors: list[ValidationIssue],
) -> AudioValidationResult:
    """Build a validation result with deterministic status semantics."""

    if errors:
        status = ValidationStatus.INVALID
    elif warnings:
        status = ValidationStatus.VALID_WITH_WARNINGS
    else:
        status = ValidationStatus.VALID

    return AudioValidationResult(
        status=status,
        file_name=file_name,
        metadata=metadata,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )
