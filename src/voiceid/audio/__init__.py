"""Audio domain models and validation helpers for VoiceID."""

from voiceid.audio.models import (
    AudioMetadata,
    AudioValidationResult,
    ValidationErrorCode,
    ValidationIssue,
    ValidationStatus,
    ValidationWarningCode,
)
from voiceid.audio.preprocessing import (
    PreprocessedAudioMetadata,
    PreprocessedAudioResult,
    PreprocessingErrorCode,
    PreprocessingIssue,
    PreprocessingStatus,
)
from voiceid.audio.validation_policy import AudioValidationPolicy

__all__ = [
    "AudioMetadata",
    "AudioValidationPolicy",
    "AudioValidationResult",
    "PreprocessedAudioMetadata",
    "PreprocessedAudioResult",
    "PreprocessingErrorCode",
    "PreprocessingIssue",
    "PreprocessingStatus",
    "ValidationErrorCode",
    "ValidationIssue",
    "ValidationStatus",
    "ValidationWarningCode",
]
