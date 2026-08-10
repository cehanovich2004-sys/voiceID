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
    PREPROCESSING_CONTRACT_VERSION,
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
    "PREPROCESSING_CONTRACT_VERSION",
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
