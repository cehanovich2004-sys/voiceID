"""Public contracts and comparison API for speaker similarity."""

from voiceid.similarity.comparison import compare_speaker_embeddings
from voiceid.similarity.contracts import (
    SimilarityErrorCode,
    SimilarityIssue,
    SimilarityStatus,
    SpeakerSimilarityMetadata,
    SpeakerSimilarityResult,
)

__all__ = [
    "SimilarityErrorCode",
    "SimilarityIssue",
    "SimilarityStatus",
    "SpeakerSimilarityMetadata",
    "SpeakerSimilarityResult",
    "compare_speaker_embeddings",
]
