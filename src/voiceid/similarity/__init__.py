"""Public contracts and comparison API for speaker similarity."""

from voiceid.similarity.comparison import compare_speaker_embeddings
from voiceid.similarity.contracts import (
    SIMILARITY_COMPARISON_VERSION,
    SimilarityErrorCode,
    SimilarityIssue,
    SimilarityStatus,
    SpeakerSimilarityMetadata,
    SpeakerSimilarityResult,
)

__all__ = [
    "SIMILARITY_COMPARISON_VERSION",
    "SimilarityErrorCode",
    "SimilarityIssue",
    "SimilarityStatus",
    "SpeakerSimilarityMetadata",
    "SpeakerSimilarityResult",
    "compare_speaker_embeddings",
]
