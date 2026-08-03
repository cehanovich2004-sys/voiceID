"""Speaker embedding contracts and backends."""

from voiceid.embeddings.contracts import (
    EmbeddingErrorCode,
    EmbeddingIssue,
    EmbeddingMetadata,
    EmbeddingStatus,
    SpeakerEmbeddingResult,
)
from voiceid.embeddings.policy import SpeakerEmbeddingPolicy

__all__ = [
    "EmbeddingErrorCode",
    "EmbeddingIssue",
    "EmbeddingMetadata",
    "EmbeddingStatus",
    "SpeakerEmbeddingPolicy",
    "SpeakerEmbeddingResult",
]
