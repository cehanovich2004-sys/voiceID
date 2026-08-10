"""Speaker embedding contracts and backends."""

from voiceid.embeddings.contracts import (
    EMBEDDING_CONTRACT_VERSION,
    EmbeddingErrorCode,
    EmbeddingIssue,
    EmbeddingMetadata,
    EmbeddingStatus,
    SpeakerEmbeddingResult,
)
from voiceid.embeddings.policy import (
    SPEECHBRAIN_ECAPA_BACKEND_VERSION,
    SpeakerEmbeddingPolicy,
)

__all__ = [
    "EMBEDDING_CONTRACT_VERSION",
    "EmbeddingErrorCode",
    "EmbeddingIssue",
    "EmbeddingMetadata",
    "EmbeddingStatus",
    "SPEECHBRAIN_ECAPA_BACKEND_VERSION",
    "SpeakerEmbeddingPolicy",
    "SpeakerEmbeddingResult",
]
