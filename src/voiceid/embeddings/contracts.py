"""Typed public contracts for Phase 4 speaker embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

EmbeddingVector = npt.NDArray[np.float32]


class EmbeddingStatus(StrEnum):
    """Public status of a speaker embedding result."""

    VALID = "VALID"
    INVALID = "INVALID"


class EmbeddingErrorCode(StrEnum):
    """Stable machine-readable embedding error codes."""

    INVALID_PREPROCESSED_AUDIO = "INVALID_PREPROCESSED_AUDIO"
    UNSUPPORTED_SAMPLE_RATE = "UNSUPPORTED_SAMPLE_RATE"
    EMPTY_WAVEFORM = "EMPTY_WAVEFORM"
    NON_FINITE_WAVEFORM = "NON_FINITE_WAVEFORM"
    ZERO_OR_NEAR_ZERO_WAVEFORM = "ZERO_OR_NEAR_ZERO_WAVEFORM"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_CACHE_MISSING = "MODEL_CACHE_MISSING"
    MODEL_CACHE_CORRUPTED = "MODEL_CACHE_CORRUPTED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INVALID_EMBEDDING_SHAPE = "INVALID_EMBEDDING_SHAPE"
    INVALID_EMBEDDING_DTYPE = "INVALID_EMBEDDING_DTYPE"
    NON_FINITE_EMBEDDING = "NON_FINITE_EMBEDDING"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"


@dataclass(frozen=True, slots=True)
class EmbeddingIssue:
    """User-safe embedding issue with a stable code."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable issue representation."""

        return {
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingMetadata:
    """Non-sensitive speaker embedding metadata."""

    embedding_dimension: int
    model_identifier: str
    model_revision: str
    backend_name: str
    device: str
    input_sample_rate_hz: int
    input_samples: int
    input_duration_seconds: float
    normalized: bool

    def to_dict(self) -> dict[str, str | int | float | bool]:
        """Return public metadata without embedding values."""

        return {
            "embedding_dimension": self.embedding_dimension,
            "model_identifier": self.model_identifier,
            "model_revision": self.model_revision,
            "backend_name": self.backend_name,
            "device": self.device,
            "input_sample_rate_hz": self.input_sample_rate_hz,
            "input_samples": self.input_samples,
            "input_duration_seconds": self.input_duration_seconds,
            "normalized": self.normalized,
        }


@dataclass(frozen=True, slots=True, repr=False)
class SpeakerEmbeddingResult:
    """Structured Phase 4 embedding result.

    The embedding is intentionally excluded from repr(), str(), and to_dict()
    because it is a sensitive biometric template.
    """

    status: EmbeddingStatus
    embedding: EmbeddingVector | None
    metadata: EmbeddingMetadata | None
    errors: tuple[EmbeddingIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Return whether a valid embedding was produced."""

        return self.status == EmbeddingStatus.VALID

    def __repr__(self) -> str:
        """Return a safe representation without embedding values."""

        return (
            "SpeakerEmbeddingResult("
            f"status={self.status!r}, "
            f"metadata={self.metadata!r}, "
            f"errors={self.errors!r})"
        )

    def __str__(self) -> str:
        """Return a safe string representation without embedding values."""

        return repr(self)

    def to_dict(self) -> dict[str, object]:
        """Return the public dictionary contract without embedding values."""

        return {
            "status": self.status.value,
            "metadata": self.metadata.to_dict() if self.metadata else {},
            "errors": [error.to_dict() for error in self.errors],
        }


def build_invalid_embedding_result(
    *,
    code: EmbeddingErrorCode,
    message: str,
) -> SpeakerEmbeddingResult:
    """Build an invalid embedding result without partial embedding data."""

    return SpeakerEmbeddingResult(
        status=EmbeddingStatus.INVALID,
        embedding=None,
        metadata=None,
        errors=(EmbeddingIssue(code=code.value, message=message),),
    )
