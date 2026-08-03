"""Typed public contracts for Phase 5A speaker similarity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from voiceid.embeddings.contracts import EMBEDDING_DIMENSION

COSINE_SIMILARITY_METRIC: Final = "cosine_similarity"
SIMILARITY_COMPARISON_VERSION: Final = "1"


class SimilarityStatus(StrEnum):
    """Public status of a speaker similarity result."""

    VALID = "VALID"
    INVALID = "INVALID"


class SimilarityErrorCode(StrEnum):
    """Stable machine-readable similarity error codes."""

    INVALID_REFERENCE = "INVALID_REFERENCE"
    INVALID_CANDIDATE = "INVALID_CANDIDATE"
    INVALID_EMBEDDING = "INVALID_EMBEDDING"
    ZERO_OR_NEAR_ZERO_EMBEDDING = "ZERO_OR_NEAR_ZERO_EMBEDDING"
    INCOMPATIBLE_EMBEDDINGS = "INCOMPATIBLE_EMBEDDINGS"
    COMPARISON_ERROR = "COMPARISON_ERROR"


_ERROR_MESSAGES: Final[dict[SimilarityErrorCode, str]] = {
    SimilarityErrorCode.INVALID_REFERENCE: (
        "The reference embedding result is invalid."
    ),
    SimilarityErrorCode.INVALID_CANDIDATE: (
        "The candidate embedding result is invalid."
    ),
    SimilarityErrorCode.INVALID_EMBEDDING: (
        "A speaker embedding violates the comparison contract."
    ),
    SimilarityErrorCode.ZERO_OR_NEAR_ZERO_EMBEDDING: (
        "A speaker embedding has zero or near-zero L2 norm."
    ),
    SimilarityErrorCode.INCOMPATIBLE_EMBEDDINGS: (
        "The speaker embeddings are not compatible for comparison."
    ),
    SimilarityErrorCode.COMPARISON_ERROR: (
        "Speaker embedding comparison failed safely."
    ),
}


@dataclass(frozen=True, slots=True)
class SimilarityIssue:
    """User-safe similarity issue with a stable code and message."""

    code: str
    message: str

    def __post_init__(self) -> None:
        """Reject unknown codes and arbitrary public error messages."""

        if type(self.code) is not str or type(self.message) is not str:
            raise ValueError("Similarity issue code and message must be strings.")
        try:
            error_code = SimilarityErrorCode(self.code)
        except ValueError as exc:
            raise ValueError("Similarity issue code must be supported.") from exc
        if self.message != _ERROR_MESSAGES[error_code]:
            raise ValueError("Similarity issue message must be stable and safe.")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable issue representation."""

        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class SpeakerSimilarityMetadata:
    """Non-sensitive metadata describing a speaker similarity calculation."""

    metric: str
    comparison_version: str
    embedding_dimension: int
    normalized: bool

    def __post_init__(self) -> None:
        """Enforce the fixed Phase 5A comparison contract."""

        if type(self.metric) is not str or self.metric != COSINE_SIMILARITY_METRIC:
            raise ValueError("Similarity metric must be cosine_similarity.")
        if (
            type(self.comparison_version) is not str
            or self.comparison_version != SIMILARITY_COMPARISON_VERSION
        ):
            raise ValueError("Similarity comparison version must be 1.")
        if (
            type(self.embedding_dimension) is not int
            or self.embedding_dimension != EMBEDDING_DIMENSION
        ):
            raise ValueError("Similarity embedding dimension must be 192.")
        if type(self.normalized) is not bool:
            raise ValueError("Similarity normalized flag must be boolean.")

    def to_dict(self) -> dict[str, str | int | bool]:
        """Return public metadata without embedding values or vector norms."""

        return {
            "metric": self.metric,
            "comparison_version": self.comparison_version,
            "embedding_dimension": self.embedding_dimension,
            "normalized": self.normalized,
        }


@dataclass(frozen=True, slots=True, repr=False)
class SpeakerSimilarityResult:
    """Privacy-safe result of comparing two compatible speaker embeddings."""

    status: SimilarityStatus
    similarity: float | None
    metadata: SpeakerSimilarityMetadata | None
    errors: tuple[SimilarityIssue, ...] = ()

    def __post_init__(self) -> None:
        """Prevent contradictory public result states."""

        if not isinstance(self.status, SimilarityStatus):
            raise ValueError("Similarity status must be a SimilarityStatus.")
        if self.status == SimilarityStatus.VALID:
            self._validate_valid()
            return
        self._validate_invalid()

    @property
    def is_valid(self) -> bool:
        """Return whether similarity was calculated successfully."""

        return self.status == SimilarityStatus.VALID

    def __repr__(self) -> str:
        """Return a safe representation without embeddings or vector norms."""

        similarity = f"{self.similarity:.6f}" if self.similarity is not None else "None"
        return (
            "SpeakerSimilarityResult("
            f"status={self.status!r}, "
            f"similarity={similarity}, "
            f"metadata={self.metadata!r}, "
            f"errors={self.errors!r})"
        )

    def __str__(self) -> str:
        """Return the same privacy-safe representation as repr()."""

        return repr(self)

    def to_dict(self) -> dict[str, object]:
        """Return the public contract without embeddings or vector norms."""

        return {
            "status": self.status.value,
            "similarity": self.similarity,
            "metadata": self.metadata.to_dict() if self.metadata else {},
            "errors": [error.to_dict() for error in self.errors],
        }

    def _validate_valid(self) -> None:
        if type(self.similarity) is not float:
            raise ValueError("VALID similarity result requires a Python float.")
        if not math.isfinite(self.similarity):
            raise ValueError("VALID similarity result requires a finite score.")
        if not -1.0 <= self.similarity <= 1.0:
            raise ValueError("VALID similarity score must be in [-1.0, 1.0].")
        if not isinstance(self.metadata, SpeakerSimilarityMetadata):
            raise ValueError("VALID similarity result requires metadata.")
        if self.errors != ():
            raise ValueError("VALID similarity result cannot contain errors.")

    def _validate_invalid(self) -> None:
        if self.similarity is not None:
            raise ValueError("INVALID similarity result cannot contain a score.")
        if self.metadata is not None:
            raise ValueError("INVALID similarity result cannot contain metadata.")
        if (
            not isinstance(self.errors, tuple)
            or len(self.errors) != 1
            or not isinstance(self.errors[0], SimilarityIssue)
        ):
            raise ValueError("INVALID similarity result requires exactly one error.")


def build_invalid_similarity_result(
    code: SimilarityErrorCode,
) -> SpeakerSimilarityResult:
    """Build an invalid result with one stable privacy-safe issue."""

    return SpeakerSimilarityResult(
        status=SimilarityStatus.INVALID,
        similarity=None,
        metadata=None,
        errors=(SimilarityIssue(code=code.value, message=_ERROR_MESSAGES[code]),),
    )
