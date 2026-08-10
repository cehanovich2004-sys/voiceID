"""Typed public contracts for Phase 4 speaker embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt

from voiceid.audio.preprocessing import PREPROCESSING_CONTRACT_VERSION
from voiceid.embeddings.policy import SPEECHBRAIN_ECAPA_EMBEDDING_DIMENSION

EmbeddingVector = npt.NDArray[np.float32]
EMBEDDING_DIMENSION = SPEECHBRAIN_ECAPA_EMBEDDING_DIMENSION
EMBEDDING_CONTRACT_VERSION: Final = "phase4b-v1"
_VERSION_IDENTIFIER_CHARACTERS: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyz0123456789.-_+"
)


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
    backend_version: str = field(repr=False)
    preprocessing_contract_version: str = field(repr=False)
    embedding_contract_version: str = field(repr=False)
    device: str
    input_sample_rate_hz: int
    input_samples: int
    input_duration_seconds: float
    normalized: bool

    def __post_init__(self) -> None:
        """Validate fixed contract versions and backend provenance."""

        _validate_version_field(
            name="preprocessing contract version",
            value=self.preprocessing_contract_version,
            expected=PREPROCESSING_CONTRACT_VERSION,
        )
        _validate_version_field(
            name="embedding contract version",
            value=self.embedding_contract_version,
            expected=EMBEDDING_CONTRACT_VERSION,
        )
        _validate_version_field(
            name="backend version",
            value=self.backend_version,
        )

    def to_dict(self) -> dict[str, str | int | float | bool]:
        """Return public metadata without embedding values."""

        _validate_embedding_metadata_versions(self)
        return {
            "embedding_dimension": self.embedding_dimension,
            "model_identifier": self.model_identifier,
            "model_revision": self.model_revision,
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "preprocessing_contract_version": self.preprocessing_contract_version,
            "embedding_contract_version": self.embedding_contract_version,
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

    def __post_init__(self) -> None:
        """Enforce the public result invariants at construction time."""

        if self.status == EmbeddingStatus.VALID:
            embedding = _validate_valid_result(
                embedding=self.embedding,
                metadata=self.metadata,
                errors=self.errors,
            )
            object.__setattr__(self, "embedding", embedding)
            return

        _validate_invalid_result(
            embedding=self.embedding,
            metadata=self.metadata,
            errors=self.errors,
        )

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


def _validate_valid_result(
    *,
    embedding: EmbeddingVector | None,
    metadata: EmbeddingMetadata | None,
    errors: tuple[EmbeddingIssue, ...],
) -> EmbeddingVector:
    if embedding is None:
        raise ValueError("VALID embedding result requires an embedding.")
    if metadata is None:
        raise ValueError("VALID embedding result requires metadata.")
    if errors:
        raise ValueError("VALID embedding result cannot contain errors.")
    if not isinstance(embedding, np.ndarray):
        raise ValueError("VALID embedding result requires a numpy embedding.")
    if embedding.dtype != np.float32:
        raise ValueError("VALID embedding result requires float32 embedding dtype.")
    if embedding.ndim != 1:
        raise ValueError("VALID embedding result requires a one-dimensional embedding.")
    if embedding.shape != (EMBEDDING_DIMENSION,):
        raise ValueError("VALID embedding result requires shape (192,).")
    if metadata.embedding_dimension != EMBEDDING_DIMENSION:
        raise ValueError("VALID embedding metadata dimension must be 192.")
    _validate_embedding_metadata_versions(metadata)
    if not np.all(np.isfinite(embedding)):
        raise ValueError("VALID embedding result requires finite embedding values.")
    copied = embedding.copy()
    copied.setflags(write=False)
    return copied


def _validate_invalid_result(
    *,
    embedding: EmbeddingVector | None,
    metadata: EmbeddingMetadata | None,
    errors: tuple[EmbeddingIssue, ...],
) -> None:
    if embedding is not None:
        raise ValueError("INVALID embedding result cannot contain an embedding.")
    if metadata is not None:
        raise ValueError("INVALID embedding result cannot contain metadata.")
    if len(errors) != 1:
        raise ValueError("INVALID embedding result requires exactly one error.")


def _validate_embedding_metadata_versions(metadata: EmbeddingMetadata) -> None:
    _validate_version_field(
        name="preprocessing contract version",
        value=metadata.preprocessing_contract_version,
        expected=PREPROCESSING_CONTRACT_VERSION,
    )
    _validate_version_field(
        name="embedding contract version",
        value=metadata.embedding_contract_version,
        expected=EMBEDDING_CONTRACT_VERSION,
    )
    _validate_version_field(
        name="backend version",
        value=metadata.backend_version,
    )


def _validate_version_field(
    *,
    name: str,
    value: object,
    expected: str | None = None,
) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"Embedding {name} must be a non-empty string.")
    if not is_valid_version_identifier(value):
        raise ValueError(f"Embedding {name} must be a safe version identifier.")
    if expected is not None and value != expected:
        raise ValueError(f"Embedding {name} is not supported.")


def is_valid_version_identifier(value: object) -> bool:
    """Return whether a public contract version is a safe stable identifier."""

    return (
        type(value) is str
        and bool(value.strip())
        and value == value.strip()
        and value.isascii()
        and len(value) <= 128
        and all(
            character.lower() in _VERSION_IDENTIFIER_CHARACTERS for character in value
        )
    )
