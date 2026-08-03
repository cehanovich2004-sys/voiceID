"""Pure deterministic comparison of compatible speaker embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from voiceid.embeddings.contracts import (
    EMBEDDING_DIMENSION,
    EmbeddingMetadata,
    EmbeddingStatus,
    EmbeddingVector,
    SpeakerEmbeddingResult,
)
from voiceid.similarity.contracts import (
    COSINE_SIMILARITY_METRIC,
    SIMILARITY_COMPARISON_VERSION,
    SimilarityErrorCode,
    SimilarityStatus,
    SpeakerSimilarityMetadata,
    SpeakerSimilarityResult,
    build_invalid_similarity_result,
)

EXPECTED_SAMPLE_RATE_HZ: Final = 16_000
MIN_EMBEDDING_L2_NORM: Final = 1e-8


def compare_speaker_embeddings(
    reference: SpeakerEmbeddingResult,
    candidate: SpeakerEmbeddingResult,
) -> SpeakerSimilarityResult:
    """Return raw cosine similarity for two compatible speaker embeddings."""

    try:
        _validate_result_status(reference, SimilarityErrorCode.INVALID_REFERENCE)
        _validate_result_status(candidate, SimilarityErrorCode.INVALID_CANDIDATE)

        reference_input = _validate_embedding(reference)
        candidate_input = _validate_embedding(candidate)

        reference_norm = _l2_norm_float64(reference_input.embedding)
        candidate_norm = _l2_norm_float64(candidate_input.embedding)
        if reference_norm <= MIN_EMBEDDING_L2_NORM:
            raise _SimilarityValidationError(
                SimilarityErrorCode.ZERO_OR_NEAR_ZERO_EMBEDDING
            )
        if candidate_norm <= MIN_EMBEDDING_L2_NORM:
            raise _SimilarityValidationError(
                SimilarityErrorCode.ZERO_OR_NEAR_ZERO_EMBEDDING
            )

        _validate_compatibility(reference_input.metadata, candidate_input.metadata)
        similarity = _cosine_similarity_float64(
            reference_input.embedding,
            candidate_input.embedding,
            reference_norm=reference_norm,
            candidate_norm=candidate_norm,
        )
        return SpeakerSimilarityResult(
            status=SimilarityStatus.VALID,
            similarity=similarity,
            metadata=_build_metadata(reference_input.metadata),
            errors=(),
        )
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except MemoryError:
        return build_invalid_similarity_result(SimilarityErrorCode.COMPARISON_ERROR)
    except _SimilarityValidationError as exc:
        return build_invalid_similarity_result(exc.code)
    except Exception:
        return build_invalid_similarity_result(SimilarityErrorCode.COMPARISON_ERROR)


@dataclass(frozen=True, slots=True)
class _ValidatedEmbedding:
    embedding: EmbeddingVector
    metadata: EmbeddingMetadata


class _SimilarityValidationError(Exception):
    def __init__(self, code: SimilarityErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


def _validate_result_status(
    result: SpeakerEmbeddingResult,
    error_code: SimilarityErrorCode,
) -> None:
    if not isinstance(result, SpeakerEmbeddingResult):
        raise _SimilarityValidationError(error_code)
    if (
        not isinstance(result.status, EmbeddingStatus)
        or result.status is not EmbeddingStatus.VALID
        or not result.is_valid
        or result.errors != ()
    ):
        raise _SimilarityValidationError(error_code)


def _validate_embedding(result: SpeakerEmbeddingResult) -> _ValidatedEmbedding:
    embedding = result.embedding
    metadata = result.metadata
    if not isinstance(embedding, np.ndarray):
        raise _SimilarityValidationError(SimilarityErrorCode.INVALID_EMBEDDING)
    if not isinstance(metadata, EmbeddingMetadata):
        raise _SimilarityValidationError(SimilarityErrorCode.INVALID_EMBEDDING)
    if embedding.dtype != np.float32 or embedding.ndim != 1:
        raise _SimilarityValidationError(SimilarityErrorCode.INVALID_EMBEDDING)
    if metadata.embedding_dimension != EMBEDDING_DIMENSION:
        raise _SimilarityValidationError(SimilarityErrorCode.INVALID_EMBEDDING)
    if embedding.shape != (metadata.embedding_dimension,):
        raise _SimilarityValidationError(SimilarityErrorCode.INVALID_EMBEDDING)
    if embedding.flags.writeable is not False:
        raise _SimilarityValidationError(SimilarityErrorCode.INVALID_EMBEDDING)
    if not bool(np.all(np.isfinite(embedding))):
        raise _SimilarityValidationError(SimilarityErrorCode.INVALID_EMBEDDING)
    _validate_input_metadata(metadata)
    return _ValidatedEmbedding(embedding=embedding, metadata=metadata)


def _validate_input_metadata(metadata: EmbeddingMetadata) -> None:
    if type(metadata.input_samples) is not int or metadata.input_samples <= 0:
        raise _SimilarityValidationError(SimilarityErrorCode.INVALID_EMBEDDING)
    if (
        type(metadata.input_duration_seconds) is not float
        or not math.isfinite(metadata.input_duration_seconds)
        or metadata.input_duration_seconds <= 0.0
    ):
        raise _SimilarityValidationError(SimilarityErrorCode.INVALID_EMBEDDING)
    try:
        expected_duration = round(
            metadata.input_samples / EXPECTED_SAMPLE_RATE_HZ,
            6,
        )
    except OverflowError:
        raise _SimilarityValidationError(
            SimilarityErrorCode.INVALID_EMBEDDING
        ) from None
    if metadata.input_duration_seconds != expected_duration:
        raise _SimilarityValidationError(SimilarityErrorCode.INVALID_EMBEDDING)


def _l2_norm_float64(embedding: EmbeddingVector) -> float:
    values = embedding.astype(np.float64, copy=False)
    squared_sum = np.sum(np.square(values), dtype=np.float64)
    norm = float(np.sqrt(squared_sum))
    if not math.isfinite(norm):
        raise _SimilarityValidationError(SimilarityErrorCode.INVALID_EMBEDDING)
    return norm


def _validate_compatibility(
    reference: EmbeddingMetadata,
    candidate: EmbeddingMetadata,
) -> None:
    metadata_fields_are_valid = (
        type(reference.input_sample_rate_hz) is int
        and reference.input_sample_rate_hz == EXPECTED_SAMPLE_RATE_HZ
        and type(candidate.input_sample_rate_hz) is int
        and candidate.input_sample_rate_hz == EXPECTED_SAMPLE_RATE_HZ
        and type(reference.model_identifier) is str
        and bool(reference.model_identifier)
        and type(candidate.model_identifier) is str
        and bool(candidate.model_identifier)
        and type(reference.model_revision) is str
        and bool(reference.model_revision)
        and type(candidate.model_revision) is str
        and bool(candidate.model_revision)
        and type(reference.backend_name) is str
        and bool(reference.backend_name)
        and type(candidate.backend_name) is str
        and bool(candidate.backend_name)
        and type(reference.normalized) is bool
        and type(candidate.normalized) is bool
    )
    metadata_matches = (
        reference.embedding_dimension == candidate.embedding_dimension
        and reference.model_identifier == candidate.model_identifier
        and reference.model_revision == candidate.model_revision
        and reference.backend_name == candidate.backend_name
        and reference.normalized == candidate.normalized
    )
    if not metadata_fields_are_valid or not metadata_matches:
        raise _SimilarityValidationError(SimilarityErrorCode.INCOMPATIBLE_EMBEDDINGS)


def _cosine_similarity_float64(
    reference: EmbeddingVector,
    candidate: EmbeddingVector,
    *,
    reference_norm: float,
    candidate_norm: float,
) -> float:
    reference_values = reference.astype(np.float64, copy=False)
    candidate_values = candidate.astype(np.float64, copy=False)
    dot_product = float(np.dot(reference_values, candidate_values))
    similarity = dot_product / (reference_norm * candidate_norm)
    if not math.isfinite(similarity):
        raise _SimilarityValidationError(SimilarityErrorCode.COMPARISON_ERROR)
    return float(np.clip(similarity, -1.0, 1.0))


def _build_metadata(source: EmbeddingMetadata) -> SpeakerSimilarityMetadata:
    return SpeakerSimilarityMetadata(
        metric=COSINE_SIMILARITY_METRIC,
        comparison_version=SIMILARITY_COMPARISON_VERSION,
        embedding_dimension=EMBEDDING_DIMENSION,
        normalized=source.normalized,
    )
