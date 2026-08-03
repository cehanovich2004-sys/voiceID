"""Tests for speaker embedding public contracts."""

from __future__ import annotations

import numpy as np
import pytest

from voiceid.embeddings.contracts import (
    EmbeddingErrorCode,
    EmbeddingIssue,
    EmbeddingMetadata,
    EmbeddingStatus,
    SpeakerEmbeddingResult,
    build_invalid_embedding_result,
)


def test_valid_embedding_result_hides_embedding_values() -> None:
    embedding = np.arange(192, dtype=np.float32)
    result = SpeakerEmbeddingResult(
        status=EmbeddingStatus.VALID,
        embedding=embedding,
        metadata=EmbeddingMetadata(
            embedding_dimension=192,
            model_identifier="model",
            model_revision="revision",
            backend_name="fake",
            device="cpu",
            input_sample_rate_hz=16000,
            input_samples=16000,
            input_duration_seconds=1.0,
            normalized=False,
        ),
    )

    assert result.is_valid
    assert "191" not in repr(result)
    assert "191" not in str(result)
    assert "embedding" not in result.to_dict()


def test_valid_embedding_result_copies_embedding_as_read_only() -> None:
    embedding = np.arange(192, dtype=np.float32)
    result = SpeakerEmbeddingResult(
        status=EmbeddingStatus.VALID,
        embedding=embedding,
        metadata=_metadata(),
    )

    assert result.embedding is not None
    assert result.embedding.dtype == np.float32
    assert result.embedding.shape == (192,)
    assert result.embedding.flags.writeable is False
    embedding[0] = np.float32(999.0)
    assert result.embedding[0] == np.float32(0.0)
    with pytest.raises(ValueError, match="read-only"):
        result.embedding[1] = np.float32(np.nan)
    assert np.isfinite(result.embedding).all()


def test_valid_embedding_result_accepts_exact_phase4_dimension() -> None:
    result = SpeakerEmbeddingResult(
        status=EmbeddingStatus.VALID,
        embedding=np.full(192, 0.1, dtype=np.float32),
        metadata=_metadata(embedding_dimension=192),
    )

    assert result.is_valid
    assert result.embedding is not None
    assert result.embedding.shape == (192,)


def test_invalid_embedding_result_has_no_partial_embedding_or_metadata() -> None:
    result = build_invalid_embedding_result(
        code=EmbeddingErrorCode.INFERENCE_FAILED,
        message="safe message",
    )

    assert result.status == EmbeddingStatus.INVALID
    assert result.embedding is None
    assert result.metadata is None
    assert result.errors[0].code == "INFERENCE_FAILED"
    assert result.to_dict()["metadata"] == {}


def test_invalid_embedding_result_accepts_exactly_one_error() -> None:
    result = SpeakerEmbeddingResult(
        status=EmbeddingStatus.INVALID,
        embedding=None,
        metadata=None,
        errors=(_issue(),),
    )

    assert result.status == EmbeddingStatus.INVALID
    assert len(result.errors) == 1


@pytest.mark.parametrize(
    ("case_name", "match"),
    [
        (
            "valid_without_embedding",
            "requires an embedding",
        ),
        (
            "valid_without_metadata",
            "requires metadata",
        ),
        (
            "valid_with_errors",
            "cannot contain errors",
        ),
        (
            "valid_with_float64_embedding",
            "float32",
        ),
        (
            "valid_with_two_dimensional_embedding",
            "one-dimensional",
        ),
        (
            "valid_with_wrong_embedding_dimension",
            "shape",
        ),
        (
            "valid_with_one_dimension_metadata",
            "shape",
        ),
        (
            "valid_with_193_dimension_embedding",
            "shape",
        ),
        (
            "valid_with_metadata_dimension_not_192",
            "metadata dimension",
        ),
        (
            "valid_with_non_finite_embedding",
            "finite",
        ),
        (
            "invalid_with_embedding",
            "cannot contain an embedding",
        ),
        (
            "invalid_with_metadata",
            "cannot contain metadata",
        ),
        (
            "invalid_without_errors",
            "requires exactly one error",
        ),
        (
            "invalid_with_two_errors",
            "requires exactly one error",
        ),
    ],
)
def test_embedding_result_rejects_invalid_public_states(
    case_name: str,
    match: str,
) -> None:
    kwargs = _invalid_result_kwargs(case_name)
    with pytest.raises(ValueError, match=match):
        SpeakerEmbeddingResult(**kwargs)  # type: ignore[arg-type]


def _embedding() -> np.ndarray:
    return np.full(192, 0.1, dtype=np.float32)


def _metadata(*, embedding_dimension: int = 192) -> EmbeddingMetadata:
    return EmbeddingMetadata(
        embedding_dimension=embedding_dimension,
        model_identifier="model",
        model_revision="revision",
        backend_name="fake",
        device="cpu",
        input_sample_rate_hz=16000,
        input_samples=16000,
        input_duration_seconds=1.0,
        normalized=False,
    )


def _issue() -> EmbeddingIssue:
    return EmbeddingIssue(code="INFERENCE_FAILED", message="safe")


def _invalid_result_kwargs(case_name: str) -> dict[str, object]:
    cases = {
        "valid_without_embedding": {
            "status": EmbeddingStatus.VALID,
            "embedding": None,
            "metadata": _metadata(),
            "errors": (),
        },
        "valid_without_metadata": {
            "status": EmbeddingStatus.VALID,
            "embedding": _embedding(),
            "metadata": None,
            "errors": (),
        },
        "valid_with_errors": {
            "status": EmbeddingStatus.VALID,
            "embedding": _embedding(),
            "metadata": _metadata(),
            "errors": (_issue(),),
        },
        "valid_with_float64_embedding": {
            "status": EmbeddingStatus.VALID,
            "embedding": np.full(192, 0.1, dtype=np.float64),
            "metadata": _metadata(),
            "errors": (),
        },
        "valid_with_two_dimensional_embedding": {
            "status": EmbeddingStatus.VALID,
            "embedding": np.full((1, 192), 0.1, dtype=np.float32),
            "metadata": _metadata(),
            "errors": (),
        },
        "valid_with_wrong_embedding_dimension": {
            "status": EmbeddingStatus.VALID,
            "embedding": np.full(191, 0.1, dtype=np.float32),
            "metadata": _metadata(),
            "errors": (),
        },
        "valid_with_one_dimension_metadata": {
            "status": EmbeddingStatus.VALID,
            "embedding": np.full(1, 0.1, dtype=np.float32),
            "metadata": _metadata(embedding_dimension=1),
            "errors": (),
        },
        "valid_with_193_dimension_embedding": {
            "status": EmbeddingStatus.VALID,
            "embedding": np.full(193, 0.1, dtype=np.float32),
            "metadata": _metadata(),
            "errors": (),
        },
        "valid_with_metadata_dimension_not_192": {
            "status": EmbeddingStatus.VALID,
            "embedding": _embedding(),
            "metadata": _metadata(embedding_dimension=191),
            "errors": (),
        },
        "valid_with_non_finite_embedding": {
            "status": EmbeddingStatus.VALID,
            "embedding": np.full(192, np.nan, dtype=np.float32),
            "metadata": _metadata(),
            "errors": (),
        },
        "invalid_with_embedding": {
            "status": EmbeddingStatus.INVALID,
            "embedding": _embedding(),
            "metadata": None,
            "errors": (_issue(),),
        },
        "invalid_with_metadata": {
            "status": EmbeddingStatus.INVALID,
            "embedding": None,
            "metadata": _metadata(),
            "errors": (_issue(),),
        },
        "invalid_without_errors": {
            "status": EmbeddingStatus.INVALID,
            "embedding": None,
            "metadata": None,
            "errors": (),
        },
        "invalid_with_two_errors": {
            "status": EmbeddingStatus.INVALID,
            "embedding": None,
            "metadata": None,
            "errors": (_issue(), _issue()),
        },
    }
    return cases[case_name]
