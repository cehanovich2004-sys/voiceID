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
            "requires at least one error",
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


def _metadata() -> EmbeddingMetadata:
    return EmbeddingMetadata(
        embedding_dimension=192,
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
    }
    return cases[case_name]
