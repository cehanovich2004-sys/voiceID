"""Tests for speaker embedding public contracts."""

from __future__ import annotations

import numpy as np

from voiceid.embeddings.contracts import (
    EmbeddingErrorCode,
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
