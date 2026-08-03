"""Unit tests for the SpeechBrain ECAPA backend adapter boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from voiceid.embeddings.backends.speechbrain_ecapa import (
    SpeechBrainEcapaBackend,
    _extract_speechbrain_embedding,
)
from voiceid.embeddings.contracts import EmbeddingErrorCode
from voiceid.embeddings.loader import EmbeddingModelError
from voiceid.embeddings.policy import SPEECHBRAIN_ECAPA_BACKEND_VERSION


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


def test_speechbrain_backend_version_is_static_without_ml_import_or_network() -> None:
    backend = SpeechBrainEcapaBackend(classifier=object())

    assert backend.backend_version == SPEECHBRAIN_ECAPA_BACKEND_VERSION
    assert backend.backend_version == "speechbrain-ecapa-adapter-v1"


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 192),
        (1, 192),
    ],
)
def test_speechbrain_singleton_shapes_are_squeezed_to_public_embedding(
    shape: tuple[int, ...],
) -> None:
    output = _tensor(np.full(shape, 0.1, dtype=np.float32))

    embedding = _extract_speechbrain_embedding(output)

    assert embedding.shape == (192,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()


@pytest.mark.parametrize(
    "shape",
    [
        (2, 96),
        (192, 1),
        (1, 2, 96),
        (1, 1, 191),
        (1, 193),
        (192,),
    ],
)
def test_speechbrain_unexpected_shapes_are_rejected(shape: tuple[int, ...]) -> None:
    output = _tensor(np.full(shape, 0.1, dtype=np.float32))

    with pytest.raises(EmbeddingModelError) as exc_info:
        _extract_speechbrain_embedding(output)

    assert exc_info.value.code == EmbeddingErrorCode.INVALID_EMBEDDING_SHAPE


def _tensor(array: np.ndarray) -> Any:
    return _FakeTensor(array)
