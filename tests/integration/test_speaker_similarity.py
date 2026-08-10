"""Integration checks for the core-only Phase 5A public API."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from voiceid.audio.preprocessing import PREPROCESSING_CONTRACT_VERSION
from voiceid.embeddings.contracts import (
    EMBEDDING_CONTRACT_VERSION,
    EmbeddingMetadata,
    EmbeddingStatus,
    SpeakerEmbeddingResult,
)
from voiceid.services import compare_speaker_embeddings
from voiceid.similarity import SimilarityStatus


def test_two_fake_embedding_results_compare_without_model_or_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    def fail_loader(*args: object, **kwargs: object) -> None:
        raise AssertionError("model loader must not be called")

    from voiceid.embeddings.loader import EmbeddingModelLoader

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(EmbeddingModelLoader, "get_backend", fail_loader)
    before = set(os.listdir(tmp_path))

    result = compare_speaker_embeddings(_result(0), _result(0))

    assert result.status == SimilarityStatus.VALID
    assert result.similarity == pytest.approx(1.0, abs=1e-7)
    assert set(os.listdir(tmp_path)) == before


def test_installed_public_import_does_not_require_ml_dependencies(
    tmp_path: Path,
) -> None:
    script = """
import importlib.abc
import sys

class BlockMlImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'torch', 'torchaudio', 'speechbrain'}:
            raise ImportError(f'blocked optional ML dependency: {fullname}')
        return None

sys.meta_path.insert(0, BlockMlImports())
from voiceid.services import compare_speaker_embeddings
from voiceid.similarity import SpeakerSimilarityResult

assert callable(compare_speaker_embeddings)
assert SpeakerSimilarityResult.__name__ == 'SpeakerSimilarityResult'
assert not {'torch', 'torchaudio', 'speechbrain'} & set(sys.modules)
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _result(index: int) -> SpeakerEmbeddingResult:
    embedding = np.zeros(192, dtype=np.float32)
    embedding[index] = np.float32(1.0)
    return SpeakerEmbeddingResult(
        status=EmbeddingStatus.VALID,
        embedding=embedding,
        metadata=EmbeddingMetadata(
            embedding_dimension=192,
            model_identifier="model",
            model_revision="revision",
            backend_name="fake",
            backend_version="fake-backend-v1",
            preprocessing_contract_version=PREPROCESSING_CONTRACT_VERSION,
            embedding_contract_version=EMBEDDING_CONTRACT_VERSION,
            device="cpu",
            input_sample_rate_hz=16000,
            input_samples=16000,
            input_duration_seconds=1.0,
            normalized=False,
        ),
    )
