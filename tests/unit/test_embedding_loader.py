"""Tests for embedding model loader abstractions."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.test_speaker_embedding_service import FakeBackend
from voiceid.embeddings.backends.base import EmbeddingBackend
from voiceid.embeddings.backends.speechbrain_ecapa import (
    REQUIRED_SPEECHBRAIN_ECAPA_FILES,
    SpeechBrainEcapaBackendFactory,
    bootstrap_speechbrain_ecapa_cache,
    default_speechbrain_ecapa_config,
)
from voiceid.embeddings.contracts import EmbeddingErrorCode
from voiceid.embeddings.loader import (
    EmbeddingBackendFactory,
    EmbeddingModelError,
    EmbeddingModelLoader,
)


class CountingFactory(EmbeddingBackendFactory):
    def __init__(self) -> None:
        self.calls = 0
        self.backend = FakeBackend()

    def load(self) -> EmbeddingBackend:
        self.calls += 1
        return self.backend


def test_embedding_model_loader_loads_once() -> None:
    factory = CountingFactory()
    loader = EmbeddingModelLoader(factory)

    assert loader.get_backend() is loader.get_backend()
    assert factory.calls == 1


def test_offline_bootstrap_is_rejected_without_download() -> None:
    config = default_speechbrain_ecapa_config(
        cache_dir=Path("/safe/nonexistent"),
        offline=True,
    )

    with pytest.raises(EmbeddingModelError) as exc_info:
        bootstrap_speechbrain_ecapa_cache(config)

    assert exc_info.value.code == EmbeddingErrorCode.MODEL_CACHE_MISSING


def test_required_speechbrain_snapshot_manifest_is_minimal() -> None:
    assert set(REQUIRED_SPEECHBRAIN_ECAPA_FILES) == {
        "hyperparams.yaml",
        "embedding_model.ckpt",
        "mean_var_norm_emb.ckpt",
        "classifier.ckpt",
        "label_encoder.txt",
        "config.json",
    }
    assert not any(
        name.endswith((".wav", ".flac", ".mp3"))
        for name in REQUIRED_SPEECHBRAIN_ECAPA_FILES
    )


def test_offline_incomplete_cache_is_controlled(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "speechbrain_ecapa_snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "hyperparams.yaml").write_text("placeholder", encoding="utf-8")
    config = default_speechbrain_ecapa_config(cache_dir=tmp_path, offline=True)

    with pytest.raises(EmbeddingModelError) as exc_info:
        SpeechBrainEcapaBackendFactory(config).load()

    assert exc_info.value.code == EmbeddingErrorCode.MODEL_CACHE_CORRUPTED
