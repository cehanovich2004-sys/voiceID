"""Opt-in real-model smoke tests for SpeechBrain ECAPA backend."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from voiceid.audio.preprocessing import (
    PreprocessedAudioMetadata,
    PreprocessedAudioResult,
    PreprocessingStatus,
)
from voiceid.embeddings.backends.speechbrain_ecapa import (
    SpeechBrainEcapaBackendFactory,
    default_speechbrain_ecapa_config,
)
from voiceid.embeddings.loader import EmbeddingModelLoader
from voiceid.services.speaker_embedding import SpeakerEmbeddingService


@pytest.mark.real_model
def test_speechbrain_ecapa_offline_real_model_smoke(tmp_path: Path) -> None:
    if os.environ.get("VOICEID_RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("VOICEID_RUN_REAL_MODEL_TESTS=1 is required")
    cache_dir_raw = os.environ.get("VOICEID_SPEECHBRAIN_ECAPA_CACHE_DIR")
    if not cache_dir_raw:
        pytest.skip("VOICEID_SPEECHBRAIN_ECAPA_CACHE_DIR is not set")

    config = default_speechbrain_ecapa_config(
        cache_dir=Path(cache_dir_raw),
        offline=True,
    )
    service = SpeakerEmbeddingService(
        loader=EmbeddingModelLoader(SpeechBrainEcapaBackendFactory(config)),
    )
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    audio = _synthetic_audio()

    result1 = service.embed(audio)
    result2 = service.embed(audio)

    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert result1.is_valid
    assert result2.is_valid
    assert result1.embedding is not None
    assert result2.embedding is not None
    assert result1.embedding.shape == (192,)
    assert result1.embedding.dtype == np.float32
    assert np.isfinite(result1.embedding).all()
    assert np.allclose(result1.embedding, result2.embedding, rtol=0.0, atol=1e-6)
    assert after == before


def _synthetic_audio() -> PreprocessedAudioResult:
    sample_rate = 16000
    samples = sample_rate * 2
    time = np.arange(samples, dtype=np.float32) / np.float32(sample_rate)
    waveform = np.asarray(0.1 * np.sin(2 * np.pi * 440 * time), dtype=np.float32)
    return PreprocessedAudioResult(
        status=PreprocessingStatus.VALID,
        file_name="synthetic.wav",
        waveform=waveform,
        metadata=PreprocessedAudioMetadata(
            source_sample_rate_hz=sample_rate,
            source_channels=1,
            source_duration_seconds=2.0,
            output_sample_rate_hz=sample_rate,
            output_channels=1,
            output_samples=samples,
            output_duration_seconds=2.0,
            downmixed_to_mono=False,
            dc_offset_removed=True,
            resampled=False,
            resample_up=1,
            resample_down=1,
            safety_clipped=False,
        ),
        errors=(),
    )
