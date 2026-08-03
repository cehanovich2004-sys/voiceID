"""Tests for speaker embedding validation policy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from voiceid.embeddings.policy import (
    SPEECHBRAIN_ECAPA_EMBEDDING_DIMENSION,
    SPEECHBRAIN_ECAPA_MODEL_ID,
    SPEECHBRAIN_ECAPA_MODEL_REVISION,
    TARGET_EMBEDDING_SAMPLE_RATE_HZ,
    ZERO_OR_NEAR_ZERO_RMS_THRESHOLD,
    SpeakerEmbeddingPolicy,
)


def test_default_embedding_policy_values_are_pinned() -> None:
    policy = SpeakerEmbeddingPolicy()

    assert policy.expected_sample_rate_hz == TARGET_EMBEDDING_SAMPLE_RATE_HZ
    assert policy.expected_embedding_dimension == SPEECHBRAIN_ECAPA_EMBEDDING_DIMENSION
    assert policy.zero_or_near_zero_rms_threshold == ZERO_OR_NEAR_ZERO_RMS_THRESHOLD
    assert SPEECHBRAIN_ECAPA_MODEL_ID == "speechbrain/spkrec-ecapa-voxceleb"
    assert (
        SPEECHBRAIN_ECAPA_MODEL_REVISION == "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
    )


def test_embedding_policy_is_immutable() -> None:
    policy = SpeakerEmbeddingPolicy()

    with pytest.raises(FrozenInstanceError):
        policy.expected_embedding_dimension = 256  # type: ignore[misc]
