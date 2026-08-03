"""Embedding validation policy for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

SPEECHBRAIN_ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
SPEECHBRAIN_ECAPA_MODEL_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
SPEECHBRAIN_ECAPA_EMBEDDING_DIMENSION = 192
SPEECHBRAIN_ECAPA_BACKEND_VERSION: Final = "speechbrain-ecapa-adapter-v1"
TARGET_EMBEDDING_SAMPLE_RATE_HZ = 16000
PUBLIC_FLOAT_DECIMALS = 6
ZERO_OR_NEAR_ZERO_RMS_THRESHOLD = 1e-8


@dataclass(frozen=True, slots=True)
class SpeakerEmbeddingPolicy:
    """Immutable Phase 4 embedding validation policy.

    The RMS threshold is a deterministic degenerate-signal guard. It is not VAD
    and does not decide whether the input contains enough speech.
    """

    expected_sample_rate_hz: int = TARGET_EMBEDDING_SAMPLE_RATE_HZ
    expected_embedding_dimension: int = SPEECHBRAIN_ECAPA_EMBEDDING_DIMENSION
    zero_or_near_zero_rms_threshold: float = ZERO_OR_NEAR_ZERO_RMS_THRESHOLD
    public_float_decimals: int = PUBLIC_FLOAT_DECIMALS
