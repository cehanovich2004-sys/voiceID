"""Backend protocol for speaker embedding extraction."""

from __future__ import annotations

from typing import Protocol

from voiceid.embeddings.contracts import EmbeddingVector


class EmbeddingBackend(Protocol):
    """Library-independent speaker embedding backend."""

    @property
    def backend_name(self) -> str:
        """Return a stable backend name."""

    @property
    def backend_version(self) -> str:
        """Return the stable backend adapter contract version."""

    @property
    def model_identifier(self) -> str:
        """Return the pinned model identifier."""

    @property
    def model_revision(self) -> str:
        """Return the pinned model revision."""

    @property
    def device(self) -> str:
        """Return the inference device."""

    @property
    def embedding_dimension(self) -> int:
        """Return the expected embedding dimension."""

    def embed(self, waveform: EmbeddingVector, sample_rate_hz: int) -> EmbeddingVector:
        """Extract a speaker embedding from a preprocessed mono waveform."""
