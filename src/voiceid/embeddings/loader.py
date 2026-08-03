"""Thread-safe model loader abstraction for embedding backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from voiceid.embeddings.backends.base import EmbeddingBackend
from voiceid.embeddings.contracts import EmbeddingErrorCode


class EmbeddingModelError(Exception):
    """Internal model loading error with a stable public code."""

    def __init__(self, code: EmbeddingErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


@dataclass(frozen=True, slots=True)
class EmbeddingModelConfig:
    """Configuration for a pinned embedding backend."""

    model_identifier: str
    model_revision: str
    cache_dir: Path
    offline: bool = True
    device: str = "cpu"


class EmbeddingModelLoader:
    """Load an embedding backend at most once per loader instance."""

    def __init__(self, backend_factory: EmbeddingBackendFactory) -> None:
        self._backend_factory = backend_factory
        self._backend: EmbeddingBackend | None = None
        self._lock = Lock()

    def get_backend(self) -> EmbeddingBackend:
        """Return the cached backend, loading it on first access."""

        backend = self._backend
        if backend is not None:
            return backend

        with self._lock:
            backend = self._backend
            if backend is None:
                try:
                    backend = self._backend_factory.load()
                except KeyboardInterrupt:
                    raise
                except SystemExit:
                    raise
                except MemoryError:
                    raise
                except EmbeddingModelError:
                    raise
                except Exception as exc:
                    raise EmbeddingModelError(
                        EmbeddingErrorCode.MODEL_LOAD_FAILED
                    ) from exc
                self._backend = backend
            return backend


class EmbeddingBackendFactory:
    """Factory interface used by the loader."""

    def load(self) -> EmbeddingBackend:
        """Load and return an embedding backend."""

        raise NotImplementedError
