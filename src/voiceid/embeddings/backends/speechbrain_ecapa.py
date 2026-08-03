"""SpeechBrain ECAPA-TDNN speaker embedding backend."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np

from voiceid.embeddings.backends.base import EmbeddingBackend
from voiceid.embeddings.contracts import EmbeddingErrorCode, EmbeddingVector
from voiceid.embeddings.loader import (
    EmbeddingBackendFactory,
    EmbeddingModelConfig,
    EmbeddingModelError,
)
from voiceid.embeddings.policy import (
    SPEECHBRAIN_ECAPA_EMBEDDING_DIMENSION,
    SPEECHBRAIN_ECAPA_MODEL_ID,
    SPEECHBRAIN_ECAPA_MODEL_REVISION,
    TARGET_EMBEDDING_SAMPLE_RATE_HZ,
)

REQUIRED_SPEECHBRAIN_ECAPA_FILES = (
    "hyperparams.yaml",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "classifier.ckpt",
    "label_encoder.txt",
    "config.json",
)


class SpeechBrainEcapaBackend(EmbeddingBackend):
    """Adapter around SpeechBrain ECAPA-TDNN tensor-only embedding extraction."""

    def __init__(
        self,
        *,
        classifier: Any,
        model_identifier: str = SPEECHBRAIN_ECAPA_MODEL_ID,
        model_revision: str = SPEECHBRAIN_ECAPA_MODEL_REVISION,
        device: str = "cpu",
    ) -> None:
        self._classifier = classifier
        self._model_identifier = model_identifier
        self._model_revision = model_revision
        self._device = device

    @property
    def backend_name(self) -> str:
        """Return a stable backend name."""

        return "speechbrain-ecapa-tdnn"

    @property
    def model_identifier(self) -> str:
        """Return the pinned model identifier."""

        return self._model_identifier

    @property
    def model_revision(self) -> str:
        """Return the pinned model revision."""

        return self._model_revision

    @property
    def device(self) -> str:
        """Return the inference device."""

        return self._device

    @property
    def embedding_dimension(self) -> int:
        """Return the expected embedding dimension."""

        return SPEECHBRAIN_ECAPA_EMBEDDING_DIMENSION

    def embed(self, waveform: EmbeddingVector, sample_rate_hz: int) -> EmbeddingVector:
        """Extract a raw, unnormalized ECAPA embedding from a mono 16 kHz waveform."""

        if sample_rate_hz != TARGET_EMBEDDING_SAMPLE_RATE_HZ:
            raise EmbeddingModelError(EmbeddingErrorCode.UNSUPPORTED_SAMPLE_RATE)

        try:
            torch = import_module("torch")
        except ImportError as exc:
            raise EmbeddingModelError(EmbeddingErrorCode.MODEL_NOT_LOADED) from exc

        try:
            tensor = torch.from_numpy(
                waveform.astype(np.float32, copy=False)
            ).unsqueeze(0)
            with torch.inference_mode():
                output = self._classifier.encode_batch(tensor, normalize=False)
            embedding = output.detach().cpu().numpy().astype(np.float32, copy=False)
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except MemoryError:
            raise
        except Exception as exc:
            raise EmbeddingModelError(EmbeddingErrorCode.INFERENCE_FAILED) from exc

        return cast(
            EmbeddingVector, np.asarray(embedding.reshape(-1), dtype=np.float32)
        )


class SpeechBrainEcapaBackendFactory(EmbeddingBackendFactory):
    """Factory for pinned SpeechBrain ECAPA-TDNN backends."""

    def __init__(self, config: EmbeddingModelConfig) -> None:
        self._config = config

    def load(self) -> EmbeddingBackend:
        """Load the pinned SpeechBrain backend."""

        try:
            snapshot_path = _prepare_snapshot(self._config)
            _validate_snapshot(snapshot_path)
            classifier = _load_classifier(snapshot_path, self._config)
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except MemoryError:
            raise
        except EmbeddingModelError:
            raise
        except Exception as exc:
            raise EmbeddingModelError(EmbeddingErrorCode.MODEL_LOAD_FAILED) from exc

        return SpeechBrainEcapaBackend(
            classifier=classifier,
            model_identifier=self._config.model_identifier,
            model_revision=self._config.model_revision,
            device=self._config.device,
        )


def bootstrap_speechbrain_ecapa_cache(config: EmbeddingModelConfig) -> None:
    """Explicitly download the pinned minimal SpeechBrain ECAPA snapshot."""

    if config.offline:
        raise EmbeddingModelError(EmbeddingErrorCode.MODEL_CACHE_MISSING)
    snapshot_path = _prepare_snapshot(config)
    _validate_snapshot(snapshot_path)


def default_speechbrain_ecapa_config(
    *,
    cache_dir: str | Path,
    offline: bool = True,
    device: str = "cpu",
) -> EmbeddingModelConfig:
    """Build the default pinned SpeechBrain ECAPA config."""

    return EmbeddingModelConfig(
        model_identifier=SPEECHBRAIN_ECAPA_MODEL_ID,
        model_revision=SPEECHBRAIN_ECAPA_MODEL_REVISION,
        cache_dir=Path(cache_dir),
        offline=offline,
        device=device,
    )


def _prepare_snapshot(config: EmbeddingModelConfig) -> Path:
    if config.model_identifier != SPEECHBRAIN_ECAPA_MODEL_ID:
        raise EmbeddingModelError(EmbeddingErrorCode.MODEL_LOAD_FAILED)
    if config.model_revision != SPEECHBRAIN_ECAPA_MODEL_REVISION:
        raise EmbeddingModelError(EmbeddingErrorCode.MODEL_LOAD_FAILED)
    if config.device != "cpu":
        raise EmbeddingModelError(EmbeddingErrorCode.MODEL_LOAD_FAILED)

    snapshot_dir = config.cache_dir / "speechbrain_ecapa_snapshot"
    if config.offline:
        if not snapshot_dir.is_dir():
            raise EmbeddingModelError(EmbeddingErrorCode.MODEL_CACHE_MISSING)
        return snapshot_dir

    try:
        huggingface_hub = import_module("huggingface_hub")
    except ImportError as exc:
        raise EmbeddingModelError(EmbeddingErrorCode.MODEL_NOT_LOADED) from exc

    try:
        downloaded = huggingface_hub.snapshot_download(
            repo_id=config.model_identifier,
            revision=config.model_revision,
            local_dir=snapshot_dir,
            allow_patterns=list(REQUIRED_SPEECHBRAIN_ECAPA_FILES),
            local_files_only=False,
        )
    except Exception as exc:
        raise EmbeddingModelError(EmbeddingErrorCode.MODEL_LOAD_FAILED) from exc
    return Path(downloaded)


def _validate_snapshot(snapshot_path: Path) -> None:
    if not snapshot_path.is_dir():
        raise EmbeddingModelError(EmbeddingErrorCode.MODEL_CACHE_MISSING)
    missing = [
        file_name
        for file_name in REQUIRED_SPEECHBRAIN_ECAPA_FILES
        if not (snapshot_path / file_name).is_file()
    ]
    if missing:
        raise EmbeddingModelError(EmbeddingErrorCode.MODEL_CACHE_CORRUPTED)


def _load_classifier(
    snapshot_path: Path,
    config: EmbeddingModelConfig,
) -> Any:
    try:
        speaker_module = import_module("speechbrain.inference.speaker")
        fetching_module = import_module("speechbrain.utils.fetching")
    except ImportError as exc:
        raise EmbeddingModelError(EmbeddingErrorCode.MODEL_NOT_LOADED) from exc

    savedir = config.cache_dir / "speechbrain_ecapa_savedir"
    try:
        classifier = speaker_module.EncoderClassifier.from_hparams(
            source=str(snapshot_path),
            savedir=str(savedir),
            run_opts={"device": config.device},
            local_strategy=fetching_module.LocalStrategy.COPY,
        )
        _set_eval(classifier)
    except Exception as exc:
        raise EmbeddingModelError(EmbeddingErrorCode.MODEL_LOAD_FAILED) from exc
    return classifier


def _set_eval(classifier: Any) -> None:
    modules = getattr(classifier, "mods", None)
    if modules is not None and hasattr(modules, "eval"):
        modules.eval()
