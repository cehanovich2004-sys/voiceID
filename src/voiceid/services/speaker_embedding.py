"""Application use case for Phase 4 speaker embedding extraction."""

from __future__ import annotations

import math

import numpy as np

from voiceid.audio.preprocessing import (
    PREPROCESSING_CONTRACT_VERSION,
    PreprocessedAudioResult,
    PreprocessingStatus,
)
from voiceid.embeddings.backends.base import EmbeddingBackend
from voiceid.embeddings.contracts import (
    EMBEDDING_CONTRACT_VERSION,
    EmbeddingErrorCode,
    EmbeddingMetadata,
    EmbeddingStatus,
    EmbeddingVector,
    SpeakerEmbeddingResult,
    build_invalid_embedding_result,
)
from voiceid.embeddings.loader import EmbeddingModelError, EmbeddingModelLoader
from voiceid.embeddings.policy import SpeakerEmbeddingPolicy


class SpeakerEmbeddingService:
    """Extract speaker embeddings from Phase 3 preprocessed audio."""

    def __init__(
        self,
        *,
        loader: EmbeddingModelLoader,
        policy: SpeakerEmbeddingPolicy | None = None,
    ) -> None:
        self._loader = loader
        self._policy = policy or SpeakerEmbeddingPolicy()

    def embed(
        self,
        preprocessed_audio: PreprocessedAudioResult,
    ) -> SpeakerEmbeddingResult:
        """Return a validated speaker embedding result."""

        try:
            waveform = self._validate_preprocessed_audio(preprocessed_audio)
            backend = self._loader.get_backend()
            embedding = backend.embed(
                waveform.copy(),
                self._policy.expected_sample_rate_hz,
            )
            return self._build_valid_result(
                embedding=embedding,
                backend=backend,
                input_samples=int(waveform.shape[0]),
            )
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except MemoryError:
            return _invalid_result(EmbeddingErrorCode.MEMORY_LIMIT_EXCEEDED)
        except EmbeddingModelError as exc:
            return _invalid_result(exc.code)
        except _EmbeddingInputError as exc:
            return _invalid_result(exc.code)
        except _EmbeddingOutputError as exc:
            return _invalid_result(exc.code)
        except Exception:
            return _invalid_result(EmbeddingErrorCode.INFERENCE_FAILED)

    def _validate_preprocessed_audio(
        self,
        preprocessed_audio: PreprocessedAudioResult,
    ) -> EmbeddingVector:
        if not isinstance(preprocessed_audio, PreprocessedAudioResult):
            raise _EmbeddingInputError(EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO)
        if (
            preprocessed_audio.status != PreprocessingStatus.VALID
            or not preprocessed_audio.is_valid
            or preprocessed_audio.errors != ()
            or preprocessed_audio.waveform is None
            or preprocessed_audio.metadata is None
        ):
            raise _EmbeddingInputError(EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO)

        metadata = preprocessed_audio.metadata
        if metadata.output_channels != 1:
            raise _EmbeddingInputError(EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO)
        if metadata.output_sample_rate_hz != self._policy.expected_sample_rate_hz:
            raise _EmbeddingInputError(EmbeddingErrorCode.UNSUPPORTED_SAMPLE_RATE)
        if metadata.source_sample_rate_hz <= 0 or metadata.source_channels <= 0:
            raise _EmbeddingInputError(EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO)

        waveform = preprocessed_audio.waveform
        if waveform.dtype != np.float32:
            raise _EmbeddingInputError(EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO)
        if waveform.ndim != 1:
            raise _EmbeddingInputError(EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO)
        if waveform.size == 0:
            raise _EmbeddingInputError(EmbeddingErrorCode.EMPTY_WAVEFORM)
        if metadata.output_samples <= 0 or metadata.output_duration_seconds <= 0:
            raise _EmbeddingInputError(EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO)
        if metadata.source_duration_seconds <= 0:
            raise _EmbeddingInputError(EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO)
        if int(metadata.output_samples) != int(waveform.shape[0]):
            raise _EmbeddingInputError(EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO)

        expected_duration = waveform.shape[0] / self._policy.expected_sample_rate_hz
        if not math.isclose(
            metadata.output_duration_seconds,
            round(expected_duration, self._policy.public_float_decimals),
            rel_tol=0.0,
            abs_tol=10**-self._policy.public_float_decimals,
        ):
            raise _EmbeddingInputError(EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO)

        if not np.all(np.isfinite(waveform)):
            raise _EmbeddingInputError(EmbeddingErrorCode.NON_FINITE_WAVEFORM)
        if bool(np.any((waveform < -1.0) | (waveform > 1.0))):
            raise _EmbeddingInputError(EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO)

        rms = _rms_float64(waveform)
        if rms <= self._policy.zero_or_near_zero_rms_threshold:
            raise _EmbeddingInputError(EmbeddingErrorCode.ZERO_OR_NEAR_ZERO_WAVEFORM)

        return waveform

    def _build_valid_result(
        self,
        *,
        embedding: EmbeddingVector,
        backend: EmbeddingBackend,
        input_samples: int,
    ) -> SpeakerEmbeddingResult:
        checked = _validate_embedding(
            embedding,
            expected_dimension=self._policy.expected_embedding_dimension,
        )
        metadata = EmbeddingMetadata(
            embedding_dimension=self._policy.expected_embedding_dimension,
            model_identifier=backend.model_identifier,
            model_revision=backend.model_revision,
            backend_name=backend.backend_name,
            backend_version=backend.backend_version,
            preprocessing_contract_version=PREPROCESSING_CONTRACT_VERSION,
            embedding_contract_version=EMBEDDING_CONTRACT_VERSION,
            device=backend.device,
            input_sample_rate_hz=self._policy.expected_sample_rate_hz,
            input_samples=input_samples,
            input_duration_seconds=round(
                input_samples / self._policy.expected_sample_rate_hz,
                self._policy.public_float_decimals,
            ),
            normalized=False,
        )
        return SpeakerEmbeddingResult(
            status=EmbeddingStatus.VALID,
            embedding=checked,
            metadata=metadata,
            errors=(),
        )


class _EmbeddingInputError(Exception):
    def __init__(self, code: EmbeddingErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


class _EmbeddingOutputError(Exception):
    def __init__(self, code: EmbeddingErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


def _validate_embedding(
    embedding: EmbeddingVector,
    *,
    expected_dimension: int,
) -> EmbeddingVector:
    if not isinstance(embedding, np.ndarray):
        raise _EmbeddingOutputError(EmbeddingErrorCode.INVALID_EMBEDDING_DTYPE)
    if embedding.dtype != np.float32:
        raise _EmbeddingOutputError(EmbeddingErrorCode.INVALID_EMBEDDING_DTYPE)
    if embedding.ndim != 1 or embedding.shape != (expected_dimension,):
        raise _EmbeddingOutputError(EmbeddingErrorCode.INVALID_EMBEDDING_SHAPE)
    if not np.all(np.isfinite(embedding)):
        raise _EmbeddingOutputError(EmbeddingErrorCode.NON_FINITE_EMBEDDING)
    return embedding.copy()


def _rms_float64(waveform: EmbeddingVector) -> float:
    squared = np.square(waveform.astype(np.float64, copy=False))
    return float(np.sqrt(np.mean(squared, dtype=np.float64)))


def _invalid_result(code: EmbeddingErrorCode) -> SpeakerEmbeddingResult:
    messages = {
        EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO: (
            "The preprocessed audio is not valid for embedding extraction."
        ),
        EmbeddingErrorCode.UNSUPPORTED_SAMPLE_RATE: (
            "The preprocessed audio sample rate is not supported."
        ),
        EmbeddingErrorCode.EMPTY_WAVEFORM: "The preprocessed waveform is empty.",
        EmbeddingErrorCode.NON_FINITE_WAVEFORM: (
            "The preprocessed waveform contains non-finite values."
        ),
        EmbeddingErrorCode.ZERO_OR_NEAR_ZERO_WAVEFORM: (
            "The preprocessed waveform is zero or near-zero."
        ),
        EmbeddingErrorCode.MODEL_NOT_LOADED: "The embedding model is not loaded.",
        EmbeddingErrorCode.MODEL_LOAD_FAILED: (
            "The embedding model could not be loaded safely."
        ),
        EmbeddingErrorCode.MODEL_CACHE_MISSING: (
            "The embedding model cache is missing."
        ),
        EmbeddingErrorCode.MODEL_CACHE_CORRUPTED: (
            "The embedding model cache is incomplete or corrupted."
        ),
        EmbeddingErrorCode.INFERENCE_FAILED: (
            "Speaker embedding inference failed safely."
        ),
        EmbeddingErrorCode.INVALID_EMBEDDING_SHAPE: (
            "The embedding backend returned an invalid embedding shape."
        ),
        EmbeddingErrorCode.INVALID_EMBEDDING_DTYPE: (
            "The embedding backend returned an invalid embedding dtype."
        ),
        EmbeddingErrorCode.NON_FINITE_EMBEDDING: (
            "The embedding backend returned non-finite values."
        ),
        EmbeddingErrorCode.MEMORY_LIMIT_EXCEEDED: (
            "Speaker embedding extraction exceeded available memory."
        ),
    }
    return build_invalid_embedding_result(code=code, message=messages[code])
