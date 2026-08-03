"""Tests for the Phase 4 speaker embedding application service."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

from voiceid.audio.preprocessing import (
    PreprocessedAudioMetadata,
    PreprocessedAudioResult,
    PreprocessingIssue,
    PreprocessingStatus,
)
from voiceid.embeddings.backends.base import EmbeddingBackend
from voiceid.embeddings.contracts import EmbeddingErrorCode, EmbeddingVector
from voiceid.embeddings.loader import (
    EmbeddingBackendFactory,
    EmbeddingModelError,
    EmbeddingModelLoader,
)
from voiceid.embeddings.policy import SpeakerEmbeddingPolicy
from voiceid.services.speaker_embedding import SpeakerEmbeddingService


class FakeBackend(EmbeddingBackend):
    def __init__(
        self,
        *,
        embedding: EmbeddingVector | None = None,
        exception: BaseException | None = None,
    ) -> None:
        self.calls = 0
        self.received_pathname = False
        self.received_waveforms: list[EmbeddingVector] = []
        self._embedding = (
            embedding if embedding is not None else np.full(192, 0.25, dtype=np.float32)
        )
        self._exception = exception

    @property
    def backend_name(self) -> str:
        return "fake"

    @property
    def model_identifier(self) -> str:
        return "fake-model"

    @property
    def model_revision(self) -> str:
        return "fake-revision"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def embedding_dimension(self) -> int:
        return 192

    def embed(self, waveform: EmbeddingVector, sample_rate_hz: int) -> EmbeddingVector:
        self.calls += 1
        self.received_pathname = isinstance(waveform, (str, Path))
        self.received_waveforms.append(waveform.copy())
        if self._exception is not None:
            raise self._exception
        assert sample_rate_hz == 16000
        return self._embedding.copy()


class FakeFactory(EmbeddingBackendFactory):
    def __init__(
        self,
        backend: FakeBackend | None = None,
        exception: BaseException | None = None,
    ) -> None:
        self.calls = 0
        self._backend = backend or FakeBackend()
        self._exception = exception

    def load(self) -> EmbeddingBackend:
        self.calls += 1
        if self._exception is not None:
            raise self._exception
        return self._backend


def test_valid_phase3_result_produces_valid_embedding() -> None:
    backend = FakeBackend()
    result = _service(backend).embed(_valid_audio_result())

    assert result.is_valid
    assert result.embedding is not None
    assert result.embedding.dtype == np.float32
    assert result.embedding.shape == (192,)
    assert np.isfinite(result.embedding).all()
    assert result.metadata is not None
    assert result.metadata.embedding_dimension == 192
    assert result.metadata.normalized is False
    assert backend.calls == 1


def test_invalid_phase3_result_does_not_call_backend() -> None:
    backend = FakeBackend()
    invalid = PreprocessedAudioResult(
        status=PreprocessingStatus.INVALID,
        file_name="safe.wav",
        waveform=None,
        metadata=None,
        errors=(PreprocessingIssue(code="INVALID_INPUT", message="safe"),),
    )

    result = _service(backend).embed(invalid)

    assert _code(result) == EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO
    assert backend.calls == 0


@pytest.mark.parametrize(
    ("case_name", "expected"),
    [
        ("unsupported_sample_rate", EmbeddingErrorCode.UNSUPPORTED_SAMPLE_RATE),
        ("empty_waveform", EmbeddingErrorCode.EMPTY_WAVEFORM),
        ("nan_waveform", EmbeddingErrorCode.NON_FINITE_WAVEFORM),
        ("inf_waveform", EmbeddingErrorCode.NON_FINITE_WAVEFORM),
        ("zero_waveform", EmbeddingErrorCode.ZERO_OR_NEAR_ZERO_WAVEFORM),
        ("near_zero_waveform", EmbeddingErrorCode.ZERO_OR_NEAR_ZERO_WAVEFORM),
        ("out_of_range_waveform", EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO),
        ("two_dimensional_waveform", EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO),
        ("float64_waveform", EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO),
        ("metadata_sample_mismatch", EmbeddingErrorCode.INVALID_PREPROCESSED_AUDIO),
    ],
)
def test_input_validation_rejects_invalid_phase3_contract(
    case_name: str,
    expected: EmbeddingErrorCode,
) -> None:
    backend = FakeBackend()
    audio = _invalid_audio_case(case_name)

    result = _service(backend).embed(audio)

    assert _code(result) == expected
    assert result.embedding is None
    assert result.metadata is None
    assert backend.calls == 0


@pytest.mark.parametrize(
    ("amplitude", "expected_valid"),
    [
        (0.999e-8, False),
        (1.0e-8, False),
        (1.001e-8, True),
    ],
)
def test_zero_near_zero_rms_boundary(
    amplitude: float,
    expected_valid: bool,
) -> None:
    audio = _valid_audio_result(
        waveform=np.full(16000, amplitude, dtype=np.float32),
        metadata=_metadata(output_duration_seconds=1.0),
    )

    result = _service(FakeBackend()).embed(audio)

    assert result.is_valid is expected_valid
    if not expected_valid:
        assert _code(result) == EmbeddingErrorCode.ZERO_OR_NEAR_ZERO_WAVEFORM


def test_loader_called_once_and_backend_reused() -> None:
    backend = FakeBackend()
    factory = FakeFactory(backend)
    service = SpeakerEmbeddingService(loader=EmbeddingModelLoader(factory))

    result1 = service.embed(_valid_audio_result())
    result2 = service.embed(_valid_audio_result())

    assert result1.is_valid and result2.is_valid
    assert factory.calls == 1
    assert backend.calls == 2


def test_deterministic_fake_embedding_for_repeated_input() -> None:
    service = _service(FakeBackend())
    audio = _valid_audio_result()

    result1 = service.embed(audio)
    result2 = service.embed(audio)

    assert result1.embedding is not None
    assert result2.embedding is not None
    assert np.array_equal(result1.embedding, result2.embedding)


@pytest.mark.parametrize(
    ("embedding", "expected"),
    [
        (
            np.full((1, 192), 0.1, dtype=np.float32),
            EmbeddingErrorCode.INVALID_EMBEDDING_SHAPE,
        ),
        (
            np.full(191, 0.1, dtype=np.float32),
            EmbeddingErrorCode.INVALID_EMBEDDING_SHAPE,
        ),
        (
            np.full(192, 0.1, dtype=np.float64),
            EmbeddingErrorCode.INVALID_EMBEDDING_DTYPE,
        ),
        (
            np.full(192, np.nan, dtype=np.float32),
            EmbeddingErrorCode.NON_FINITE_EMBEDDING,
        ),
    ],
)
def test_embedding_output_validation(
    embedding: EmbeddingVector,
    expected: EmbeddingErrorCode,
) -> None:
    result = _service(FakeBackend(embedding=embedding)).embed(_valid_audio_result())

    assert _code(result) == expected
    assert result.embedding is None
    assert result.metadata is None


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (
            RuntimeError("internal path and token details"),
            EmbeddingErrorCode.INFERENCE_FAILED,
        ),
        (
            EmbeddingModelError(EmbeddingErrorCode.MODEL_CACHE_MISSING),
            EmbeddingErrorCode.MODEL_CACHE_MISSING,
        ),
        (
            EmbeddingModelError(EmbeddingErrorCode.MODEL_CACHE_CORRUPTED),
            EmbeddingErrorCode.MODEL_CACHE_CORRUPTED,
        ),
        (MemoryError("secret"), EmbeddingErrorCode.MEMORY_LIMIT_EXCEEDED),
    ],
)
def test_backend_exceptions_are_sanitized(
    exception: BaseException,
    expected: EmbeddingErrorCode,
) -> None:
    result = _service(FakeBackend(exception=exception)).embed(_valid_audio_result())

    assert _code(result) == expected
    assert "internal path" not in repr(result)
    assert "secret" not in repr(result)


@pytest.mark.parametrize(
    "exception",
    [KeyboardInterrupt(), SystemExit()],
)
def test_keyboard_interrupt_and_system_exit_passthrough(
    exception: BaseException,
) -> None:
    with pytest.raises(type(exception)):
        _service(FakeBackend(exception=exception)).embed(_valid_audio_result())


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (
            EmbeddingModelError(EmbeddingErrorCode.MODEL_LOAD_FAILED),
            EmbeddingErrorCode.MODEL_LOAD_FAILED,
        ),
        (RuntimeError("secret"), EmbeddingErrorCode.MODEL_LOAD_FAILED),
        (MemoryError("secret"), EmbeddingErrorCode.MEMORY_LIMIT_EXCEEDED),
    ],
)
def test_loader_exceptions_are_sanitized(
    exception: BaseException,
    expected: EmbeddingErrorCode,
) -> None:
    factory = FakeFactory(exception=exception)
    service = SpeakerEmbeddingService(loader=EmbeddingModelLoader(factory))

    result = service.embed(_valid_audio_result())

    assert _code(result) == expected
    assert result.embedding is None
    assert result.metadata is None
    assert "secret" not in repr(result)


def test_input_waveform_is_not_modified() -> None:
    waveform = _sine_wave()
    original_bytes = waveform.tobytes()

    result = _service(FakeBackend()).embed(_valid_audio_result(waveform=waveform))

    assert result.is_valid
    assert waveform.tobytes() == original_bytes


def test_backend_does_not_receive_pathname() -> None:
    backend = FakeBackend()

    result = _service(backend).embed(_valid_audio_result())

    assert result.is_valid
    assert backend.received_pathname is False


def test_safe_to_dict_repr_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    embedding = np.arange(192, dtype=np.float32)
    result = _service(FakeBackend(embedding=embedding)).embed(_valid_audio_result())

    public = result.to_dict()
    log_text = caplog.text

    assert "embedding" not in public
    assert "191" not in repr(result)
    assert "191" not in str(result)
    assert "191" not in log_text
    assert "internal path" not in repr(result)


def test_no_audio_temp_model_or_embedding_artifacts_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    result = _service(FakeBackend()).embed(_valid_audio_result())

    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert result.is_valid
    assert after == before


def _service(backend: FakeBackend) -> SpeakerEmbeddingService:
    return SpeakerEmbeddingService(
        loader=EmbeddingModelLoader(FakeFactory(backend)),
        policy=SpeakerEmbeddingPolicy(),
    )


def _valid_audio_result(
    *,
    waveform: np.ndarray | None = None,
    metadata: PreprocessedAudioMetadata | None = None,
) -> PreprocessedAudioResult:
    active_waveform = waveform if waveform is not None else _sine_wave()
    active_metadata = metadata or _metadata(
        output_samples=int(active_waveform.shape[0]),
        output_duration_seconds=round(active_waveform.shape[0] / 16000, 6),
    )
    return PreprocessedAudioResult(
        status=PreprocessingStatus.VALID,
        file_name="safe.wav",
        waveform=active_waveform,
        metadata=active_metadata,
        errors=(),
    )


def _metadata(
    *,
    output_sample_rate_hz: int = 16000,
    output_samples: int = 16000,
    output_duration_seconds: float = 1.0,
) -> PreprocessedAudioMetadata:
    return PreprocessedAudioMetadata(
        source_sample_rate_hz=16000,
        source_channels=1,
        source_duration_seconds=output_duration_seconds,
        output_sample_rate_hz=output_sample_rate_hz,
        output_channels=1,
        output_samples=output_samples,
        output_duration_seconds=output_duration_seconds,
        downmixed_to_mono=False,
        dc_offset_removed=True,
        resampled=False,
        resample_up=1,
        resample_down=1,
        safety_clipped=False,
    )


def _sine_wave(samples: int = 16000) -> EmbeddingVector:
    time = np.arange(samples, dtype=np.float32) / np.float32(16000)
    return np.asarray(0.1 * np.sin(2 * np.pi * 440 * time), dtype=np.float32)


def _code(result: object) -> EmbeddingErrorCode:
    error_code = result.errors[0].code  # type: ignore[attr-defined]
    return EmbeddingErrorCode(error_code)


def _invalid_audio_case(case_name: str) -> PreprocessedAudioResult:
    cases = {
        "unsupported_sample_rate": _valid_audio_result(
            metadata=_metadata(output_sample_rate_hz=8000),
        ),
        "empty_waveform": _valid_audio_result(
            waveform=np.array([], dtype=np.float32),
        ),
        "nan_waveform": _valid_audio_result(
            waveform=np.array([np.nan], dtype=np.float32),
        ),
        "inf_waveform": _valid_audio_result(
            waveform=np.array([np.inf], dtype=np.float32),
        ),
        "zero_waveform": _valid_audio_result(
            waveform=np.zeros(16000, dtype=np.float32),
        ),
        "near_zero_waveform": _valid_audio_result(
            waveform=np.full(16000, 1e-9, dtype=np.float32),
        ),
        "out_of_range_waveform": _valid_audio_result(
            waveform=np.full(16000, 2.0, dtype=np.float32),
        ),
        "two_dimensional_waveform": _valid_audio_result(
            waveform=np.ones((16000, 1), dtype=np.float32),
        ),
        "float64_waveform": _valid_audio_result(
            waveform=np.ones(16000, dtype=np.float64),
        ),
        "metadata_sample_mismatch": _valid_audio_result(
            waveform=np.full(16000, 0.1, dtype=np.float32),
            metadata=_metadata(output_samples=15999),
        ),
    }
    if case_name not in cases:
        _never()
    return cases[case_name]


def _never() -> NoReturn:
    raise AssertionError("unreachable")
