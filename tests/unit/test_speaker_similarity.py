"""Unit tests for deterministic speaker embedding comparison."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import voiceid.similarity.comparison as comparison_module
from voiceid.embeddings.contracts import (
    EmbeddingErrorCode,
    EmbeddingMetadata,
    EmbeddingStatus,
    SpeakerEmbeddingResult,
    build_invalid_embedding_result,
)
from voiceid.similarity import (
    SimilarityErrorCode,
    SimilarityStatus,
    SpeakerSimilarityResult,
    compare_speaker_embeddings,
)


def _basis(index: int) -> np.ndarray:
    vector = np.zeros(192, dtype=np.float32)
    vector[index] = np.float32(1.0)
    return vector


@pytest.mark.parametrize(
    ("reference_vector", "candidate_vector", "expected"),
    [
        (_basis(0), _basis(0), 1.0),
        (_basis(0), _basis(1), 0.0),
        (_basis(0), -_basis(0), -1.0),
    ],
)
def test_canonical_cosine_cases(
    reference_vector: np.ndarray,
    candidate_vector: np.ndarray,
    expected: float,
) -> None:
    result = compare_speaker_embeddings(
        _result(reference_vector),
        _result(candidate_vector),
    )

    assert result.status == SimilarityStatus.VALID
    assert result.similarity == pytest.approx(expected, abs=1e-7)


def test_close_vectors_have_high_but_not_perfect_similarity() -> None:
    candidate = _basis(0)
    candidate[1] = np.float32(0.1)

    result = compare_speaker_embeddings(_result(_basis(0)), _result(candidate))

    assert result.similarity is not None
    assert 0.99 < result.similarity < 1.0


def test_positive_scaling_does_not_change_cosine_similarity() -> None:
    reference = _basis(0) * np.float32(2.0)
    candidate = _basis(0) * np.float32(7.0)

    result = compare_speaker_embeddings(_result(reference), _result(candidate))

    assert result.similarity == pytest.approx(1.0, abs=1e-7)


def test_valid_result_contains_fixed_safe_comparison_metadata() -> None:
    result = compare_speaker_embeddings(_result(_basis(0)), _result(_basis(0)))

    assert result.metadata is not None
    assert result.metadata.metric == "cosine_similarity"
    assert result.metadata.comparison_version == "1"
    assert result.metadata.embedding_dimension == 192
    assert result.metadata.normalized is False
    assert result.metadata.to_dict() == {
        "metric": "cosine_similarity",
        "comparison_version": "1",
        "embedding_dimension": 192,
        "normalized": False,
    }


@pytest.mark.parametrize(
    "value",
    [0.0, 1e-9, 1e-8],
)
def test_zero_or_near_zero_reference_is_invalid(value: float) -> None:
    vector = np.zeros(192, dtype=np.float32)
    vector[0] = np.float32(value)

    result = compare_speaker_embeddings(_result(vector), _result(_basis(0)))

    _assert_invalid(result, SimilarityErrorCode.ZERO_OR_NEAR_ZERO_EMBEDDING)


def test_norm_above_numeric_guard_is_valid() -> None:
    vector = np.zeros(192, dtype=np.float32)
    vector[0] = np.float32(1.0001e-8)

    result = compare_speaker_embeddings(_result(vector), _result(_basis(0)))

    assert result.status == SimilarityStatus.VALID
    assert result.similarity == pytest.approx(1.0, abs=1e-7)


def test_zero_candidate_is_invalid_after_valid_reference() -> None:
    result = compare_speaker_embeddings(
        _result(_basis(0)),
        _result(np.zeros(192, dtype=np.float32)),
    )

    _assert_invalid(result, SimilarityErrorCode.ZERO_OR_NEAR_ZERO_EMBEDDING)


@pytest.mark.parametrize(
    "bad_embedding",
    [
        np.ones(192, dtype=np.float64),
        np.ones((1, 192), dtype=np.float32),
        np.ones(191, dtype=np.float32),
        np.full(192, np.nan, dtype=np.float32),
        np.full(192, np.inf, dtype=np.float32),
        np.full(192, -np.inf, dtype=np.float32),
    ],
)
def test_invalid_reference_embedding_runtime_invariants(
    bad_embedding: np.ndarray,
) -> None:
    reference = _tamper(_result(_basis(0)), embedding=bad_embedding)

    result = compare_speaker_embeddings(reference, _result(_basis(0)))

    _assert_invalid(result, SimilarityErrorCode.INVALID_EMBEDDING)


def test_metadata_dimension_mismatch_is_invalid_embedding() -> None:
    metadata = replace(_metadata(), embedding_dimension=191)
    reference = _tamper(_result(_basis(0)), metadata=metadata)

    result = compare_speaker_embeddings(reference, _result(_basis(0)))

    _assert_invalid(result, SimilarityErrorCode.INVALID_EMBEDDING)


@pytest.mark.parametrize("argument", ["reference", "candidate"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_samples", 0),
        ("input_samples", -1),
        ("input_samples", True),
        ("input_samples", 16000.0),
        ("input_samples", np.int64(16000)),
        ("input_samples", 10**1000),
        ("input_duration_seconds", 0.0),
        ("input_duration_seconds", -1.0),
        ("input_duration_seconds", float("nan")),
        ("input_duration_seconds", float("inf")),
        ("input_duration_seconds", float("-inf")),
        ("input_duration_seconds", 1),
        ("input_duration_seconds", np.float64(1.0)),
    ],
)
def test_invalid_runtime_input_metadata_is_rejected(
    argument: str,
    field: str,
    value: object,
) -> None:
    reference = _result(_basis(0))
    candidate = _result(_basis(0))
    target = reference if argument == "reference" else candidate
    object.__setattr__(
        target,
        "metadata",
        replace(_metadata(), **{field: value}),
    )

    result = compare_speaker_embeddings(reference, candidate)

    _assert_invalid(result, SimilarityErrorCode.INVALID_EMBEDDING)


@pytest.mark.parametrize("argument", ["reference", "candidate"])
def test_inconsistent_samples_and_duration_are_rejected(argument: str) -> None:
    reference = _result(_basis(0))
    candidate = _result(_basis(0))
    target = reference if argument == "reference" else candidate
    object.__setattr__(
        target,
        "metadata",
        replace(
            _metadata(),
            input_samples=16001,
            input_duration_seconds=1.0,
        ),
    )

    result = compare_speaker_embeddings(reference, candidate)

    _assert_invalid(result, SimilarityErrorCode.INVALID_EMBEDDING)


@pytest.mark.parametrize("argument", ["reference", "candidate"])
def test_writable_embedding_is_rejected_without_mutation(argument: str) -> None:
    reference = _result(_basis(0))
    candidate = _result(_basis(0))
    target = reference if argument == "reference" else candidate
    writable = _basis(0)
    before = writable.tobytes()
    object.__setattr__(target, "embedding", writable)

    result = compare_speaker_embeddings(reference, candidate)

    _assert_invalid(result, SimilarityErrorCode.INVALID_EMBEDDING)
    assert target.embedding is writable
    assert writable.flags.writeable is True
    assert writable.tobytes() == before


def test_invalid_reference_metadata_precedes_candidate_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _tamper(
        _result(_basis(0)),
        metadata=replace(_metadata(), input_samples=0),
    )
    candidate = _tamper(
        _result(_basis(0)),
        metadata=replace(_metadata(), input_duration_seconds=float("nan")),
    )
    original_validate = comparison_module._validate_embedding
    calls: list[SpeakerEmbeddingResult] = []

    def track(result: SpeakerEmbeddingResult) -> object:
        calls.append(result)
        return original_validate(result)

    monkeypatch.setattr(comparison_module, "_validate_embedding", track)

    result = compare_speaker_embeddings(reference, candidate)

    _assert_invalid(result, SimilarityErrorCode.INVALID_EMBEDDING)
    assert len(calls) == 1
    assert calls[0] is reference


@pytest.mark.parametrize(
    ("field", "canary"),
    [
        ("input_samples", 987654321),
        ("input_duration_seconds", 123456.789012),
    ],
)
@pytest.mark.parametrize("argument", ["reference", "candidate"])
def test_invalid_metadata_canary_is_not_exposed(
    argument: str,
    field: str,
    canary: int | float,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reference = _result(_basis(0))
    candidate = _result(_basis(0))
    target = reference if argument == "reference" else candidate
    object.__setattr__(
        target,
        "metadata",
        replace(_metadata(), **{field: canary}),
    )

    result = compare_speaker_embeddings(reference, candidate)

    _assert_invalid(result, SimilarityErrorCode.INVALID_EMBEDDING)
    surfaces = (
        repr(result),
        str(result),
        str(result.to_dict()),
        str(result.errors),
        caplog.text,
    )
    assert all(str(canary) not in surface for surface in surfaces)


@pytest.mark.parametrize("missing_field", ["embedding", "metadata"])
def test_missing_reference_embedding_state_is_rejected(missing_field: str) -> None:
    reference = _result(_basis(0))
    object.__setattr__(reference, missing_field, None)

    result = compare_speaker_embeddings(reference, _result(_basis(0)))

    _assert_invalid(result, SimilarityErrorCode.INVALID_EMBEDDING)


def test_forged_valid_reference_with_errors_is_invalid_reference() -> None:
    reference = _result(_basis(0))
    invalid = build_invalid_embedding_result(
        code=EmbeddingErrorCode.INFERENCE_FAILED,
        message="safe",
    )
    object.__setattr__(reference, "errors", invalid.errors)

    result = compare_speaker_embeddings(reference, _result(_basis(0)))

    _assert_invalid(result, SimilarityErrorCode.INVALID_REFERENCE)


def test_forged_valid_candidate_with_errors_is_invalid_candidate() -> None:
    candidate = _result(_basis(0))
    invalid = build_invalid_embedding_result(
        code=EmbeddingErrorCode.INFERENCE_FAILED,
        message="safe",
    )
    object.__setattr__(candidate, "errors", invalid.errors)

    result = compare_speaker_embeddings(_result(_basis(0)), candidate)

    _assert_invalid(result, SimilarityErrorCode.INVALID_CANDIDATE)


def test_candidate_dimension_mismatch_is_invalid_embedding() -> None:
    candidate = _tamper(
        _result(_basis(0)),
        embedding=np.ones(191, dtype=np.float32),
        metadata=replace(_metadata(), embedding_dimension=191),
    )

    result = compare_speaker_embeddings(_result(_basis(0)), candidate)

    _assert_invalid(result, SimilarityErrorCode.INVALID_EMBEDDING)


def test_wrong_reference_object_is_invalid_without_attribute_error() -> None:
    result = compare_speaker_embeddings(  # type: ignore[arg-type]
        object(),
        _result(_basis(0)),
    )

    _assert_invalid(result, SimilarityErrorCode.INVALID_REFERENCE)


def test_wrong_candidate_object_is_invalid_without_attribute_error() -> None:
    result = compare_speaker_embeddings(  # type: ignore[arg-type]
        _result(_basis(0)),
        object(),
    )

    _assert_invalid(result, SimilarityErrorCode.INVALID_CANDIDATE)


@pytest.mark.parametrize(
    ("argument", "expected_code"),
    [
        ("reference", SimilarityErrorCode.INVALID_REFERENCE),
        ("candidate", SimilarityErrorCode.INVALID_CANDIDATE),
    ],
)
def test_forged_string_valid_status_is_rejected(
    argument: str,
    expected_code: SimilarityErrorCode,
) -> None:
    reference = _result(_basis(0))
    candidate = _result(_basis(0))
    target = reference if argument == "reference" else candidate
    object.__setattr__(target, "status", "VALID")

    result = compare_speaker_embeddings(reference, candidate)

    _assert_invalid(result, expected_code)


def test_invalid_phase4_reference_and_candidate_precedence() -> None:
    invalid = build_invalid_embedding_result(
        code=EmbeddingErrorCode.INFERENCE_FAILED,
        message="safe",
    )

    result = compare_speaker_embeddings(invalid, invalid)

    _assert_invalid(result, SimilarityErrorCode.INVALID_REFERENCE)


def test_candidate_status_is_checked_before_reference_embedding() -> None:
    invalid_candidate = build_invalid_embedding_result(
        code=EmbeddingErrorCode.INFERENCE_FAILED,
        message="safe",
    )
    invalid_reference_embedding = _tamper(
        _result(_basis(0)),
        embedding=np.ones(191, dtype=np.float32),
    )

    result = compare_speaker_embeddings(
        invalid_reference_embedding,
        invalid_candidate,
    )

    _assert_invalid(result, SimilarityErrorCode.INVALID_CANDIDATE)


@pytest.mark.parametrize(
    ("field", "candidate_value"),
    [
        ("model_identifier", "other-model"),
        ("model_revision", "other-revision"),
        ("backend_name", "other-backend"),
        ("normalized", True),
        ("normalized", 0),
        ("input_sample_rate_hz", 8000),
    ],
)
def test_incompatible_metadata_is_rejected(
    field: str,
    candidate_value: object,
) -> None:
    candidate_metadata = replace(_metadata(), **{field: candidate_value})
    candidate = _tamper(_result(_basis(0)), metadata=candidate_metadata)

    result = compare_speaker_embeddings(_result(_basis(0)), candidate)

    _assert_invalid(result, SimilarityErrorCode.INCOMPATIBLE_EMBEDDINGS)


def test_non_target_reference_sample_rate_is_incompatible() -> None:
    reference = _tamper(
        _result(_basis(0)),
        metadata=replace(_metadata(), input_sample_rate_hz=48000),
    )

    result = compare_speaker_embeddings(reference, _result(_basis(0)))

    _assert_invalid(result, SimilarityErrorCode.INCOMPATIBLE_EMBEDDINGS)


@pytest.mark.parametrize(
    "sample_rate",
    [16000.0, np.int64(16000), True],
)
@pytest.mark.parametrize("argument", ["reference", "candidate"])
def test_sample_rate_must_be_exact_python_int(
    sample_rate: object,
    argument: str,
) -> None:
    reference = _result(_basis(0))
    candidate = _result(_basis(0))
    target = reference if argument == "reference" else candidate
    object.__setattr__(
        target,
        "metadata",
        replace(_metadata(), input_sample_rate_hz=sample_rate),
    )

    result = compare_speaker_embeddings(reference, candidate)

    _assert_invalid(result, SimilarityErrorCode.INCOMPATIBLE_EMBEDDINGS)


def test_different_inference_devices_remain_compatible() -> None:
    reference = _result(_basis(0), device="cpu")
    candidate = _result(_basis(0), device="other-device")

    result = compare_speaker_embeddings(reference, candidate)

    assert result.status == SimilarityStatus.VALID
    assert result.similarity == pytest.approx(1.0, abs=1e-7)


def test_unexpected_exception_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> float:
        raise RuntimeError("CANARY_EXCEPTION_/Users/private/token")

    monkeypatch.setattr(comparison_module, "_cosine_similarity_float64", fail)

    result = compare_speaker_embeddings(_result(_basis(0)), _result(_basis(0)))

    _assert_invalid(result, SimilarityErrorCode.COMPARISON_ERROR)
    assert "CANARY_EXCEPTION" not in repr(result)
    assert "CANARY_EXCEPTION" not in str(result)
    assert "CANARY_EXCEPTION" not in str(result.to_dict())
    assert "/Users/private" not in str(result.to_dict())


def test_memory_error_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> float:
        raise MemoryError("CANARY_MEMORY")

    monkeypatch.setattr(comparison_module, "_cosine_similarity_float64", fail)

    result = compare_speaker_embeddings(_result(_basis(0)), _result(_basis(0)))

    _assert_invalid(result, SimilarityErrorCode.COMPARISON_ERROR)
    assert "CANARY_MEMORY" not in repr(result)


@pytest.mark.parametrize("exception", [KeyboardInterrupt(), SystemExit()])
def test_process_control_exceptions_pass_through(
    monkeypatch: pytest.MonkeyPatch,
    exception: BaseException,
) -> None:
    def fail(*args: object, **kwargs: object) -> float:
        raise exception

    monkeypatch.setattr(comparison_module, "_cosine_similarity_float64", fail)

    with pytest.raises(type(exception)):
        compare_speaker_embeddings(_result(_basis(0)), _result(_basis(0)))


def test_numeric_overshoot_is_clipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        comparison_module.np,
        "dot",
        lambda reference, candidate: np.float64(1.0 + 1e-12),
    )

    result = compare_speaker_embeddings(_result(_basis(0)), _result(_basis(0)))

    assert result.similarity == 1.0


def test_output_invariants_determinism_and_symmetry() -> None:
    reference_vector = np.arange(1, 193, dtype=np.float32)
    candidate_vector = np.arange(192, 0, -1, dtype=np.float32)
    reference = _result(reference_vector)
    candidate = _result(candidate_vector)

    first = compare_speaker_embeddings(reference, candidate)
    repeated = compare_speaker_embeddings(reference, candidate)
    reversed_result = compare_speaker_embeddings(candidate, reference)

    assert type(first.similarity) is float
    assert first.similarity is not None
    assert np.isfinite(first.similarity)
    assert -1.0 <= first.similarity <= 1.0
    assert repeated.similarity == pytest.approx(first.similarity, abs=1e-12)
    assert reversed_result.similarity == pytest.approx(first.similarity, abs=1e-12)


def test_inputs_remain_unchanged_and_read_only() -> None:
    reference = _result(np.arange(1, 193, dtype=np.float32))
    candidate = _result(np.arange(192, 0, -1, dtype=np.float32))
    assert reference.embedding is not None
    assert candidate.embedding is not None
    reference_before = reference.embedding.tobytes()
    candidate_before = candidate.embedding.tobytes()

    compare_speaker_embeddings(reference, candidate)

    assert reference.embedding.tobytes() == reference_before
    assert candidate.embedding.tobytes() == candidate_before
    assert reference.embedding.flags.writeable is False
    assert candidate.embedding.flags.writeable is False


def test_safe_result_surfaces_exclude_inputs_and_private_metadata() -> None:
    canary = np.float32(0.12345679)
    reference_vector = np.full(192, canary, dtype=np.float32)
    reference = _result(
        reference_vector,
        device="/Users/private/cache/TOKEN_CANARY",
    )

    result = compare_speaker_embeddings(reference, _result(_basis(0)))
    public_surfaces = (repr(result), str(result), str(result.to_dict()))

    for surface in public_surfaces:
        assert "0.12345679" not in surface
        assert "/Users/private" not in surface
        assert "TOKEN_CANARY" not in surface
        assert "array(" not in surface
        assert "embedding=[" not in surface.lower()


@pytest.mark.parametrize(
    ("field", "canary"),
    [
        ("model_identifier", "/Users/private/TOKEN_MODEL_CANARY"),
        ("model_revision", "SECRET_REVISION_CANARY"),
        ("backend_name", "TOKEN_BACKEND_CANARY"),
    ],
)
def test_untrusted_compatible_model_metadata_is_not_public(
    field: str,
    canary: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    metadata = replace(_metadata(), **{field: canary})
    reference = _tamper(_result(_basis(0)), metadata=metadata)
    candidate = _tamper(_result(_basis(0)), metadata=metadata)

    result = compare_speaker_embeddings(reference, candidate)

    assert result.status == SimilarityStatus.VALID
    assert result.errors == ()
    surfaces = (
        repr(result),
        str(result),
        str(result.to_dict()),
        str(result.errors),
        caplog.text,
    )
    assert all(canary not in surface for surface in surfaces)
    assert result.metadata is not None
    assert "model_identifier" not in result.metadata.to_dict()
    assert "model_revision" not in result.metadata.to_dict()
    assert "backend_name" not in result.metadata.to_dict()


def _metadata(*, device: str = "cpu") -> EmbeddingMetadata:
    return EmbeddingMetadata(
        embedding_dimension=192,
        model_identifier="speechbrain/spkrec-ecapa-voxceleb",
        model_revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        backend_name="speechbrain-ecapa",
        device=device,
        input_sample_rate_hz=16000,
        input_samples=16000,
        input_duration_seconds=1.0,
        normalized=False,
    )


def _result(vector: np.ndarray, *, device: str = "cpu") -> SpeakerEmbeddingResult:
    return SpeakerEmbeddingResult(
        status=EmbeddingStatus.VALID,
        embedding=vector,
        metadata=_metadata(device=device),
        errors=(),
    )


_UNSET = object()


def _tamper(
    result: SpeakerEmbeddingResult,
    *,
    embedding: object = _UNSET,
    metadata: object = _UNSET,
) -> SpeakerEmbeddingResult:
    if embedding is not _UNSET:
        object.__setattr__(result, "embedding", embedding)
    if metadata is not _UNSET:
        object.__setattr__(result, "metadata", metadata)
    return result


def _assert_invalid(
    result: SpeakerSimilarityResult,
    code: SimilarityErrorCode,
) -> None:
    assert result.status == SimilarityStatus.INVALID
    assert result.similarity is None
    assert result.metadata is None
    errors = result.errors
    assert len(errors) == 1
    assert errors[0].code == code.value
