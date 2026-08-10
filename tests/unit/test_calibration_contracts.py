"""Tests for Phase 5B experimental calibration contracts."""

from __future__ import annotations

import pytest

import voiceid.calibration.contracts as calibration_contracts
from voiceid.audio import PREPROCESSING_CONTRACT_VERSION
from voiceid.calibration import (
    CALIBRATION_CONTRACT_VERSION,
    CALIBRATION_PROTOCOL_IDENTIFIER,
    CalibrationComparisonClass,
    CalibrationExperimentPlan,
    CalibrationPairRecord,
    CalibrationPartition,
    CalibrationProcessingProvenance,
    CalibrationProtocolIdentity,
    CalibrationSampleRecord,
    validate_calibration_experiment_plan,
)
from voiceid.embeddings import (
    EMBEDDING_CONTRACT_VERSION,
    SPEECHBRAIN_ECAPA_BACKEND_VERSION,
)
from voiceid.embeddings.policy import (
    SPEECHBRAIN_ECAPA_MODEL_ID,
    SPEECHBRAIN_ECAPA_MODEL_REVISION,
)
from voiceid.similarity import SIMILARITY_COMPARISON_VERSION

_SAFE_ERROR_MESSAGE = "Invalid calibration contract."
_SAFE_PLAN_ERROR_MESSAGE = "Invalid calibration experiment plan."
_BASELINE_SHA = "adcbadb094c3de5d6e4e5852ed2b448cf341542a"


class _ExplosiveString:
    def __str__(self) -> str:
        raise AssertionError("str should not be called")

    def __repr__(self) -> str:
        raise AssertionError("repr should not be called")


class _ImpersonatedPartition:
    CALIBRATION = "CALIBRATION"
    HOLDOUT = "HOLDOUT"


class _ImpersonatedComparisonClass:
    GENUINE = "GENUINE"


def test_valid_contracts_and_trusted_validator_accept_minimal_plan() -> None:
    plan = _plan()

    validate_calibration_experiment_plan(plan)

    assert plan.protocol.protocol_identifier == CALIBRATION_PROTOCOL_IDENTIFIER
    assert plan.protocol.calibration_contract_version == CALIBRATION_CONTRACT_VERSION
    assert plan.provenance.preprocessing_contract_version == (
        PREPROCESSING_CONTRACT_VERSION
    )
    assert plan.provenance.embedding_contract_version == EMBEDDING_CONTRACT_VERSION
    assert plan.provenance.backend_version == SPEECHBRAIN_ECAPA_BACKEND_VERSION
    assert plan.provenance.model_identifier == SPEECHBRAIN_ECAPA_MODEL_ID
    assert plan.provenance.model_revision == SPEECHBRAIN_ECAPA_MODEL_REVISION
    assert plan.provenance.comparison_version == SIMILARITY_COMPARISON_VERSION
    assert len(plan.samples) == 2
    assert len(plan.pairs) == 1


def test_public_all_exports_only_approved_contract_surface() -> None:
    assert calibration_contracts.__all__ == [
        "CALIBRATION_CONTRACT_VERSION",
        "CALIBRATION_PROTOCOL_IDENTIFIER",
        "CalibrationComparisonClass",
        "CalibrationExperimentPlan",
        "CalibrationPairRecord",
        "CalibrationPartition",
        "CalibrationProcessingProvenance",
        "CalibrationProtocolIdentity",
        "CalibrationSampleRecord",
        "validate_calibration_experiment_plan",
    ]


def test_plan_has_no_serialization_or_execution_behavior() -> None:
    plan = _plan()

    assert not hasattr(plan, "to_dict")
    assert not hasattr(plan, "run")
    assert not hasattr(plan, "score")
    assert not hasattr(plan, "threshold")
    assert not hasattr(plan, "verdict")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_identifier", "phase5b-other-protocol-v1"),
        ("protocol_identifier", None),
        ("protocol_identifier", b"phase5b"),
        ("calibration_contract_version", "phase5b-experiment-contracts-v2"),
        ("calibration_contract_version", ""),
        ("calibration_contract_version", object()),
        ("repository_commit_sha", "abc"),
        ("repository_commit_sha", "g" * 40),
        ("repository_commit_sha", True),
    ],
)
def test_protocol_identity_rejects_malformed_fields(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationProtocolIdentity(**_protocol_values(field, value))  # type: ignore[arg-type]


def test_protocol_identity_accepts_uppercase_hex_sha() -> None:
    protocol = CalibrationProtocolIdentity(
        protocol_identifier=CALIBRATION_PROTOCOL_IDENTIFIER,
        calibration_contract_version=CALIBRATION_CONTRACT_VERSION,
        repository_commit_sha="A" * 40,
    )

    assert protocol.repository_commit_sha == "A" * 40


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("preprocessing_contract_version", "phase3-v2"),
        ("preprocessing_contract_version", None),
        ("embedding_contract_version", "phase4b-v2"),
        ("embedding_contract_version", b"phase4b-v1"),
        ("backend_version", "fake-backend-v1"),
        ("backend_version", "/Users/private/backend"),
        ("model_identifier", "other/model"),
        ("model_identifier", "https://example.test/model"),
        ("model_revision", "0f99f2d0ebe89ac095bcc5903c4dd8f72b36728x"),
        ("model_revision", "0f99f2d0ebe89ac095bcc5903c4dd8f72b36728"),
        ("embedding_dimension", 191),
        ("embedding_dimension", True),
        ("embedding_dimension", 192.0),
        ("input_sample_rate_hz", 8000),
        ("input_sample_rate_hz", True),
        ("input_sample_rate_hz", 16000.0),
        ("normalized", True),
        ("normalized", 0),
        ("comparison_version", "2"),
        ("comparison_version", None),
    ],
)
def test_processing_provenance_rejects_malformed_fields(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationProcessingProvenance(**_provenance_values(field, value))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_id", "sample"),
        ("sample_id", "sample-"),
        ("sample_id", "sample-WithUppercase"),
        ("sample_id", "sample-has space"),
        ("sample_id", "sample-../private"),
        ("sample_id", "sample-token://secret"),
        ("sample_id", "sample-" + "a" * 58),
        ("sample_id", None),
        ("sample_id", b"sample-001"),
        ("subject_id", "speaker-001"),
        ("subject_id", "subject-/Users/private"),
        ("subject_id", True),
        ("source_group_id", "source-.."),
        ("source_group_id", "source-001/path"),
        ("source_group_id", object()),
        ("partition", "CALIBRATION"),
        ("partition", _ImpersonatedPartition.CALIBRATION),
    ],
)
def test_sample_record_rejects_malformed_fields(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationSampleRecord(**_sample_values(field, value))  # type: ignore[arg-type]


def test_identifier_boundary_length_is_enforced() -> None:
    accepted = CalibrationSampleRecord(
        sample_id="sample-" + "a" * 57,
        subject_id="subject-" + "a" * 56,
        source_group_id="source-" + "a" * 57,
        partition=CalibrationPartition.CALIBRATION,
    )

    assert len(accepted.sample_id) == 64
    assert len(accepted.subject_id) == 64
    assert len(accepted.source_group_id) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pair_id", "pair"),
        ("pair_id", "pair-"),
        ("pair_id", "pair-WithUppercase"),
        ("pair_id", "pair-has space"),
        ("pair_id", "pair-../secret"),
        ("reference_sample_id", "subject-001"),
        ("reference_sample_id", "/Users/private/sample-001"),
        ("probe_sample_id", "sample-token://secret"),
        ("comparison_class", "GENUINE"),
        ("comparison_class", _ImpersonatedComparisonClass.GENUINE),
        ("partition", "HOLDOUT"),
        ("partition", _ImpersonatedPartition.HOLDOUT),
    ],
)
def test_pair_record_rejects_malformed_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationPairRecord(**_pair_values(field, value))  # type: ignore[arg-type]


def test_plan_requires_exact_tuple_collections() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationExperimentPlan(
            protocol=_protocol(),
            provenance=_provenance(),
            samples=list(_samples()),  # type: ignore[arg-type]
            pairs=_pairs(),
        )
    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationExperimentPlan(
            protocol=_protocol(),
            provenance=_provenance(),
            samples=_samples(),
            pairs=list(_pairs()),  # type: ignore[arg-type]
        )


def test_caller_owned_mutable_collection_cannot_mutate_created_plan() -> None:
    samples = list(_samples())
    plan = CalibrationExperimentPlan(
        protocol=_protocol(),
        provenance=_provenance(),
        samples=tuple(samples),
        pairs=_pairs(),
    )

    samples.append(
        CalibrationSampleRecord(
            sample_id="sample-extra",
            subject_id="subject-extra",
            source_group_id="source-extra",
            partition=CalibrationPartition.CALIBRATION,
        )
    )

    assert len(plan.samples) == 2
    validate_calibration_experiment_plan(plan)


@pytest.mark.parametrize(
    "case_name",
    [
        "empty",
        "duplicate",
    ],
)
def test_plan_rejects_empty_or_duplicate_samples(
    case_name: str,
) -> None:
    samples = (
        ()
        if case_name == "empty"
        else (
            _sample("sample-001", "subject-001", "source-001"),
            _sample("sample-001", "subject-001", "source-002"),
        )
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationExperimentPlan(
            protocol=_protocol(),
            provenance=_provenance(),
            samples=samples,
            pairs=_pairs(),
        )


@pytest.mark.parametrize(
    "case_name",
    [
        "empty",
        "duplicate",
    ],
)
def test_plan_rejects_empty_or_duplicate_pairs(
    case_name: str,
) -> None:
    pairs = (
        ()
        if case_name == "empty"
        else (
            _pair("pair-001", "sample-001", "sample-002"),
            _pair("pair-001", "sample-001", "sample-002"),
        )
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationExperimentPlan(
            protocol=_protocol(),
            provenance=_provenance(),
            samples=_samples(),
            pairs=pairs,
        )


@pytest.mark.parametrize(
    ("reference_sample_id", "probe_sample_id"),
    [
        ("sample-missing", "sample-002"),
        ("sample-001", "sample-missing"),
    ],
)
def test_plan_rejects_missing_pair_references(
    reference_sample_id: str,
    probe_sample_id: str,
) -> None:
    pair = _pair("pair-001", reference_sample_id, probe_sample_id)

    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationExperimentPlan(
            protocol=_protocol(),
            provenance=_provenance(),
            samples=_samples(),
            pairs=(pair,),
        )


def test_plan_rejects_same_sample_pair() -> None:
    pair = _pair("pair-001", "sample-001", "sample-001")

    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationExperimentPlan(
            protocol=_protocol(),
            provenance=_provenance(),
            samples=_samples(),
            pairs=(pair,),
        )


def test_plan_rejects_same_source_group_pair() -> None:
    samples = (
        _sample("sample-001", "subject-001", "source-001"),
        _sample("sample-002", "subject-001", "source-001"),
    )

    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationExperimentPlan(
            protocol=_protocol(),
            provenance=_provenance(),
            samples=samples,
            pairs=_pairs(),
        )


def test_plan_rejects_partition_mismatch() -> None:
    samples = (
        _sample("sample-001", "subject-001", "source-001"),
        _sample(
            "sample-002",
            "subject-001",
            "source-002",
            partition=CalibrationPartition.HOLDOUT,
        ),
    )

    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationExperimentPlan(
            protocol=_protocol(),
            provenance=_provenance(),
            samples=samples,
            pairs=_pairs(),
        )


def test_genuine_pair_requires_same_subject() -> None:
    samples = (
        _sample("sample-001", "subject-001", "source-001"),
        _sample("sample-002", "subject-002", "source-002"),
    )

    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationExperimentPlan(
            protocol=_protocol(),
            provenance=_provenance(),
            samples=samples,
            pairs=_pairs(),
        )


def test_impostor_pair_requires_different_subjects() -> None:
    pair = _pair(
        "pair-001",
        "sample-001",
        "sample-002",
        comparison_class=CalibrationComparisonClass.IMPOSTOR,
    )

    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationExperimentPlan(
            protocol=_protocol(),
            provenance=_provenance(),
            samples=_samples(),
            pairs=(pair,),
        )


def test_holdout_impostor_plan_is_supported_without_requiring_both_classes() -> None:
    samples = (
        _sample(
            "sample-001",
            "subject-001",
            "source-001",
            partition=CalibrationPartition.HOLDOUT,
        ),
        _sample(
            "sample-002",
            "subject-002",
            "source-002",
            partition=CalibrationPartition.HOLDOUT,
        ),
    )
    pair = _pair(
        "pair-001",
        "sample-001",
        "sample-002",
        comparison_class=CalibrationComparisonClass.IMPOSTOR,
        partition=CalibrationPartition.HOLDOUT,
    )
    plan = CalibrationExperimentPlan(
        protocol=_protocol(),
        provenance=_provenance(),
        samples=samples,
        pairs=(pair,),
    )

    validate_calibration_experiment_plan(plan)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("protocol", object()),
        ("provenance", object()),
        ("samples", (object(),)),
        ("pairs", (object(),)),
    ],
)
def test_trusted_validator_rejects_forged_top_level_state(
    attribute: str,
    value: object,
) -> None:
    plan = _plan()
    object.__setattr__(plan, attribute, value)

    with pytest.raises(ValueError, match=_SAFE_PLAN_ERROR_MESSAGE):
        validate_calibration_experiment_plan(plan)


@pytest.mark.parametrize(
    ("target_name", "attribute", "value"),
    [
        ("protocol", "repository_commit_sha", "TOKEN_CANARY"),
        ("provenance", "backend_version", "/Users/private/backend"),
        ("sample", "subject_id", "subject-token://secret"),
        ("pair", "reference_sample_id", "sample-missing"),
    ],
)
def test_trusted_validator_rejects_forged_nested_state_without_leaks(
    target_name: str,
    attribute: str,
    value: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = _plan()
    target = {
        "protocol": plan.protocol,
        "provenance": plan.provenance,
        "sample": plan.samples[0],
        "pair": plan.pairs[0],
    }[target_name]
    object.__setattr__(target, attribute, value)

    with pytest.raises(ValueError) as exc_info:
        validate_calibration_experiment_plan(plan)

    captured = capsys.readouterr()
    surfaces = (
        repr(plan),
        str(plan),
        repr(plan.protocol),
        repr(plan.provenance),
        repr(plan.samples[0]),
        repr(plan.pairs[0]),
        str(exc_info.value),
        captured.out,
        captured.err,
    )
    assert all(str(value) not in surface for surface in surfaces)


def test_validator_does_not_call_user_str_or_repr_for_errors() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationSampleRecord(
            sample_id=_ExplosiveString(),
            subject_id="subject-001",
            source_group_id="source-001",
            partition=CalibrationPartition.CALIBRATION,
        )  # type: ignore[arg-type]


def test_memory_and_process_control_exceptions_are_not_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()

    def raise_memory_error(_plan: object) -> None:
        raise MemoryError

    monkeypatch.setattr(
        calibration_contracts,
        "_validate_experiment_plan",
        raise_memory_error,
    )
    with pytest.raises(MemoryError):
        validate_calibration_experiment_plan(plan)

    def raise_keyboard_interrupt(_plan: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        calibration_contracts,
        "_validate_experiment_plan",
        raise_keyboard_interrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        validate_calibration_experiment_plan(plan)

    def raise_system_exit(_plan: object) -> None:
        raise SystemExit

    monkeypatch.setattr(
        calibration_contracts,
        "_validate_experiment_plan",
        raise_system_exit,
    )
    with pytest.raises(SystemExit):
        validate_calibration_experiment_plan(plan)


def test_forbidden_public_fields_are_absent_from_contracts() -> None:
    forbidden = {
        "audio",
        "waveform",
        "embedding",
        "filename",
        "path",
        "cache",
        "threshold",
        "score",
        "verdict",
        "consent_reference",
        "condition_labels",
        "strata",
        "correlation_group_id",
    }
    contract_fields = (
        set(CalibrationProtocolIdentity.__dataclass_fields__)
        | set(CalibrationProcessingProvenance.__dataclass_fields__)
        | set(CalibrationSampleRecord.__dataclass_fields__)
        | set(CalibrationPairRecord.__dataclass_fields__)
        | set(CalibrationExperimentPlan.__dataclass_fields__)
    )

    assert forbidden.isdisjoint(contract_fields)


def test_safe_repr_str_hide_identifiers_and_canaries() -> None:
    plan = _plan()
    surfaces = (
        repr(plan),
        str(plan),
        repr(plan.protocol),
        str(plan.protocol),
        repr(plan.provenance),
        repr(plan.samples[0]),
        repr(plan.pairs[0]),
    )

    for surface in surfaces:
        assert "sample-001" not in surface
        assert "subject-001" not in surface
        assert "source-001" not in surface
        assert "pair-001" not in surface
        assert "adcbadb" not in surface


def _protocol() -> CalibrationProtocolIdentity:
    return CalibrationProtocolIdentity(
        protocol_identifier=CALIBRATION_PROTOCOL_IDENTIFIER,
        calibration_contract_version=CALIBRATION_CONTRACT_VERSION,
        repository_commit_sha=_BASELINE_SHA,
    )


def _provenance() -> CalibrationProcessingProvenance:
    return CalibrationProcessingProvenance(
        preprocessing_contract_version=PREPROCESSING_CONTRACT_VERSION,
        embedding_contract_version=EMBEDDING_CONTRACT_VERSION,
        backend_version=SPEECHBRAIN_ECAPA_BACKEND_VERSION,
        model_identifier=SPEECHBRAIN_ECAPA_MODEL_ID,
        model_revision=SPEECHBRAIN_ECAPA_MODEL_REVISION,
        embedding_dimension=192,
        input_sample_rate_hz=16000,
        normalized=False,
        comparison_version=SIMILARITY_COMPARISON_VERSION,
    )


def _sample(
    sample_id: str,
    subject_id: str,
    source_group_id: str,
    *,
    partition: CalibrationPartition = CalibrationPartition.CALIBRATION,
) -> CalibrationSampleRecord:
    return CalibrationSampleRecord(
        sample_id=sample_id,
        subject_id=subject_id,
        source_group_id=source_group_id,
        partition=partition,
    )


def _pair(
    pair_id: str,
    reference_sample_id: str,
    probe_sample_id: str,
    *,
    comparison_class: CalibrationComparisonClass = (CalibrationComparisonClass.GENUINE),
    partition: CalibrationPartition = CalibrationPartition.CALIBRATION,
) -> CalibrationPairRecord:
    return CalibrationPairRecord(
        pair_id=pair_id,
        reference_sample_id=reference_sample_id,
        probe_sample_id=probe_sample_id,
        comparison_class=comparison_class,
        partition=partition,
    )


def _samples() -> tuple[CalibrationSampleRecord, ...]:
    return (
        _sample("sample-001", "subject-001", "source-001"),
        _sample("sample-002", "subject-001", "source-002"),
    )


def _pairs() -> tuple[CalibrationPairRecord, ...]:
    return (_pair("pair-001", "sample-001", "sample-002"),)


def _plan() -> CalibrationExperimentPlan:
    return CalibrationExperimentPlan(
        protocol=_protocol(),
        provenance=_provenance(),
        samples=_samples(),
        pairs=_pairs(),
    )


def _protocol_values(field: str, value: object) -> dict[str, object]:
    values: dict[str, object] = {
        "protocol_identifier": CALIBRATION_PROTOCOL_IDENTIFIER,
        "calibration_contract_version": CALIBRATION_CONTRACT_VERSION,
        "repository_commit_sha": _BASELINE_SHA,
    }
    values[field] = value
    return values


def _provenance_values(field: str, value: object) -> dict[str, object]:
    values: dict[str, object] = {
        "preprocessing_contract_version": PREPROCESSING_CONTRACT_VERSION,
        "embedding_contract_version": EMBEDDING_CONTRACT_VERSION,
        "backend_version": SPEECHBRAIN_ECAPA_BACKEND_VERSION,
        "model_identifier": SPEECHBRAIN_ECAPA_MODEL_ID,
        "model_revision": SPEECHBRAIN_ECAPA_MODEL_REVISION,
        "embedding_dimension": 192,
        "input_sample_rate_hz": 16000,
        "normalized": False,
        "comparison_version": SIMILARITY_COMPARISON_VERSION,
    }
    values[field] = value
    return values


def _sample_values(field: str, value: object) -> dict[str, object]:
    values: dict[str, object] = {
        "sample_id": "sample-001",
        "subject_id": "subject-001",
        "source_group_id": "source-001",
        "partition": CalibrationPartition.CALIBRATION,
    }
    values[field] = value
    return values


def _pair_values(field: str, value: object) -> dict[str, object]:
    values: dict[str, object] = {
        "pair_id": "pair-001",
        "reference_sample_id": "sample-001",
        "probe_sample_id": "sample-002",
        "comparison_class": CalibrationComparisonClass.GENUINE,
        "partition": CalibrationPartition.CALIBRATION,
    }
    values[field] = value
    return values
