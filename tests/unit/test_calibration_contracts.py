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
_SAMPLE_ID_1 = "smp_0123456789abcdef0123456789abcdef"
_SAMPLE_ID_2 = "smp_11111111111111111111111111111111"
_SAMPLE_ID_MISSING = "smp_22222222222222222222222222222222"
_SAMPLE_ID_EXTRA = "smp_33333333333333333333333333333333"
_SUBJECT_ID_1 = "sub_0123456789abcdef0123456789abcdef"
_SUBJECT_ID_2 = "sub_11111111111111111111111111111111"
_SUBJECT_ID_EXTRA = "sub_33333333333333333333333333333333"
_SOURCE_GROUP_ID_1 = "src_0123456789abcdef0123456789abcdef"
_SOURCE_GROUP_ID_2 = "src_11111111111111111111111111111111"
_SOURCE_GROUP_ID_EXTRA = "src_33333333333333333333333333333333"
_PAIR_ID_1 = "pair_0123456789abcdef0123456789abcdef"

_NORMATIVE_STRING_FIELDS = [
    ("protocol", "protocol_identifier", CALIBRATION_PROTOCOL_IDENTIFIER),
    ("protocol", "calibration_contract_version", CALIBRATION_CONTRACT_VERSION),
    ("protocol", "repository_commit_sha", _BASELINE_SHA),
    ("provenance", "preprocessing_contract_version", PREPROCESSING_CONTRACT_VERSION),
    ("provenance", "embedding_contract_version", EMBEDDING_CONTRACT_VERSION),
    ("provenance", "backend_version", SPEECHBRAIN_ECAPA_BACKEND_VERSION),
    ("provenance", "model_identifier", SPEECHBRAIN_ECAPA_MODEL_ID),
    ("provenance", "model_revision", SPEECHBRAIN_ECAPA_MODEL_REVISION),
    ("provenance", "comparison_version", SIMILARITY_COMPARISON_VERSION),
]

_IDENTIFIER_FIELD_CASES = [
    ("sample", "sample_id", _SAMPLE_ID_1),
    ("sample", "subject_id", _SUBJECT_ID_1),
    ("sample", "source_group_id", _SOURCE_GROUP_ID_1),
    ("pair", "pair_id", _PAIR_ID_1),
    ("pair", "reference_sample_id", _SAMPLE_ID_1),
    ("pair", "probe_sample_id", _SAMPLE_ID_2),
]


class _ExplosiveString:
    def __str__(self) -> str:
        raise AssertionError("str should not be called")

    def __repr__(self) -> str:
        raise AssertionError("repr should not be called")


class _StrSubclass(str):
    """A str subclass that must not be accepted as an exact string."""


class _ExplosiveComparable:
    def __init__(self) -> None:
        self.eq_called = False
        self.ne_called = False
        self.str_called = False
        self.repr_called = False

    def __eq__(self, _other: object) -> bool:
        self.eq_called = True
        raise AssertionError("TOKEN_CANARY __eq__ should not be called")

    def __ne__(self, _other: object) -> bool:
        self.ne_called = True
        raise AssertionError("TOKEN_CANARY __ne__ should not be called")

    def __str__(self) -> str:
        self.str_called = True
        raise AssertionError("TOKEN_CANARY __str__ should not be called")

    def __repr__(self) -> str:
        self.repr_called = True
        raise AssertionError("TOKEN_CANARY __repr__ should not be called")


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
    ("target_name", "field", "expected"),
    _NORMATIVE_STRING_FIELDS,
)
def test_normative_string_fields_reject_str_subclass_at_constructor_boundary(
    target_name: str,
    field: str,
    expected: str,
) -> None:
    value = _StrSubclass(expected)

    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        _build_normative_contract_with_value(target_name, field, value)


@pytest.mark.parametrize(
    ("target_name", "field", "expected"),
    _NORMATIVE_STRING_FIELDS,
)
def test_normative_string_fields_reject_forged_str_subclass_at_trusted_boundary(
    target_name: str,
    field: str,
    expected: str,
) -> None:
    plan = _plan()
    _forge_normative_field(plan, target_name, field, _StrSubclass(expected))

    with pytest.raises(ValueError, match=_SAFE_PLAN_ERROR_MESSAGE):
        validate_calibration_experiment_plan(plan)


@pytest.mark.parametrize(
    ("target_name", "field", "_expected"),
    _NORMATIVE_STRING_FIELDS,
)
def test_normative_string_validation_does_not_call_user_comparison_methods(
    target_name: str,
    field: str,
    _expected: str,
) -> None:
    value = _ExplosiveComparable()

    with pytest.raises(ValueError) as exc_info:
        _build_normative_contract_with_value(target_name, field, value)

    assert str(exc_info.value) == _SAFE_ERROR_MESSAGE
    assert not value.eq_called
    assert not value.ne_called
    assert not value.str_called
    assert not value.repr_called
    assert "TOKEN_CANARY" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("target_name", "field", "expected"),
    _IDENTIFIER_FIELD_CASES,
)
def test_identifier_fields_reject_str_subclass_at_constructor_boundary(
    target_name: str,
    field: str,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        _build_identifier_contract_with_value(
            target_name,
            field,
            _StrSubclass(expected),
        )


@pytest.mark.parametrize(
    ("target_name", "field", "expected"),
    _IDENTIFIER_FIELD_CASES,
)
def test_identifier_fields_reject_forged_str_subclass_at_trusted_boundary(
    target_name: str,
    field: str,
    expected: str,
) -> None:
    plan = _plan()
    _forge_identifier_field(plan, target_name, field, _StrSubclass(expected))

    with pytest.raises(ValueError, match=_SAFE_PLAN_ERROR_MESSAGE):
        validate_calibration_experiment_plan(plan)


@pytest.mark.parametrize(
    ("target_name", "field", "_expected"),
    _IDENTIFIER_FIELD_CASES,
)
def test_identifier_validation_does_not_call_user_comparison_methods(
    target_name: str,
    field: str,
    _expected: str,
) -> None:
    value = _ExplosiveComparable()

    with pytest.raises(ValueError) as exc_info:
        _build_identifier_contract_with_value(target_name, field, value)

    assert str(exc_info.value) == _SAFE_ERROR_MESSAGE
    assert not value.eq_called
    assert not value.ne_called
    assert not value.str_called
    assert not value.repr_called
    assert "TOKEN_CANARY" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_id", "smp_"),
        ("sample_id", "smp_0123456789abcdef0123456789abcde"),
        ("sample_id", "smp_0123456789abcdef0123456789abcdef0"),
        ("sample_id", "smp_0123456789abcdef0123456789abcdeg"),
        ("sample_id", "smp_0123456789ABCDEF0123456789ABCDEF"),
        ("sample_id", "sample-person-name-session-01.wav"),
        ("sample_id", "smp_0123456789abcdef0123456789abcd ef"),
        ("sample_id", "smp_../private"),
        ("sample_id", "smp_token://secret"),
        ("sample_id", None),
        ("sample_id", b"smp_0123456789abcdef0123456789abcdef"),
        ("subject_id", "subject-jane-doe"),
        ("subject_id", "sub_0123456789ABCDEF0123456789ABCDEF"),
        ("subject_id", "sub_/Users/private"),
        ("subject_id", True),
        ("source_group_id", "source-ghp_fakeqatokencanary"),
        ("source_group_id", "source-https-example.com"),
        ("source_group_id", "src_0123456789abcdef0123456789abcdeg"),
        ("source_group_id", "src_0123456789abcdef0123456789abc\n"),
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


def test_opaque_identifier_grammar_accepts_exact_prefix_and_lowercase_hex() -> None:
    accepted = CalibrationSampleRecord(
        sample_id=_SAMPLE_ID_1,
        subject_id=_SUBJECT_ID_1,
        source_group_id=_SOURCE_GROUP_ID_1,
        partition=CalibrationPartition.CALIBRATION,
    )

    assert accepted.sample_id == "smp_0123456789abcdef0123456789abcdef"
    assert accepted.subject_id == "sub_0123456789abcdef0123456789abcdef"
    assert accepted.source_group_id == "src_0123456789abcdef0123456789abcdef"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pair_id", "pair"),
        ("pair_id", "pair_"),
        ("pair_id", "pair_0123456789abcdef0123456789abcde"),
        ("pair_id", "pair_0123456789abcdef0123456789abcdef0"),
        ("pair_id", "pair_0123456789abcdef0123456789abcdeg"),
        ("pair_id", "pair_0123456789ABCDEF0123456789ABCDEF"),
        ("pair_id", "pair-jane-doe"),
        ("pair_id", "pair_../secret"),
        ("reference_sample_id", _SUBJECT_ID_1),
        ("reference_sample_id", "/Users/private/sample-001"),
        ("probe_sample_id", "smp_token://secret"),
        ("comparison_class", "GENUINE"),
        ("comparison_class", _ImpersonatedComparisonClass.GENUINE),
        ("partition", "HOLDOUT"),
        ("partition", _ImpersonatedPartition.HOLDOUT),
    ],
)
def test_pair_record_rejects_malformed_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationPairRecord(**_pair_values(field, value))  # type: ignore[arg-type]


def test_plan_accepts_exact_list_collections_and_copies_them_defensively() -> None:
    samples = list(_samples())
    pairs = list(_pairs())

    plan = CalibrationExperimentPlan(
        protocol=_protocol(),
        provenance=_provenance(),
        samples=samples,
        pairs=pairs,
    )
    samples.append(
        CalibrationSampleRecord(
            sample_id=_SAMPLE_ID_EXTRA,
            subject_id=_SUBJECT_ID_EXTRA,
            source_group_id=_SOURCE_GROUP_ID_EXTRA,
            partition=CalibrationPartition.CALIBRATION,
        )
    )
    pairs.clear()

    assert type(plan.samples) is tuple
    assert type(plan.pairs) is tuple
    assert len(plan.samples) == 2
    assert len(plan.pairs) == 1
    validate_calibration_experiment_plan(plan)


def test_plan_accepts_exact_tuple_collections() -> None:
    plan = CalibrationExperimentPlan(
        protocol=_protocol(),
        provenance=_provenance(),
        samples=_samples(),
        pairs=_pairs(),
    )

    assert type(plan.samples) is tuple
    assert type(plan.pairs) is tuple


def test_plan_rejects_non_builtin_collection_inputs() -> None:
    class _ListSubclass(list):
        pass

    class _TupleSubclass(tuple):
        pass

    invalid_sample_collections = (
        iter(_samples()),
        (sample for sample in _samples()),
        {"sample": _samples()[0]},
        "samples",
        _ListSubclass(_samples()),
        _TupleSubclass(_samples()),
    )
    for samples in invalid_sample_collections:
        with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
            CalibrationExperimentPlan(
                protocol=_protocol(),
                provenance=_provenance(),
                samples=samples,  # type: ignore[arg-type]
                pairs=_pairs(),
            )

    invalid_pair_collections = (
        iter(_pairs()),
        (pair for pair in _pairs()),
        {"pair": _pairs()[0]},
        "pairs",
        _ListSubclass(_pairs()),
        _TupleSubclass(_pairs()),
    )
    for pairs in invalid_pair_collections:
        with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
            CalibrationExperimentPlan(
                protocol=_protocol(),
                provenance=_provenance(),
                samples=_samples(),
                pairs=pairs,  # type: ignore[arg-type]
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
            sample_id=_SAMPLE_ID_EXTRA,
            subject_id=_SUBJECT_ID_EXTRA,
            source_group_id=_SOURCE_GROUP_ID_EXTRA,
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
            _sample(_SAMPLE_ID_1, _SUBJECT_ID_1, _SOURCE_GROUP_ID_1),
            _sample(_SAMPLE_ID_1, _SUBJECT_ID_1, _SOURCE_GROUP_ID_2),
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
            _pair(_PAIR_ID_1, _SAMPLE_ID_1, _SAMPLE_ID_2),
            _pair(_PAIR_ID_1, _SAMPLE_ID_1, _SAMPLE_ID_2),
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
        (_SAMPLE_ID_MISSING, _SAMPLE_ID_2),
        (_SAMPLE_ID_1, _SAMPLE_ID_MISSING),
    ],
)
def test_plan_rejects_missing_pair_references(
    reference_sample_id: str,
    probe_sample_id: str,
) -> None:
    pair = _pair(_PAIR_ID_1, reference_sample_id, probe_sample_id)

    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationExperimentPlan(
            protocol=_protocol(),
            provenance=_provenance(),
            samples=_samples(),
            pairs=(pair,),
        )


def test_plan_rejects_same_sample_pair() -> None:
    pair = _pair(_PAIR_ID_1, _SAMPLE_ID_1, _SAMPLE_ID_1)

    with pytest.raises(ValueError, match=_SAFE_ERROR_MESSAGE):
        CalibrationExperimentPlan(
            protocol=_protocol(),
            provenance=_provenance(),
            samples=_samples(),
            pairs=(pair,),
        )


def test_plan_rejects_same_source_group_pair() -> None:
    samples = (
        _sample(_SAMPLE_ID_1, _SUBJECT_ID_1, _SOURCE_GROUP_ID_1),
        _sample(_SAMPLE_ID_2, _SUBJECT_ID_1, _SOURCE_GROUP_ID_1),
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
        _sample(_SAMPLE_ID_1, _SUBJECT_ID_1, _SOURCE_GROUP_ID_1),
        _sample(
            _SAMPLE_ID_2,
            _SUBJECT_ID_1,
            _SOURCE_GROUP_ID_2,
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
        _sample(_SAMPLE_ID_1, _SUBJECT_ID_1, _SOURCE_GROUP_ID_1),
        _sample(_SAMPLE_ID_2, _SUBJECT_ID_2, _SOURCE_GROUP_ID_2),
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
        _PAIR_ID_1,
        _SAMPLE_ID_1,
        _SAMPLE_ID_2,
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
            _SAMPLE_ID_1,
            _SUBJECT_ID_1,
            _SOURCE_GROUP_ID_1,
            partition=CalibrationPartition.HOLDOUT,
        ),
        _sample(
            _SAMPLE_ID_2,
            _SUBJECT_ID_2,
            _SOURCE_GROUP_ID_2,
            partition=CalibrationPartition.HOLDOUT,
        ),
    )
    pair = _pair(
        _PAIR_ID_1,
        _SAMPLE_ID_1,
        _SAMPLE_ID_2,
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
    ("attribute", "value"),
    [
        ("samples", []),
        ("pairs", []),
    ],
)
def test_trusted_validator_rejects_forged_list_collections(
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
        ("pair", "reference_sample_id", _SAMPLE_ID_MISSING),
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
            subject_id=_SUBJECT_ID_1,
            source_group_id=_SOURCE_GROUP_ID_1,
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
        assert _SAMPLE_ID_1 not in surface
        assert _SUBJECT_ID_1 not in surface
        assert _SOURCE_GROUP_ID_1 not in surface
        assert _PAIR_ID_1 not in surface
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
        _sample(_SAMPLE_ID_1, _SUBJECT_ID_1, _SOURCE_GROUP_ID_1),
        _sample(_SAMPLE_ID_2, _SUBJECT_ID_1, _SOURCE_GROUP_ID_2),
    )


def _pairs() -> tuple[CalibrationPairRecord, ...]:
    return (_pair(_PAIR_ID_1, _SAMPLE_ID_1, _SAMPLE_ID_2),)


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
        "sample_id": _SAMPLE_ID_1,
        "subject_id": _SUBJECT_ID_1,
        "source_group_id": _SOURCE_GROUP_ID_1,
        "partition": CalibrationPartition.CALIBRATION,
    }
    values[field] = value
    return values


def _pair_values(field: str, value: object) -> dict[str, object]:
    values: dict[str, object] = {
        "pair_id": _PAIR_ID_1,
        "reference_sample_id": _SAMPLE_ID_1,
        "probe_sample_id": _SAMPLE_ID_2,
        "comparison_class": CalibrationComparisonClass.GENUINE,
        "partition": CalibrationPartition.CALIBRATION,
    }
    values[field] = value
    return values


def _build_normative_contract_with_value(
    target_name: str,
    field: str,
    value: object,
) -> object:
    if target_name == "protocol":
        return CalibrationProtocolIdentity(**_protocol_values(field, value))  # type: ignore[arg-type]
    if target_name == "provenance":
        return CalibrationProcessingProvenance(**_provenance_values(field, value))  # type: ignore[arg-type]
    raise AssertionError("unknown test target")


def _forge_normative_field(
    plan: CalibrationExperimentPlan,
    target_name: str,
    field: str,
    value: object,
) -> None:
    target = plan.protocol if target_name == "protocol" else plan.provenance
    object.__setattr__(target, field, value)


def _build_identifier_contract_with_value(
    target_name: str,
    field: str,
    value: object,
) -> object:
    if target_name == "sample":
        return CalibrationSampleRecord(**_sample_values(field, value))  # type: ignore[arg-type]
    if target_name == "pair":
        return CalibrationPairRecord(**_pair_values(field, value))  # type: ignore[arg-type]
    raise AssertionError("unknown test target")


def _forge_identifier_field(
    plan: CalibrationExperimentPlan,
    target_name: str,
    field: str,
    value: object,
) -> None:
    target = plan.samples[0] if target_name == "sample" else plan.pairs[0]
    object.__setattr__(target, field, value)
