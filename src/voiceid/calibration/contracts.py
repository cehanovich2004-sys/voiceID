"""Executable contracts for Phase 5B experimental calibration plans."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from voiceid.audio import PREPROCESSING_CONTRACT_VERSION
from voiceid.embeddings.contracts import EMBEDDING_CONTRACT_VERSION, EMBEDDING_DIMENSION
from voiceid.embeddings.policy import (
    SPEECHBRAIN_ECAPA_BACKEND_VERSION,
    SPEECHBRAIN_ECAPA_MODEL_ID,
    SPEECHBRAIN_ECAPA_MODEL_REVISION,
    TARGET_EMBEDDING_SAMPLE_RATE_HZ,
)
from voiceid.similarity import SIMILARITY_COMPARISON_VERSION

CALIBRATION_PROTOCOL_IDENTIFIER: Final = "phase5b-experimental-calibration-protocol-v1"
CALIBRATION_CONTRACT_VERSION: Final = "phase5b-experiment-contracts-v1"

_CONTRACT_ERROR_MESSAGE: Final = "Invalid calibration contract."
_PLAN_ERROR_MESSAGE: Final = "Invalid calibration experiment plan."
_IDENTIFIER_MAX_LENGTH: Final = 64
_IDENTIFIER_RE: Final = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_SHA_RE: Final = re.compile(r"^[0-9a-fA-F]{40}$")


class CalibrationPartition(StrEnum):
    """Approved Phase 5B experimental partitions."""

    CALIBRATION = "CALIBRATION"
    HOLDOUT = "HOLDOUT"


class CalibrationComparisonClass(StrEnum):
    """Approved Phase 5B comparison labels."""

    GENUINE = "GENUINE"
    IMPOSTOR = "IMPOSTOR"


@dataclass(frozen=True, slots=True, repr=False)
class CalibrationProtocolIdentity:
    """Protocol identity for a planned calibration experiment."""

    protocol_identifier: str
    calibration_contract_version: str
    repository_commit_sha: str

    def __post_init__(self) -> None:
        """Validate protocol identity without exposing malformed values."""

        _raise_contract_error_if_invalid(_validate_protocol_identity, self)

    def __repr__(self) -> str:
        """Return a privacy-safe representation."""

        return "CalibrationProtocolIdentity(redacted=True)"

    def __str__(self) -> str:
        """Return the same privacy-safe representation as repr()."""

        return repr(self)


@dataclass(frozen=True, slots=True, repr=False)
class CalibrationProcessingProvenance:
    """Processing provenance required before score calibration can be planned."""

    preprocessing_contract_version: str
    embedding_contract_version: str
    backend_version: str
    model_identifier: str
    model_revision: str
    embedding_dimension: int
    input_sample_rate_hz: int
    normalized: bool
    comparison_version: str

    def __post_init__(self) -> None:
        """Validate processing provenance without exposing malformed values."""

        _raise_contract_error_if_invalid(_validate_processing_provenance, self)

    def __repr__(self) -> str:
        """Return a privacy-safe representation."""

        return "CalibrationProcessingProvenance(redacted=True)"

    def __str__(self) -> str:
        """Return the same privacy-safe representation as repr()."""

        return repr(self)


@dataclass(frozen=True, slots=True, repr=False)
class CalibrationSampleRecord:
    """Privacy-minimized sample record for an approved experiment plan."""

    sample_id: str
    subject_id: str
    source_group_id: str
    partition: CalibrationPartition

    def __post_init__(self) -> None:
        """Validate sample metadata without exposing identifiers."""

        _raise_contract_error_if_invalid(_validate_sample_record, self)

    def __repr__(self) -> str:
        """Return a privacy-safe representation."""

        return "CalibrationSampleRecord(redacted=True)"

    def __str__(self) -> str:
        """Return the same privacy-safe representation as repr()."""

        return repr(self)


@dataclass(frozen=True, slots=True, repr=False)
class CalibrationPairRecord:
    """Reference/probe comparison pair planned for calibration analysis."""

    pair_id: str
    reference_sample_id: str
    probe_sample_id: str
    comparison_class: CalibrationComparisonClass
    partition: CalibrationPartition

    def __post_init__(self) -> None:
        """Validate pair metadata without exposing identifiers."""

        _raise_contract_error_if_invalid(_validate_pair_record, self)

    def __repr__(self) -> str:
        """Return a privacy-safe representation."""

        return "CalibrationPairRecord(redacted=True)"

    def __str__(self) -> str:
        """Return the same privacy-safe representation as repr()."""

        return repr(self)


@dataclass(frozen=True, slots=True, repr=False)
class CalibrationExperimentPlan:
    """Declarative calibration experiment plan without execution or scores."""

    protocol: CalibrationProtocolIdentity
    provenance: CalibrationProcessingProvenance
    samples: tuple[CalibrationSampleRecord, ...]
    pairs: tuple[CalibrationPairRecord, ...]

    def __post_init__(self) -> None:
        """Validate the complete plan at construction time."""

        _raise_contract_error_if_invalid(_validate_experiment_plan, self)

    def __repr__(self) -> str:
        """Return a privacy-safe representation without sample or pair IDs."""

        return "CalibrationExperimentPlan(redacted=True)"

    def __str__(self) -> str:
        """Return the same privacy-safe representation as repr()."""

        return repr(self)


def validate_calibration_experiment_plan(plan: CalibrationExperimentPlan) -> None:
    """Revalidate a complete plan after crossing a trust boundary.

    This function intentionally returns no data. It is a fail-closed guard for
    forged or mutated frozen dataclasses before future calibration tooling may
    trust a plan.
    """

    try:
        _validate_experiment_plan(plan)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        raise ValueError(_PLAN_ERROR_MESSAGE) from None


class _CalibrationContractError(Exception):
    """Private sentinel for sanitized contract validation failures."""


def _raise_contract_error_if_invalid(
    validator: Callable[[object], None],
    value: object,
) -> None:
    try:
        validator(value)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        raise ValueError(_CONTRACT_ERROR_MESSAGE) from None


def _validate_protocol_identity(value: object) -> None:
    if type(value) is not CalibrationProtocolIdentity:
        raise _CalibrationContractError
    if value.protocol_identifier != CALIBRATION_PROTOCOL_IDENTIFIER:
        raise _CalibrationContractError
    if value.calibration_contract_version != CALIBRATION_CONTRACT_VERSION:
        raise _CalibrationContractError
    if not _is_sha(value.repository_commit_sha):
        raise _CalibrationContractError


def _validate_processing_provenance(value: object) -> None:
    if type(value) is not CalibrationProcessingProvenance:
        raise _CalibrationContractError
    if value.preprocessing_contract_version != PREPROCESSING_CONTRACT_VERSION:
        raise _CalibrationContractError
    if value.embedding_contract_version != EMBEDDING_CONTRACT_VERSION:
        raise _CalibrationContractError
    if value.backend_version != SPEECHBRAIN_ECAPA_BACKEND_VERSION:
        raise _CalibrationContractError
    if value.model_identifier != SPEECHBRAIN_ECAPA_MODEL_ID:
        raise _CalibrationContractError
    if value.model_revision != SPEECHBRAIN_ECAPA_MODEL_REVISION:
        raise _CalibrationContractError
    if not _is_sha(value.model_revision):
        raise _CalibrationContractError
    if type(value.embedding_dimension) is not int:
        raise _CalibrationContractError
    if value.embedding_dimension != EMBEDDING_DIMENSION:
        raise _CalibrationContractError
    if type(value.input_sample_rate_hz) is not int:
        raise _CalibrationContractError
    if value.input_sample_rate_hz != TARGET_EMBEDDING_SAMPLE_RATE_HZ:
        raise _CalibrationContractError
    if type(value.normalized) is not bool or value.normalized is not False:
        raise _CalibrationContractError
    if value.comparison_version != SIMILARITY_COMPARISON_VERSION:
        raise _CalibrationContractError


def _validate_sample_record(value: object) -> None:
    if type(value) is not CalibrationSampleRecord:
        raise _CalibrationContractError
    _validate_identifier(value.sample_id, prefix="sample")
    _validate_identifier(value.subject_id, prefix="subject")
    _validate_identifier(value.source_group_id, prefix="source")
    if type(value.partition) is not CalibrationPartition:
        raise _CalibrationContractError


def _validate_pair_record(value: object) -> None:
    if type(value) is not CalibrationPairRecord:
        raise _CalibrationContractError
    _validate_identifier(value.pair_id, prefix="pair")
    _validate_identifier(value.reference_sample_id, prefix="sample")
    _validate_identifier(value.probe_sample_id, prefix="sample")
    if type(value.comparison_class) is not CalibrationComparisonClass:
        raise _CalibrationContractError
    if type(value.partition) is not CalibrationPartition:
        raise _CalibrationContractError


def _validate_experiment_plan(value: object) -> None:
    if type(value) is not CalibrationExperimentPlan:
        raise _CalibrationContractError
    _validate_protocol_identity(value.protocol)
    _validate_processing_provenance(value.provenance)
    if type(value.samples) is not tuple or type(value.pairs) is not tuple:
        raise _CalibrationContractError
    if not value.samples or not value.pairs:
        raise _CalibrationContractError

    samples_by_id: dict[str, CalibrationSampleRecord] = {}
    for sample in value.samples:
        _validate_sample_record(sample)
        if sample.sample_id in samples_by_id:
            raise _CalibrationContractError
        samples_by_id[sample.sample_id] = sample

    pair_ids: set[str] = set()
    for pair in value.pairs:
        _validate_pair_record(pair)
        if pair.pair_id in pair_ids:
            raise _CalibrationContractError
        pair_ids.add(pair.pair_id)
        _validate_pair_against_samples(pair, samples_by_id)


def _validate_pair_against_samples(
    pair: CalibrationPairRecord,
    samples_by_id: dict[str, CalibrationSampleRecord],
) -> None:
    reference = samples_by_id.get(pair.reference_sample_id)
    probe = samples_by_id.get(pair.probe_sample_id)
    if reference is None or probe is None:
        raise _CalibrationContractError
    if reference.sample_id == probe.sample_id:
        raise _CalibrationContractError
    if reference.source_group_id == probe.source_group_id:
        raise _CalibrationContractError
    if (
        pair.partition is not reference.partition
        or pair.partition is not probe.partition
    ):
        raise _CalibrationContractError
    if (
        pair.comparison_class is CalibrationComparisonClass.GENUINE
        and reference.subject_id != probe.subject_id
    ):
        raise _CalibrationContractError
    if (
        pair.comparison_class is CalibrationComparisonClass.IMPOSTOR
        and reference.subject_id == probe.subject_id
    ):
        raise _CalibrationContractError


def _validate_identifier(value: object, *, prefix: str) -> None:
    if type(value) is not str:
        raise _CalibrationContractError
    if not value or value != value.strip():
        raise _CalibrationContractError
    if not value.isascii() or len(value) > _IDENTIFIER_MAX_LENGTH:
        raise _CalibrationContractError
    if not value.startswith(f"{prefix}-"):
        raise _CalibrationContractError
    if not _IDENTIFIER_RE.fullmatch(value):
        raise _CalibrationContractError


def _is_sha(value: object) -> bool:
    return type(value) is str and _SHA_RE.fullmatch(value) is not None


__all__ = [
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
