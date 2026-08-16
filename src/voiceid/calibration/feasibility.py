"""Opt-in local feasibility probe for Phase 5B calibration planning.

This module intentionally keeps file paths inside the manifest adapter. It
writes exploratory aggregate reports only: no paths, pseudonymous identifiers,
waveforms, embeddings, or production identity decisions are serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol, cast

from voiceid.audio import PREPROCESSING_CONTRACT_VERSION
from voiceid.audio.preprocessing import (
    PreprocessedAudioResult,
    PreprocessingErrorCode,
    PreprocessingIssue,
)
from voiceid.calibration.contracts import (
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
from voiceid.calibration.reporting import (
    FEASIBILITY_LABEL_CRITERIA_VERSION,
    FeasibilityReportSummary,
    ScoreDistribution,
    ScoreRecord,
    ThresholdMetric,
    write_html_report,
    write_score_csv,
    write_summary_csv,
    write_threshold_metrics_csv,
)
from voiceid.embeddings import (
    EMBEDDING_CONTRACT_VERSION,
    SPEECHBRAIN_ECAPA_BACKEND_VERSION,
)
from voiceid.embeddings.backends.speechbrain_ecapa import (
    SpeechBrainEcapaBackendFactory,
    default_speechbrain_ecapa_config,
)
from voiceid.embeddings.contracts import (
    EMBEDDING_DIMENSION,
    EmbeddingErrorCode,
    EmbeddingIssue,
    SpeakerEmbeddingResult,
)
from voiceid.embeddings.loader import EmbeddingModelLoader
from voiceid.embeddings.policy import (
    SPEECHBRAIN_ECAPA_MODEL_ID,
    SPEECHBRAIN_ECAPA_MODEL_REVISION,
    TARGET_EMBEDDING_SAMPLE_RATE_HZ,
)
from voiceid.services.audio_preprocessing import preprocess_wav_file
from voiceid.services.speaker_embedding import SpeakerEmbeddingService
from voiceid.similarity import (
    SIMILARITY_COMPARISON_VERSION,
    SimilarityErrorCode,
    SimilarityIssue,
    compare_speaker_embeddings,
)

MIN_FEASIBILITY_SCORES_PER_CLASS: Final = 2
PROMISING_MAX_EXPLORATORY_ERROR_RATE: Final = 0.10
NOT_PROMISING_MIN_EXPLORATORY_ERROR_RATE: Final = 0.35
DEFAULT_EXPLORATORY_THRESHOLDS: Final = tuple(index / 20 for index in range(-20, 21))

_MANIFEST_ERROR_MESSAGE: Final = "Invalid feasibility manifest."
_PROBE_ERROR_MESSAGE: Final = "Feasibility probe failed safely."
_REPORT_FILES: Final = (
    "scores.csv",
    "threshold_metrics.csv",
    "summary.csv",
    "report.html",
)
_TRUSTED_PREPROCESSING_CODES: Final = frozenset(
    code.value for code in PreprocessingErrorCode
)
_TRUSTED_EMBEDDING_CODES: Final = frozenset(code.value for code in EmbeddingErrorCode)
_TRUSTED_SIMILARITY_CODES: Final = frozenset(code.value for code in SimilarityErrorCode)


class FeasibilityLabel(StrEnum):
    """Allowed exploratory feasibility labels."""

    PROMISING = "PROMISING"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_PROMISING = "NOT_PROMISING"


class FeasibilityProbeError(ValueError):
    """Stable, privacy-safe feasibility probe failure."""


class _FeasibilityInternalError(Exception):
    """Private sentinel for controlled internal feasibility failures."""

    def __init__(self, public_message: str) -> None:
        super().__init__()
        self.public_message = public_message


class _EmbeddingService(Protocol):
    def embed(
        self,
        preprocessed_audio: PreprocessedAudioResult,
    ) -> SpeakerEmbeddingResult:
        """Return a speaker embedding for already preprocessed audio."""


@dataclass(frozen=True, slots=True)
class FeasibilityProbeResult:
    """Privacy-safe result for a local feasibility probe run."""

    label: FeasibilityLabel
    label_criteria_version: str
    report_files: tuple[str, ...]
    total_samples: int
    generated_pairs: int
    evaluated_scores: int
    invalid_counts: Counter[str]


@dataclass(frozen=True, slots=True)
class _ManifestSample:
    record: CalibrationSampleRecord
    wav_path: Path

    def __repr__(self) -> str:
        return "_ManifestSample(redacted=True)"

    def __str__(self) -> str:
        return repr(self)


@dataclass(frozen=True, slots=True)
class _Manifest:
    repository_commit_sha: str
    samples: tuple[_ManifestSample, ...]
    thresholds: tuple[float, ...]


def run_feasibility_probe(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    model_cache_dir: str | Path | None = None,
    embedding_service: _EmbeddingService | None = None,
) -> FeasibilityProbeResult:
    """Run the local exploratory feasibility probe and write safe reports."""

    try:
        return _run_feasibility_probe(
            manifest_path=manifest_path,
            output_dir=output_dir,
            model_cache_dir=model_cache_dir,
            embedding_service=embedding_service,
        )
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except _FeasibilityInternalError as exc:
        raise FeasibilityProbeError(exc.public_message) from None
    except Exception:
        raise FeasibilityProbeError(_PROBE_ERROR_MESSAGE) from None


def _run_feasibility_probe(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    model_cache_dir: str | Path | None,
    embedding_service: _EmbeddingService | None,
) -> FeasibilityProbeResult:
    manifest_file = Path(manifest_path)
    manifest = _load_manifest(manifest_file)
    pairs = _generate_pairs(tuple(sample.record for sample in manifest.samples))
    plan = _build_experiment_plan(manifest=manifest, pairs=pairs)
    validate_calibration_experiment_plan(plan)

    service = embedding_service or _build_offline_embedding_service(model_cache_dir)
    invalid_counts: Counter[str] = Counter()
    embeddings = _extract_embeddings_in_memory(
        manifest.samples,
        service=service,
        invalid_counts=invalid_counts,
    )
    scores = _score_pairs(
        pairs,
        embeddings=embeddings,
        invalid_counts=invalid_counts,
    )
    threshold_metrics = _calculate_threshold_metrics(
        scores,
        thresholds=manifest.thresholds,
    )
    summary = _summarize_probe(
        total_samples=len(manifest.samples),
        generated_pairs=len(pairs),
        scores=scores,
        threshold_metrics=threshold_metrics,
        invalid_counts=invalid_counts,
    )
    _write_reports(
        Path(output_dir),
        summary=summary,
        scores=scores,
        threshold_metrics=threshold_metrics,
    )
    return FeasibilityProbeResult(
        label=FeasibilityLabel(summary.label),
        label_criteria_version=FEASIBILITY_LABEL_CRITERIA_VERSION,
        report_files=_REPORT_FILES,
        total_samples=len(manifest.samples),
        generated_pairs=len(pairs),
        evaluated_scores=len(scores),
        invalid_counts=Counter(invalid_counts),
    )


def _load_manifest(manifest_path: Path) -> _Manifest:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _parse_manifest(raw, base_dir=manifest_path.parent)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        raise _FeasibilityInternalError(_MANIFEST_ERROR_MESSAGE) from None


def _parse_manifest(raw: object, *, base_dir: Path) -> _Manifest:
    if type(raw) is not dict:
        raise _FeasibilityInternalError(_MANIFEST_ERROR_MESSAGE)
    manifest = cast(dict[str, object], raw)
    if not set(manifest).issubset({"repository_commit_sha", "samples", "thresholds"}):
        raise _FeasibilityInternalError(_MANIFEST_ERROR_MESSAGE)
    repository_commit_sha = manifest.get("repository_commit_sha")
    samples = manifest.get("samples")
    thresholds = manifest.get("thresholds", list(DEFAULT_EXPLORATORY_THRESHOLDS))
    if type(repository_commit_sha) is not str or type(samples) is not list:
        raise _FeasibilityInternalError(_MANIFEST_ERROR_MESSAGE)

    parsed_samples = tuple(
        _parse_manifest_sample(sample, base_dir=base_dir) for sample in samples
    )
    parsed_thresholds = _parse_thresholds(thresholds)
    if not parsed_samples:
        raise _FeasibilityInternalError(_MANIFEST_ERROR_MESSAGE)
    return _Manifest(
        repository_commit_sha=repository_commit_sha,
        samples=parsed_samples,
        thresholds=parsed_thresholds,
    )


def _parse_manifest_sample(raw: object, *, base_dir: Path) -> _ManifestSample:
    if type(raw) is not dict:
        raise _FeasibilityInternalError(_MANIFEST_ERROR_MESSAGE)
    sample = cast(dict[str, object], raw)
    if set(sample) != {
        "sample_id",
        "subject_id",
        "source_group_id",
        "partition",
        "wav_path",
    }:
        raise _FeasibilityInternalError(_MANIFEST_ERROR_MESSAGE)
    sample_id = sample["sample_id"]
    subject_id = sample["subject_id"]
    source_group_id = sample["source_group_id"]
    partition = sample["partition"]
    wav_path = sample["wav_path"]
    if (
        type(sample_id) is not str
        or type(subject_id) is not str
        or type(source_group_id) is not str
        or type(partition) is not str
        or type(wav_path) is not str
        or not wav_path
    ):
        raise _FeasibilityInternalError(_MANIFEST_ERROR_MESSAGE)

    path = Path(wav_path)
    if not path.is_absolute():
        path = base_dir / path
    return _ManifestSample(
        record=CalibrationSampleRecord(
            sample_id=sample_id,
            subject_id=subject_id,
            source_group_id=source_group_id,
            partition=CalibrationPartition(partition),
        ),
        wav_path=path,
    )


def _parse_thresholds(raw: object) -> tuple[float, ...]:
    if type(raw) is not list or not raw:
        raise _FeasibilityInternalError(_MANIFEST_ERROR_MESSAGE)
    thresholds: list[float] = []
    for threshold in raw:
        if type(threshold) not in {int, float} or type(threshold) is bool:
            raise _FeasibilityInternalError(_MANIFEST_ERROR_MESSAGE)
        value = float(threshold)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise _FeasibilityInternalError(_MANIFEST_ERROR_MESSAGE)
        thresholds.append(value)
    return tuple(sorted(set(thresholds)))


def _build_experiment_plan(
    *,
    manifest: _Manifest,
    pairs: tuple[CalibrationPairRecord, ...],
) -> CalibrationExperimentPlan:
    return CalibrationExperimentPlan(
        protocol=CalibrationProtocolIdentity(
            protocol_identifier=CALIBRATION_PROTOCOL_IDENTIFIER,
            calibration_contract_version=CALIBRATION_CONTRACT_VERSION,
            repository_commit_sha=manifest.repository_commit_sha,
        ),
        provenance=CalibrationProcessingProvenance(
            preprocessing_contract_version=PREPROCESSING_CONTRACT_VERSION,
            embedding_contract_version=EMBEDDING_CONTRACT_VERSION,
            backend_version=SPEECHBRAIN_ECAPA_BACKEND_VERSION,
            model_identifier=SPEECHBRAIN_ECAPA_MODEL_ID,
            model_revision=SPEECHBRAIN_ECAPA_MODEL_REVISION,
            embedding_dimension=EMBEDDING_DIMENSION,
            input_sample_rate_hz=TARGET_EMBEDDING_SAMPLE_RATE_HZ,
            normalized=False,
            comparison_version=SIMILARITY_COMPARISON_VERSION,
        ),
        samples=tuple(sample.record for sample in manifest.samples),
        pairs=pairs,
    )


def _generate_pairs(
    samples: tuple[CalibrationSampleRecord, ...],
) -> tuple[CalibrationPairRecord, ...]:
    ordered = tuple(
        sorted(
            samples,
            key=lambda sample: (
                sample.partition.value,
                sample.subject_id,
                sample.source_group_id,
                sample.sample_id,
            ),
        )
    )
    pairs: list[CalibrationPairRecord] = []
    for index, reference in enumerate(ordered):
        for probe in ordered[index + 1 :]:
            if reference.partition is not probe.partition:
                continue
            if reference.source_group_id == probe.source_group_id:
                continue
            comparison_class = (
                CalibrationComparisonClass.GENUINE
                if reference.subject_id == probe.subject_id
                else CalibrationComparisonClass.IMPOSTOR
            )
            pairs.append(
                CalibrationPairRecord(
                    pair_id=_deterministic_pair_id(
                        reference=reference,
                        probe=probe,
                        comparison_class=comparison_class,
                    ),
                    reference_sample_id=reference.sample_id,
                    probe_sample_id=probe.sample_id,
                    comparison_class=comparison_class,
                    partition=reference.partition,
                )
            )
    return tuple(pairs)


def _deterministic_pair_id(
    *,
    reference: CalibrationSampleRecord,
    probe: CalibrationSampleRecord,
    comparison_class: CalibrationComparisonClass,
) -> str:
    payload = "|".join(
        (
            reference.sample_id,
            probe.sample_id,
            comparison_class.value,
            reference.partition.value,
        )
    )
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()[:32]
    return f"pair_{digest}"


def _build_offline_embedding_service(
    model_cache_dir: str | Path | None,
) -> SpeakerEmbeddingService:
    if model_cache_dir is None:
        raise _FeasibilityInternalError(_PROBE_ERROR_MESSAGE)
    config = default_speechbrain_ecapa_config(cache_dir=model_cache_dir, offline=True)
    return SpeakerEmbeddingService(
        loader=EmbeddingModelLoader(SpeechBrainEcapaBackendFactory(config))
    )


def _extract_embeddings_in_memory(
    samples: tuple[_ManifestSample, ...],
    *,
    service: _EmbeddingService,
    invalid_counts: Counter[str],
) -> dict[str, SpeakerEmbeddingResult]:
    embeddings: dict[str, SpeakerEmbeddingResult] = {}
    for sample in sorted(samples, key=lambda item: item.record.sample_id):
        preprocessed = preprocess_wav_file(sample.wav_path)
        if not preprocessed.is_valid:
            _increment_issue_counts(
                "preprocessing",
                preprocessed.errors,
                invalid_counts,
            )
            continue

        embedding = service.embed(preprocessed)
        if not embedding.is_valid:
            _increment_issue_counts("embedding", embedding.errors, invalid_counts)
            continue

        embeddings[sample.record.sample_id] = embedding
    return embeddings


def _score_pairs(
    pairs: tuple[CalibrationPairRecord, ...],
    *,
    embeddings: dict[str, SpeakerEmbeddingResult],
    invalid_counts: Counter[str],
) -> tuple[ScoreRecord, ...]:
    scores: list[ScoreRecord] = []
    for pair in pairs:
        reference = embeddings.get(pair.reference_sample_id)
        probe = embeddings.get(pair.probe_sample_id)
        if reference is None or probe is None:
            invalid_counts["pair_excluded.missing_embedding"] += 1
            continue

        similarity = compare_speaker_embeddings(reference, probe)
        if not similarity.is_valid or similarity.similarity is None:
            _increment_issue_counts("similarity", similarity.errors, invalid_counts)
            continue
        scores.append(
            ScoreRecord(
                row_index=len(scores) + 1,
                partition=pair.partition,
                comparison_class=pair.comparison_class,
                score=similarity.similarity,
            )
        )
    return tuple(scores)


def _increment_issue_counts(
    prefix: str,
    issues: tuple[object, ...],
    invalid_counts: Counter[str],
) -> None:
    if not issues:
        invalid_counts[f"{prefix}.unknown"] += 1
        return
    for issue in issues:
        code = _trusted_issue_code(prefix=prefix, issue=issue)
        if code is not None:
            invalid_counts[f"{prefix}.{code}"] += 1
        else:
            invalid_counts[f"{prefix}.unknown"] += 1


def _trusted_issue_code(*, prefix: str, issue: object) -> str | None:
    if prefix == "preprocessing" and type(issue) is PreprocessingIssue:
        code = issue.code
        return (
            code if type(code) is str and code in _TRUSTED_PREPROCESSING_CODES else None
        )
    if prefix == "embedding" and type(issue) is EmbeddingIssue:
        code = issue.code
        return code if type(code) is str and code in _TRUSTED_EMBEDDING_CODES else None
    if prefix == "similarity" and type(issue) is SimilarityIssue:
        code = issue.code
        return code if type(code) is str and code in _TRUSTED_SIMILARITY_CODES else None
    return None


def _calibration_scores(scores: tuple[ScoreRecord, ...]) -> tuple[ScoreRecord, ...]:
    return tuple(
        score for score in scores if score.partition is CalibrationPartition.CALIBRATION
    )


def _calculate_threshold_metrics(
    scores: tuple[ScoreRecord, ...],
    *,
    thresholds: tuple[float, ...],
) -> tuple[ThresholdMetric, ...]:
    calibration_scores = _calibration_scores(scores)
    genuine_scores = tuple(
        score.score
        for score in calibration_scores
        if score.comparison_class is CalibrationComparisonClass.GENUINE
    )
    impostor_scores = tuple(
        score.score
        for score in calibration_scores
        if score.comparison_class is CalibrationComparisonClass.IMPOSTOR
    )
    metrics = []
    for threshold in thresholds:
        false_accepts = sum(score >= threshold for score in impostor_scores)
        false_rejects = sum(score < threshold for score in genuine_scores)
        metrics.append(
            ThresholdMetric(
                threshold=threshold,
                far=_rate(false_accepts, len(impostor_scores)),
                frr=_rate(false_rejects, len(genuine_scores)),
                false_accepts=false_accepts,
                false_rejects=false_rejects,
                impostor_total=len(impostor_scores),
                genuine_total=len(genuine_scores),
            )
        )
    return tuple(metrics)


def _summarize_probe(
    *,
    total_samples: int,
    generated_pairs: int,
    scores: tuple[ScoreRecord, ...],
    threshold_metrics: tuple[ThresholdMetric, ...],
    invalid_counts: Counter[str],
) -> FeasibilityReportSummary:
    calibration_scores = _calibration_scores(scores)
    genuine_scores = tuple(
        score.score
        for score in calibration_scores
        if score.comparison_class is CalibrationComparisonClass.GENUINE
    )
    impostor_scores = tuple(
        score.score
        for score in calibration_scores
        if score.comparison_class is CalibrationComparisonClass.IMPOSTOR
    )
    overlap_low, overlap_high = _score_overlap(genuine_scores, impostor_scores)
    label = _classify_feasibility_label(
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        threshold_metrics=threshold_metrics,
        overlap_low=overlap_low,
    )
    return FeasibilityReportSummary(
        label=label.value,
        label_criteria_version=FEASIBILITY_LABEL_CRITERIA_VERSION,
        total_samples=total_samples,
        generated_pairs=generated_pairs,
        evaluated_scores=len(scores),
        genuine_distribution=_distribution(genuine_scores),
        impostor_distribution=_distribution(impostor_scores),
        overlap_low=overlap_low,
        overlap_high=overlap_high,
        invalid_counts=Counter(invalid_counts),
    )


def _classify_feasibility_label(
    *,
    genuine_scores: tuple[float, ...],
    impostor_scores: tuple[float, ...],
    threshold_metrics: tuple[ThresholdMetric, ...],
    overlap_low: float | None,
) -> FeasibilityLabel:
    if (
        len(genuine_scores) < MIN_FEASIBILITY_SCORES_PER_CLASS
        or len(impostor_scores) < MIN_FEASIBILITY_SCORES_PER_CLASS
        or not threshold_metrics
    ):
        return FeasibilityLabel.INCONCLUSIVE

    best_balanced_error = min(
        max(metric.far, metric.frr) for metric in threshold_metrics
    )
    best_has_low_errors = any(
        metric.far <= PROMISING_MAX_EXPLORATORY_ERROR_RATE
        and metric.frr <= PROMISING_MAX_EXPLORATORY_ERROR_RATE
        for metric in threshold_metrics
    )
    if overlap_low is None and best_has_low_errors:
        return FeasibilityLabel.PROMISING
    if best_balanced_error >= NOT_PROMISING_MIN_EXPLORATORY_ERROR_RATE:
        return FeasibilityLabel.NOT_PROMISING
    return FeasibilityLabel.INCONCLUSIVE


def _score_overlap(
    genuine_scores: tuple[float, ...],
    impostor_scores: tuple[float, ...],
) -> tuple[float | None, float | None]:
    if not genuine_scores or not impostor_scores:
        return None, None
    low = max(min(genuine_scores), min(impostor_scores))
    high = min(max(genuine_scores), max(impostor_scores))
    if low <= high:
        return low, high
    return None, None


def _distribution(scores: tuple[float, ...]) -> ScoreDistribution:
    if not scores:
        return ScoreDistribution(
            count=0,
            minimum=None,
            maximum=None,
            mean=None,
            median=None,
        )
    return ScoreDistribution(
        count=len(scores),
        minimum=min(scores),
        maximum=max(scores),
        mean=statistics.fmean(scores),
        median=statistics.median(scores),
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _write_reports(
    output_dir: Path,
    *,
    summary: FeasibilityReportSummary,
    scores: tuple[ScoreRecord, ...],
    threshold_metrics: tuple[ThresholdMetric, ...],
) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_score_csv(output_dir / "scores.csv", scores)
        write_threshold_metrics_csv(
            output_dir / "threshold_metrics.csv",
            threshold_metrics,
        )
        write_summary_csv(output_dir / "summary.csv", summary)
        write_html_report(
            output_dir / "report.html",
            summary=summary,
            scores=scores,
            threshold_metrics=threshold_metrics,
        )
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        raise _FeasibilityInternalError(_PROBE_ERROR_MESSAGE) from None


def main(argv: list[str] | None = None) -> int:
    """Run the feasibility probe as an explicit local module command."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a local exploratory VoiceID feasibility probe. Reports contain "
            "raw cosine scores, not probability, confidence, or identity verdicts."
        )
    )
    parser.add_argument("manifest", help="Local JSON manifest path.")
    parser.add_argument("output_dir", help="Directory for CSV and HTML reports.")
    parser.add_argument(
        "--cache-dir",
        required=True,
        help=(
            "Prepared local SpeechBrain ECAPA cache directory. Downloads are disabled."
        ),
    )
    args = parser.parse_args(argv)
    result = run_feasibility_probe(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_cache_dir=args.cache_dir,
    )
    print(result.label.value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
