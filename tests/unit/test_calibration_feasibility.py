"""Tests for the Phase 5B local feasibility probe."""

from __future__ import annotations

import json
import math
import struct
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import voiceid.calibration.feasibility as feasibility
from voiceid.audio import PREPROCESSING_CONTRACT_VERSION
from voiceid.audio.preprocessing import (
    PreprocessedAudioResult,
    PreprocessingIssue,
    PreprocessingStatus,
)
from voiceid.calibration.contracts import (
    CalibrationComparisonClass,
    CalibrationPartition,
    CalibrationSampleRecord,
)
from voiceid.calibration.feasibility import (
    FEASIBILITY_LABEL_CRITERIA_VERSION,
    MIN_FEASIBILITY_SCORES_PER_CLASS,
    NOT_PROMISING_MIN_EXPLORATORY_ERROR_RATE,
    PROMISING_MAX_EXPLORATORY_ERROR_RATE,
    FeasibilityLabel,
    FeasibilityProbeError,
    _calculate_threshold_metrics,
    _classify_feasibility_label,
    _generate_pairs,
    _increment_issue_counts,
    _summarize_probe,
    run_feasibility_probe,
)
from voiceid.calibration.reporting import (
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
from voiceid.embeddings.contracts import (
    EMBEDDING_DIMENSION,
    EmbeddingErrorCode,
    EmbeddingIssue,
    EmbeddingMetadata,
    EmbeddingStatus,
    SpeakerEmbeddingResult,
    build_invalid_embedding_result,
)
from voiceid.embeddings.policy import (
    SPEECHBRAIN_ECAPA_MODEL_ID,
    SPEECHBRAIN_ECAPA_MODEL_REVISION,
    TARGET_EMBEDDING_SAMPLE_RATE_HZ,
)
from voiceid.similarity import (
    SimilarityErrorCode,
    SimilarityIssue,
    SimilarityStatus,
    SpeakerSimilarityResult,
)

_BASELINE_SHA = "ab48bcf68b9fd7d2cd5f1edb98302b3a7a3b883d"
_SAMPLE_A1 = "smp_00000000000000000000000000000001"
_SAMPLE_A2 = "smp_00000000000000000000000000000002"
_SAMPLE_B1 = "smp_00000000000000000000000000000003"
_SAMPLE_B2 = "smp_00000000000000000000000000000004"
_SUBJECT_A = "sub_00000000000000000000000000000001"
_SUBJECT_B = "sub_00000000000000000000000000000002"
_SOURCE_1 = "src_00000000000000000000000000000001"
_SOURCE_2 = "src_00000000000000000000000000000002"
_SOURCE_3 = "src_00000000000000000000000000000003"
_SOURCE_4 = "src_00000000000000000000000000000004"


class _FakeEmbeddingService:
    def __init__(
        self,
        embeddings: tuple[np.ndarray, ...],
    ) -> None:
        self.calls = 0
        self.embeddings = embeddings

    def embed(self, _preprocessed_audio: object) -> SpeakerEmbeddingResult:
        embedding = self.embeddings[self.calls]
        self.calls += 1
        return _embedding_result(embedding)


def test_feasibility_probe_writes_privacy_safe_reports(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "reports"
    canary_dir = tmp_path / "TOKEN_PATH_CANARY"
    canary_dir.mkdir()
    wav_paths = {
        _SAMPLE_A1: canary_dir / "sample_a1_TOKEN_FILENAME_CANARY.wav",
        _SAMPLE_A2: tmp_path / "sample_a2.wav",
        _SAMPLE_B1: tmp_path / "sample_b1.wav",
        _SAMPLE_B2: tmp_path / "sample_b2.wav",
    }
    for wav_path in wav_paths.values():
        _write_wav(wav_path)
    _write_manifest(manifest_path, wav_paths)
    embeddings = (
        _unit_embedding(0, value=0.123456),
        _unit_embedding(0, value=0.123456),
        _unit_embedding(1, value=0.654321),
        _unit_embedding(1, value=0.654321),
    )

    result = run_feasibility_probe(
        manifest_path=manifest_path,
        output_dir=output_dir,
        embedding_service=_FakeEmbeddingService(embeddings),
    )

    assert result.label is FeasibilityLabel.PROMISING
    assert result.label_criteria_version == FEASIBILITY_LABEL_CRITERIA_VERSION
    assert result.report_files == (
        "scores.csv",
        "threshold_metrics.csv",
        "summary.csv",
        "report.html",
    )
    assert result.total_samples == 4
    assert result.generated_pairs == 6
    assert result.evaluated_scores == 6

    combined_report = "\n".join(
        (output_dir / file_name).read_text(encoding="utf-8")
        for file_name in result.report_files
    )
    for forbidden in (
        "TOKEN_PATH_CANARY",
        "TOKEN_FILENAME_CANARY",
        _SAMPLE_A1,
        _SUBJECT_A,
        _SOURCE_1,
        "pair_",
        "0.123456",
        "0.654321",
        str(tmp_path),
        "MATCH",
        "NO_MATCH",
    ):
        assert forbidden not in combined_report

    scores = (output_dir / "scores.csv").read_text(encoding="utf-8")
    assert "row_index,partition,comparison_class,score" in scores
    assert "GENUINE" in scores
    assert "IMPOSTOR" in scores
    assert "1.000000" in scores
    assert "0.000000" in scores


def test_manifest_error_is_generic_and_does_not_leak_path_or_id(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "repository_commit_sha": _BASELINE_SHA,
                "samples": [
                    {
                        "sample_id": "sample-person-name-session-01.wav",
                        "subject_id": _SUBJECT_A,
                        "source_group_id": _SOURCE_1,
                        "partition": "CALIBRATION",
                        "wav_path": "/Users/private/TOKEN_PATH_CANARY.wav",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FeasibilityProbeError) as exc_info:
        run_feasibility_probe(
            manifest_path=manifest_path,
            output_dir=tmp_path / "reports",
            embedding_service=_FakeEmbeddingService((_unit_embedding(0),)),
        )

    assert str(exc_info.value) == "Invalid feasibility manifest."
    assert "sample-person" not in str(exc_info.value)
    assert "TOKEN_PATH_CANARY" not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


def test_pair_generation_is_deterministic_and_obeys_subject_source_partition() -> None:
    samples = (
        _sample(_SAMPLE_B2, _SUBJECT_B, _SOURCE_4),
        _sample(_SAMPLE_A1, _SUBJECT_A, _SOURCE_1),
        _sample(_SAMPLE_B1, _SUBJECT_B, _SOURCE_3),
        _sample(_SAMPLE_A2, _SUBJECT_A, _SOURCE_2),
        _sample(
            "smp_00000000000000000000000000000005",
            _SUBJECT_A,
            _SOURCE_1,
        ),
        _sample(
            "smp_00000000000000000000000000000006",
            _SUBJECT_A,
            "src_00000000000000000000000000000006",
            partition=CalibrationPartition.HOLDOUT,
        ),
    )

    pairs1 = _generate_pairs(samples)
    pairs2 = _generate_pairs(samples)

    assert pairs1 == pairs2
    assert all(pair.partition is CalibrationPartition.CALIBRATION for pair in pairs1)
    assert all(pair.reference_sample_id != pair.probe_sample_id for pair in pairs1)
    assert any(
        pair.comparison_class is CalibrationComparisonClass.GENUINE for pair in pairs1
    )
    assert any(
        pair.comparison_class is CalibrationComparisonClass.IMPOSTOR for pair in pairs1
    )


def test_far_frr_metric_boundaries_are_deterministic() -> None:
    scores = (
        ScoreRecord(
            1,
            CalibrationPartition.CALIBRATION,
            CalibrationComparisonClass.GENUINE,
            0.8,
        ),
        ScoreRecord(
            2,
            CalibrationPartition.CALIBRATION,
            CalibrationComparisonClass.GENUINE,
            0.6,
        ),
        ScoreRecord(
            3,
            CalibrationPartition.CALIBRATION,
            CalibrationComparisonClass.IMPOSTOR,
            0.7,
        ),
        ScoreRecord(
            4,
            CalibrationPartition.CALIBRATION,
            CalibrationComparisonClass.IMPOSTOR,
            0.2,
        ),
    )

    metrics = _calculate_threshold_metrics(scores, thresholds=(0.7,))

    assert metrics == (
        ThresholdMetric(
            threshold=0.7,
            far=0.5,
            frr=0.5,
            false_accepts=1,
            false_rejects=1,
            impostor_total=2,
            genuine_total=2,
        ),
    )


def test_calibration_metrics_and_label_ignore_holdout_scores() -> None:
    calibration_scores = (
        _score(1, CalibrationComparisonClass.GENUINE, 0.9),
        _score(2, CalibrationComparisonClass.GENUINE, 0.8),
        _score(3, CalibrationComparisonClass.IMPOSTOR, -0.2),
        _score(4, CalibrationComparisonClass.IMPOSTOR, -0.1),
    )
    holdout_scores = (
        _score(
            5,
            CalibrationComparisonClass.GENUINE,
            -1.0,
            partition=CalibrationPartition.HOLDOUT,
        ),
        _score(
            6,
            CalibrationComparisonClass.GENUINE,
            -1.0,
            partition=CalibrationPartition.HOLDOUT,
        ),
        _score(
            7,
            CalibrationComparisonClass.IMPOSTOR,
            1.0,
            partition=CalibrationPartition.HOLDOUT,
        ),
        _score(
            8,
            CalibrationComparisonClass.IMPOSTOR,
            1.0,
            partition=CalibrationPartition.HOLDOUT,
        ),
    )
    thresholds = (0.5,)

    calibration_metrics = _calculate_threshold_metrics(
        calibration_scores,
        thresholds=thresholds,
    )
    combined_metrics = _calculate_threshold_metrics(
        calibration_scores + holdout_scores,
        thresholds=thresholds,
    )
    calibration_summary = _summarize_probe(
        total_samples=4,
        generated_pairs=4,
        scores=calibration_scores,
        threshold_metrics=calibration_metrics,
        invalid_counts=Counter(),
    )
    combined_summary = _summarize_probe(
        total_samples=8,
        generated_pairs=8,
        scores=calibration_scores + holdout_scores,
        threshold_metrics=combined_metrics,
        invalid_counts=Counter(),
    )

    assert combined_metrics == calibration_metrics
    assert combined_summary.label == calibration_summary.label == "PROMISING"
    assert combined_summary.overlap_low == calibration_summary.overlap_low
    assert (
        combined_summary.genuine_distribution
        == calibration_summary.genuine_distribution
    )
    assert (
        combined_summary.impostor_distribution
        == calibration_summary.impostor_distribution
    )


def test_calibration_html_histogram_ignores_many_holdout_variations(
    tmp_path: Path,
) -> None:
    calibration_scores = (
        _score(1, CalibrationComparisonClass.GENUINE, 0.9),
        _score(2, CalibrationComparisonClass.GENUINE, 0.8),
        _score(3, CalibrationComparisonClass.IMPOSTOR, -0.2),
        _score(4, CalibrationComparisonClass.IMPOSTOR, -0.1),
    )
    holdout_scores = tuple(
        _score(
            index + 5,
            (
                CalibrationComparisonClass.GENUINE
                if index % 2 == 0
                else CalibrationComparisonClass.IMPOSTOR
            ),
            -1.0 + (index % 101) / 50.0,
            partition=CalibrationPartition.HOLDOUT,
        )
        for index in range(101)
    )
    metrics = _calculate_threshold_metrics(calibration_scores, thresholds=(0.5,))
    summary = _summarize_probe(
        total_samples=105,
        generated_pairs=105,
        scores=calibration_scores + holdout_scores,
        threshold_metrics=metrics,
        invalid_counts=Counter(),
    )
    calibration_html = tmp_path / "calibration.html"
    combined_html = tmp_path / "combined.html"

    write_html_report(
        calibration_html,
        summary=summary,
        scores=calibration_scores,
        threshold_metrics=metrics,
    )
    write_html_report(
        combined_html,
        summary=summary,
        scores=calibration_scores + holdout_scores,
        threshold_metrics=metrics,
    )

    assert _calibration_histogram_section(
        calibration_html.read_text(encoding="utf-8")
    ) == _calibration_histogram_section(combined_html.read_text(encoding="utf-8"))
    assert "HOLDOUT scores are not used for exploratory metrics or label" in (
        combined_html.read_text(encoding="utf-8")
    )


def test_holdout_only_scores_do_not_fallback_to_calibration() -> None:
    holdout_scores = (
        _score(
            1,
            CalibrationComparisonClass.GENUINE,
            0.9,
            partition=CalibrationPartition.HOLDOUT,
        ),
        _score(
            2,
            CalibrationComparisonClass.GENUINE,
            0.8,
            partition=CalibrationPartition.HOLDOUT,
        ),
        _score(
            3,
            CalibrationComparisonClass.IMPOSTOR,
            -0.2,
            partition=CalibrationPartition.HOLDOUT,
        ),
        _score(
            4,
            CalibrationComparisonClass.IMPOSTOR,
            -0.1,
            partition=CalibrationPartition.HOLDOUT,
        ),
    )
    metrics = _calculate_threshold_metrics(holdout_scores, thresholds=(0.5,))
    summary = _summarize_probe(
        total_samples=4,
        generated_pairs=4,
        scores=holdout_scores,
        threshold_metrics=metrics,
        invalid_counts=Counter(),
    )

    assert summary.label == "INCONCLUSIVE"
    assert summary.genuine_distribution.count == 0
    assert summary.impostor_distribution.count == 0
    assert metrics == (ThresholdMetric(0.5, 0.0, 0.0, 0, 0, 0, 0),)


def test_label_requires_minimum_scores_per_class() -> None:
    metrics = (ThresholdMetric(0.5, 0.0, 0.0, 0, 0, 1, 1),)

    label = _classify_feasibility_label(
        genuine_scores=(0.9,) * (MIN_FEASIBILITY_SCORES_PER_CLASS - 1),
        impostor_scores=(0.1,) * MIN_FEASIBILITY_SCORES_PER_CLASS,
        threshold_metrics=metrics,
        overlap_low=None,
    )

    assert label is FeasibilityLabel.INCONCLUSIVE


def test_label_promising_boundary_allows_exact_error_threshold() -> None:
    metrics = (
        ThresholdMetric(
            0.5,
            PROMISING_MAX_EXPLORATORY_ERROR_RATE,
            PROMISING_MAX_EXPLORATORY_ERROR_RATE,
            1,
            1,
            10,
            10,
        ),
    )

    label = _classify_feasibility_label(
        genuine_scores=(0.7, 0.8),
        impostor_scores=(0.1, 0.2),
        threshold_metrics=metrics,
        overlap_low=None,
    )

    assert label is FeasibilityLabel.PROMISING


def test_label_not_promising_boundary_uses_best_balanced_error() -> None:
    metrics = (
        ThresholdMetric(
            0.5,
            NOT_PROMISING_MIN_EXPLORATORY_ERROR_RATE,
            0.1,
            7,
            2,
            20,
            20,
        ),
    )

    label = _classify_feasibility_label(
        genuine_scores=(0.4, 0.5),
        impostor_scores=(0.45, 0.55),
        threshold_metrics=metrics,
        overlap_low=0.45,
    )

    assert label is FeasibilityLabel.NOT_PROMISING


def test_label_is_inconclusive_for_middle_overlap_case() -> None:
    metrics = (ThresholdMetric(0.5, 0.2, 0.1, 2, 1, 10, 10),)

    label = _classify_feasibility_label(
        genuine_scores=(0.4, 0.8),
        impostor_scores=(0.3, 0.6),
        threshold_metrics=metrics,
        overlap_low=0.4,
    )

    assert label is FeasibilityLabel.INCONCLUSIVE


def test_missing_embedding_excludes_pairs_without_partial_score(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "reports"
    wav_paths = {
        _SAMPLE_A1: tmp_path / "a1.wav",
        _SAMPLE_A2: tmp_path / "a2.wav",
        _SAMPLE_B1: tmp_path / "b1.wav",
        _SAMPLE_B2: tmp_path / "b2.wav",
    }
    for wav_path in wav_paths.values():
        _write_wav(wav_path)
    _write_manifest(manifest_path, wav_paths)
    service = _FakeEmbeddingService(
        (
            _unit_embedding(0),
            _invalid_embedding_marker(),
            _unit_embedding(1),
            _unit_embedding(1),
        )
    )

    result = run_feasibility_probe(
        manifest_path=manifest_path,
        output_dir=output_dir,
        embedding_service=service,
    )

    assert result.label is FeasibilityLabel.INCONCLUSIVE
    assert result.evaluated_scores < result.generated_pairs
    assert result.invalid_counts["embedding.INFERENCE_FAILED"] == 1
    assert result.invalid_counts["pair_excluded.missing_embedding"] > 0


@pytest.mark.parametrize(
    ("stage", "issue_code"),
    (
        ("preprocessing", "TOKEN_PATH_CANARY"),
        ("embedding", "TOKEN_EMBEDDING_CANARY"),
        ("similarity", "TOKEN_SIMILARITY_CANARY"),
    ),
)
def test_unknown_issue_codes_are_allowlisted_to_generic_unknown(
    stage: str,
    issue_code: str,
) -> None:
    invalid_counts: Counter[str] = Counter()
    if stage == "preprocessing":
        issue: object = PreprocessingIssue(issue_code, "leak")
    elif stage == "embedding":
        issue = EmbeddingIssue(issue_code, "leak")
    else:
        issue = _forged_similarity_issue(issue_code)

    _increment_issue_counts(stage, (issue,), invalid_counts)

    assert invalid_counts == Counter({f"{stage}.unknown": 1})
    assert "TOKEN" not in repr(invalid_counts)


@pytest.mark.parametrize(
    ("stage", "issue"),
    (
        (
            "preprocessing",
            PreprocessingIssue("PREPROCESSING_ERROR", "TOKEN_MESSAGE_CANARY"),
        ),
        (
            "embedding",
            EmbeddingIssue("INFERENCE_FAILED", "TOKEN_MESSAGE_CANARY"),
        ),
        (
            "similarity",
            SimilarityIssue(
                "COMPARISON_ERROR",
                "Speaker embedding comparison failed safely.",
            ),
        ),
    ),
)
def test_known_issue_codes_preserve_only_stable_codes(
    stage: str,
    issue: object,
) -> None:
    invalid_counts: Counter[str] = Counter()

    _increment_issue_counts(stage, (issue,), invalid_counts)

    assert len(invalid_counts) == 1
    assert next(iter(invalid_counts.values())) == 1
    assert "TOKEN_MESSAGE_CANARY" not in repr(invalid_counts)


def test_malicious_issue_object_does_not_have_code_accessed() -> None:
    class MaliciousIssue:
        @property
        def code(self) -> str:
            raise AssertionError("code property should not be accessed")

    invalid_counts: Counter[str] = Counter()

    _increment_issue_counts("embedding", (MaliciousIssue(),), invalid_counts)

    assert invalid_counts == Counter({"embedding.unknown": 1})


@pytest.mark.parametrize("stage", ("preprocessing", "embedding", "similarity"))
def test_pipeline_sanitizes_unknown_stage_issue_code_canaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "reports"
    wav_paths = {
        _SAMPLE_A1: tmp_path / "a1.wav",
        _SAMPLE_A2: tmp_path / "a2.wav",
        _SAMPLE_B1: tmp_path / "b1.wav",
        _SAMPLE_B2: tmp_path / "b2.wav",
    }
    for wav_path in wav_paths.values():
        _write_wav(wav_path)
    _write_manifest(manifest_path, wav_paths)
    service: object = _FakeEmbeddingService(
        (
            _unit_embedding(0),
            _unit_embedding(0),
            _unit_embedding(1),
            _unit_embedding(1),
        )
    )

    if stage == "preprocessing":
        monkeypatch.setattr(
            feasibility,
            "preprocess_wav_file",
            lambda _path: PreprocessedAudioResult(
                status=PreprocessingStatus.INVALID,
                file_name="safe.wav",
                waveform=None,
                metadata=None,
                errors=(PreprocessingIssue("TOKEN_PREPROCESS_CANARY", "leak"),),
            ),
        )
    elif stage == "embedding":
        service = _UnknownIssueEmbeddingService()
    else:
        monkeypatch.setattr(
            feasibility,
            "compare_speaker_embeddings",
            lambda _reference, _probe: _unknown_issue_similarity_result(),
        )

    result = run_feasibility_probe(
        manifest_path=manifest_path,
        output_dir=output_dir,
        embedding_service=service,
    )

    assert result.invalid_counts[f"{stage}.unknown"] > 0
    combined_report = "\n".join(
        (output_dir / file_name).read_text(encoding="utf-8")
        for file_name in result.report_files
    )
    assert "TOKEN_" not in combined_report
    assert "CANARY" not in combined_report


def test_public_boundary_sanitizes_injected_feasibility_probe_error(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    wav_paths = {
        _SAMPLE_A1: tmp_path / "a1.wav",
        _SAMPLE_A2: tmp_path / "a2.wav",
        _SAMPLE_B1: tmp_path / "b1.wav",
        _SAMPLE_B2: tmp_path / "b2.wav",
    }
    for wav_path in wav_paths.values():
        _write_wav(wav_path)
    _write_manifest(manifest_path, wav_paths)

    class ExplodingService:
        def embed(self, _preprocessed_audio: object) -> SpeakerEmbeddingResult:
            raise FeasibilityProbeError(
                "/Users/private/TOKEN_PATH_CANARY waveform=[0.123456]"
            )

    with pytest.raises(FeasibilityProbeError) as exc_info:
        run_feasibility_probe(
            manifest_path=manifest_path,
            output_dir=tmp_path / "reports",
            embedding_service=ExplodingService(),
        )

    assert str(exc_info.value) == "Feasibility probe failed safely."
    assert "TOKEN_PATH_CANARY" not in str(exc_info.value)
    assert "0.123456" not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


@pytest.mark.parametrize("exception_type", (KeyboardInterrupt, SystemExit, MemoryError))
def test_public_boundary_does_not_mask_critical_exceptions(
    tmp_path: Path,
    exception_type: type[BaseException],
) -> None:
    manifest_path = tmp_path / "manifest.json"
    wav_paths = {
        _SAMPLE_A1: tmp_path / "a1.wav",
        _SAMPLE_A2: tmp_path / "a2.wav",
        _SAMPLE_B1: tmp_path / "b1.wav",
        _SAMPLE_B2: tmp_path / "b2.wav",
    }
    for wav_path in wav_paths.values():
        _write_wav(wav_path)
    _write_manifest(manifest_path, wav_paths)

    class ExplodingService:
        def embed(self, _preprocessed_audio: object) -> SpeakerEmbeddingResult:
            raise exception_type

    with pytest.raises(exception_type):
        run_feasibility_probe(
            manifest_path=manifest_path,
            output_dir=tmp_path / "reports",
            embedding_service=ExplodingService(),
        )


@pytest.mark.parametrize(
    "payload",
    (
        "<script>TOKEN_ELEMENT_CANARY</script>",
        '" autofocus onfocus="TOKEN_ATTRIBUTE_CANARY',
        "&TOKEN_ENTITY_CANARY;",
    ),
)
def test_html_report_rejects_forged_dynamic_payloads(
    tmp_path: Path,
    payload: str,
) -> None:
    summary = _summary()
    object.__setattr__(summary, "invalid_counts", Counter({payload: 1}))

    with pytest.raises(ValueError) as exc_info:
        write_html_report(
            tmp_path / "report.html",
            summary=summary,
            scores=(_score(1, CalibrationComparisonClass.GENUINE, 0.8),),
            threshold_metrics=(ThresholdMetric(0.5, 0.0, 0.0, 0, 0, 1, 1),),
        )

    assert str(exc_info.value) == "Invalid report summary."
    assert "TOKEN" not in str(exc_info.value)
    assert not (tmp_path / "report.html").exists()


@pytest.mark.parametrize(
    ("field", "payload"),
    (
        ("partition", "/Users/private/TOKEN_PARTITION_CANARY"),
        ("comparison_class", "TOKEN_CLASS_CANARY"),
        ("score", math.nan),
        ("score", math.inf),
        ("score", True),
    ),
)
def test_score_csv_revalidates_mutated_score_record_without_partial_write(
    tmp_path: Path,
    field: str,
    payload: object,
) -> None:
    score = _score(1, CalibrationComparisonClass.GENUINE, 0.8)
    object.__setattr__(score, field, payload)
    output = tmp_path / "scores.csv"
    output.write_text("SAFE_OLD_CONTENT", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        write_score_csv(output, (score,))

    assert "TOKEN" not in str(exc_info.value)
    assert output.read_text(encoding="utf-8") == "SAFE_OLD_CONTENT"


@pytest.mark.parametrize(
    ("field", "payload"),
    (
        ("threshold", math.nan),
        ("threshold", math.inf),
        ("threshold", True),
        ("far", math.nan),
        ("far", math.inf),
        ("far", True),
        ("frr", math.nan),
        ("frr", math.inf),
        ("frr", True),
        ("false_accepts", True),
        ("false_rejects", -1),
        ("impostor_total", True),
        ("genuine_total", -1),
    ),
)
def test_threshold_csv_revalidates_mutated_metric_without_partial_write(
    tmp_path: Path,
    field: str,
    payload: object,
) -> None:
    metric = ThresholdMetric(0.5, 0.0, 0.0, 0, 0, 1, 1)
    object.__setattr__(metric, field, payload)
    output = tmp_path / "threshold_metrics.csv"
    output.write_text("SAFE_OLD_CONTENT", encoding="utf-8")

    with pytest.raises(ValueError):
        write_threshold_metrics_csv(output, (metric,))

    assert output.read_text(encoding="utf-8") == "SAFE_OLD_CONTENT"


@pytest.mark.parametrize(
    ("field", "payload"),
    (
        ("count", True),
        ("minimum", math.nan),
        ("maximum", math.inf),
        ("mean", True),
        ("median", -math.inf),
    ),
)
def test_summary_csv_revalidates_mutated_nested_distribution_without_partial_write(
    tmp_path: Path,
    field: str,
    payload: object,
) -> None:
    summary = _summary()
    object.__setattr__(summary.genuine_distribution, field, payload)
    output = tmp_path / "summary.csv"
    output.write_text("SAFE_OLD_CONTENT", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        write_summary_csv(output, summary)

    assert "TOKEN" not in str(exc_info.value)
    assert output.read_text(encoding="utf-8") == "SAFE_OLD_CONTENT"


@pytest.mark.parametrize(
    ("field", "payload"),
    (
        ("label", "TOKEN_LABEL_CANARY"),
        ("label_criteria_version", True),
        ("total_samples", True),
        ("generated_pairs", -1),
        ("evaluated_scores", True),
        ("overlap_low", math.nan),
        ("overlap_high", math.inf),
    ),
)
def test_summary_csv_revalidates_mutated_summary_fields_without_partial_write(
    tmp_path: Path,
    field: str,
    payload: object,
) -> None:
    summary = _summary()
    object.__setattr__(summary, field, payload)
    output = tmp_path / "summary.csv"
    output.write_text("SAFE_OLD_CONTENT", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        write_summary_csv(output, summary)

    assert "TOKEN" not in str(exc_info.value)
    assert output.read_text(encoding="utf-8") == "SAFE_OLD_CONTENT"


def test_summary_csv_rejects_forged_invalid_counts_without_partial_write(
    tmp_path: Path,
) -> None:
    summary = _summary()
    object.__setattr__(
        summary,
        "invalid_counts",
        Counter({"<script>TOKEN_COUNT_CANARY</script>": True}),
    )
    output = tmp_path / "summary.csv"
    output.write_text("SAFE_OLD_CONTENT", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        write_summary_csv(output, summary)

    assert str(exc_info.value) == "Invalid report summary."
    assert "TOKEN" not in str(exc_info.value)
    assert output.read_text(encoding="utf-8") == "SAFE_OLD_CONTENT"


def test_report_serializers_reject_forged_subclasses_without_partial_write(
    tmp_path: Path,
) -> None:
    class ScoreRecordSubclass(ScoreRecord):
        pass

    class CounterSubclass(Counter[str]):
        pass

    with pytest.raises(ValueError):
        ScoreRecordSubclass(
            1,
            CalibrationPartition.CALIBRATION,
            CalibrationComparisonClass.GENUINE,
            0.8,
        )
    score = object.__new__(ScoreRecordSubclass)
    object.__setattr__(score, "row_index", 1)
    object.__setattr__(score, "partition", CalibrationPartition.CALIBRATION)
    object.__setattr__(score, "comparison_class", CalibrationComparisonClass.GENUINE)
    object.__setattr__(score, "score", 0.8)
    summary = _summary()
    object.__setattr__(summary, "invalid_counts", CounterSubclass({"safe.code": 1}))
    score_output = tmp_path / "scores.csv"
    summary_output = tmp_path / "summary.csv"
    score_output.write_text("SAFE_OLD_CONTENT", encoding="utf-8")
    summary_output.write_text("SAFE_OLD_CONTENT", encoding="utf-8")

    with pytest.raises(ValueError):
        write_score_csv(score_output, (score,))
    with pytest.raises(ValueError):
        write_summary_csv(summary_output, summary)

    assert score_output.read_text(encoding="utf-8") == "SAFE_OLD_CONTENT"
    assert summary_output.read_text(encoding="utf-8") == "SAFE_OLD_CONTENT"


def test_html_report_rejects_forged_mapping_subclass_and_bool_counts(
    tmp_path: Path,
) -> None:
    class CounterSubclass(Counter[str]):
        pass

    summary = _summary()
    object.__setattr__(summary, "invalid_counts", CounterSubclass({"safe.code": True}))

    with pytest.raises(ValueError) as exc_info:
        write_html_report(
            tmp_path / "report.html",
            summary=summary,
            scores=(_score(1, CalibrationComparisonClass.GENUINE, 0.8),),
            threshold_metrics=(ThresholdMetric(0.5, 0.0, 0.0, 0, 0, 1, 1),),
        )

    assert str(exc_info.value) == "Invalid report summary."


def test_report_contract_rejects_bool_numeric_fields() -> None:
    with pytest.raises(ValueError):
        ScoreRecord(
            True,
            CalibrationPartition.CALIBRATION,
            CalibrationComparisonClass.GENUINE,
            0.5,
        )

    with pytest.raises(ValueError):
        ThresholdMetric(0.5, 0.0, 0.0, True, 0, 1, 1)

    with pytest.raises(ValueError):
        ScoreDistribution(True, None, None, None, None)


def _embedding_result(embedding: np.ndarray) -> SpeakerEmbeddingResult:
    if embedding.shape != (EMBEDDING_DIMENSION,):
        return build_invalid_embedding_result(
            code=EmbeddingErrorCode.INFERENCE_FAILED,
            message="Speaker embedding inference failed safely.",
        )
    return SpeakerEmbeddingResult(
        status=EmbeddingStatus.VALID,
        embedding=embedding.astype(np.float32, copy=True),
        metadata=EmbeddingMetadata(
            embedding_dimension=EMBEDDING_DIMENSION,
            model_identifier=SPEECHBRAIN_ECAPA_MODEL_ID,
            model_revision=SPEECHBRAIN_ECAPA_MODEL_REVISION,
            backend_name="speechbrain-ecapa-tdnn",
            backend_version=SPEECHBRAIN_ECAPA_BACKEND_VERSION,
            preprocessing_contract_version=PREPROCESSING_CONTRACT_VERSION,
            embedding_contract_version=EMBEDDING_CONTRACT_VERSION,
            device="cpu",
            input_sample_rate_hz=TARGET_EMBEDDING_SAMPLE_RATE_HZ,
            input_samples=16000,
            input_duration_seconds=1.0,
            normalized=False,
        ),
        errors=(),
    )


def _invalid_embedding_marker() -> np.ndarray:
    return np.zeros(1, dtype=np.float32)


class _UnknownIssueEmbeddingService:
    def embed(self, _preprocessed_audio: object) -> SpeakerEmbeddingResult:
        return SpeakerEmbeddingResult(
            status=EmbeddingStatus.INVALID,
            embedding=None,
            metadata=None,
            errors=(EmbeddingIssue("TOKEN_EMBED_CANARY", "leak"),),
        )


def _unknown_issue_similarity_result() -> SpeakerSimilarityResult:
    result = SpeakerSimilarityResult(
        status=SimilarityStatus.INVALID,
        similarity=None,
        metadata=None,
        errors=(
            SimilarityIssue(
                SimilarityErrorCode.COMPARISON_ERROR.value,
                "Speaker embedding comparison failed safely.",
            ),
        ),
    )
    object.__setattr__(result.errors[0], "code", "TOKEN_SIMILARITY_CANARY")
    object.__setattr__(result.errors[0], "message", "leak")
    return result


def _unit_embedding(index: int, *, value: float = 1.0) -> np.ndarray:
    embedding = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    embedding[index] = np.float32(value)
    norm = math.sqrt(float(np.sum(np.square(embedding, dtype=np.float64))))
    return (embedding / np.float32(norm)).astype(np.float32)


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


def _score(
    row_index: int,
    comparison_class: CalibrationComparisonClass,
    score: float,
    *,
    partition: CalibrationPartition = CalibrationPartition.CALIBRATION,
) -> ScoreRecord:
    return ScoreRecord(
        row_index=row_index,
        partition=partition,
        comparison_class=comparison_class,
        score=score,
    )


def _calibration_histogram_section(document: str) -> str:
    start = document.index("<h2>CALIBRATION Histogram</h2>")
    end = document.index("<h2>Exploratory Threshold Metrics</h2>")
    return document[start:end]


def _summary() -> FeasibilityReportSummary:
    return FeasibilityReportSummary(
        label="INCONCLUSIVE",
        label_criteria_version=FEASIBILITY_LABEL_CRITERIA_VERSION,
        total_samples=4,
        generated_pairs=6,
        evaluated_scores=1,
        genuine_distribution=ScoreDistribution(
            count=1,
            minimum=0.8,
            maximum=0.8,
            mean=0.8,
            median=0.8,
        ),
        impostor_distribution=ScoreDistribution(
            count=0,
            minimum=None,
            maximum=None,
            mean=None,
            median=None,
        ),
        overlap_low=None,
        overlap_high=None,
        invalid_counts=Counter({"embedding.unknown": 1}),
    )


def _forged_similarity_issue(code: str) -> SimilarityIssue:
    issue = SimilarityIssue(
        SimilarityErrorCode.COMPARISON_ERROR.value,
        "Speaker embedding comparison failed safely.",
    )
    object.__setattr__(issue, "code", code)
    object.__setattr__(issue, "message", "TOKEN_MESSAGE_CANARY")
    return issue


def _write_manifest(manifest_path: Path, wav_paths: dict[str, Path]) -> None:
    samples = [
        {
            "sample_id": _SAMPLE_A1,
            "subject_id": _SUBJECT_A,
            "source_group_id": _SOURCE_1,
            "partition": "CALIBRATION",
            "wav_path": str(wav_paths[_SAMPLE_A1]),
        },
        {
            "sample_id": _SAMPLE_A2,
            "subject_id": _SUBJECT_A,
            "source_group_id": _SOURCE_2,
            "partition": "CALIBRATION",
            "wav_path": str(wav_paths[_SAMPLE_A2]),
        },
        {
            "sample_id": _SAMPLE_B1,
            "subject_id": _SUBJECT_B,
            "source_group_id": _SOURCE_3,
            "partition": "CALIBRATION",
            "wav_path": str(wav_paths[_SAMPLE_B1]),
        },
        {
            "sample_id": _SAMPLE_B2,
            "subject_id": _SUBJECT_B,
            "source_group_id": _SOURCE_4,
            "partition": "CALIBRATION",
            "wav_path": str(wav_paths[_SAMPLE_B2]),
        },
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "repository_commit_sha": _BASELINE_SHA,
                "samples": samples,
                "thresholds": [0.5],
            }
        ),
        encoding="utf-8",
    )


def _write_wav(path: Path) -> None:
    sample_rate = 16000
    total_samples = sample_rate
    frames = bytearray()
    for index in range(total_samples):
        sample = int(12000 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate))
        frames.extend(struct.pack("<h", sample))

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))
