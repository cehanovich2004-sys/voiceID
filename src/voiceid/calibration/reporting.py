"""Privacy-safe CSV and HTML reports for feasibility probes."""

from __future__ import annotations

import csv
import html
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from voiceid.calibration.contracts import (
    CalibrationComparisonClass,
    CalibrationPartition,
)

PUBLIC_SCORE_DECIMALS: Final = 6
HISTOGRAM_BINS: Final = 20
_ALLOWED_LABELS: Final = frozenset({"PROMISING", "INCONCLUSIVE", "NOT_PROMISING"})
_SAFE_COUNT_CODE_CHARACTERS: Final = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._"
)


@dataclass(frozen=True, slots=True, repr=False)
class ScoreRecord:
    """One reportable score row without sample, subject, pair, or path IDs."""

    row_index: int
    partition: CalibrationPartition
    comparison_class: CalibrationComparisonClass
    score: float

    def __post_init__(self) -> None:
        """Validate the privacy-safe score row contract."""

        if type(self.row_index) is not int or self.row_index <= 0:
            raise ValueError("Invalid report score row.")
        if type(self.partition) is not CalibrationPartition:
            raise ValueError("Invalid report score row.")
        if type(self.comparison_class) is not CalibrationComparisonClass:
            raise ValueError("Invalid report score row.")
        _validate_score_float(self.score)

    def __repr__(self) -> str:
        """Return a safe representation without row values."""

        return "ScoreRecord(redacted=True)"


@dataclass(frozen=True, slots=True, repr=False)
class ThresholdMetric:
    """Exploratory FAR/FRR for one threshold."""

    threshold: float
    far: float
    frr: float
    false_accepts: int
    false_rejects: int
    impostor_total: int
    genuine_total: int

    def __post_init__(self) -> None:
        """Validate threshold metric fields before report serialization."""

        _validate_score_float(self.threshold)
        _validate_rate_float(self.far)
        _validate_rate_float(self.frr)
        for value in (
            self.false_accepts,
            self.false_rejects,
            self.impostor_total,
            self.genuine_total,
        ):
            _validate_count_int(value)
        if self.false_accepts > self.impostor_total:
            raise ValueError("Invalid report threshold metric.")
        if self.false_rejects > self.genuine_total:
            raise ValueError("Invalid report threshold metric.")

    def __repr__(self) -> str:
        """Return a safe representation without metric values."""

        return "ThresholdMetric(redacted=True)"


@dataclass(frozen=True, slots=True, repr=False)
class ScoreDistribution:
    """Aggregate score distribution without individual identities."""

    count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None

    def __post_init__(self) -> None:
        """Validate distribution fields before report serialization."""

        _validate_count_int(self.count)
        values = (self.minimum, self.maximum, self.mean, self.median)
        if self.count == 0:
            if values != (None, None, None, None):
                raise ValueError("Invalid report distribution.")
            return
        for value in values:
            if value is None:
                raise ValueError("Invalid report distribution.")
            _validate_score_float(value)
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("Invalid report distribution.")

    def __repr__(self) -> str:
        """Return a safe representation without distribution values."""

        return "ScoreDistribution(redacted=True)"


@dataclass(frozen=True, slots=True, repr=False)
class FeasibilityReportSummary:
    """Privacy-safe summary for CSV and HTML reports."""

    label: str
    label_criteria_version: str
    total_samples: int
    generated_pairs: int
    evaluated_scores: int
    genuine_distribution: ScoreDistribution
    impostor_distribution: ScoreDistribution
    overlap_low: float | None
    overlap_high: float | None
    invalid_counts: Counter[str]

    def __post_init__(self) -> None:
        """Validate summary fields before report serialization."""

        _validate_summary(self)

    def __repr__(self) -> str:
        """Return a safe representation without detailed counts."""

        return "FeasibilityReportSummary(redacted=True)"


def _validate_summary(summary: FeasibilityReportSummary) -> None:
    if type(summary) is not FeasibilityReportSummary:
        raise ValueError("Invalid report summary.")
    if type(summary.label) is not str or summary.label not in _ALLOWED_LABELS:
        raise ValueError("Invalid report summary.")
    if (
        type(summary.label_criteria_version) is not str
        or not summary.label_criteria_version
        or summary.label_criteria_version.strip() != summary.label_criteria_version
    ):
        raise ValueError("Invalid report summary.")
    for value in (
        summary.total_samples,
        summary.generated_pairs,
        summary.evaluated_scores,
    ):
        _validate_count_int(value)
    if type(summary.genuine_distribution) is not ScoreDistribution:
        raise ValueError("Invalid report summary.")
    if type(summary.impostor_distribution) is not ScoreDistribution:
        raise ValueError("Invalid report summary.")
    _validate_optional_score_float(summary.overlap_low)
    _validate_optional_score_float(summary.overlap_high)
    if (
        summary.overlap_low is not None
        and summary.overlap_high is not None
        and summary.overlap_low > summary.overlap_high
    ):
        raise ValueError("Invalid report summary.")
    _validate_invalid_counts(summary.invalid_counts)


def write_score_csv(path: Path, scores: tuple[ScoreRecord, ...]) -> None:
    """Write per-score CSV without pair or sample identifiers."""

    _validate_score_records(scores)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("row_index", "partition", "comparison_class", "score"),
        )
        writer.writeheader()
        for score in scores:
            writer.writerow(
                {
                    "row_index": score.row_index,
                    "partition": score.partition.value,
                    "comparison_class": score.comparison_class.value,
                    "score": _format_float(score.score),
                }
            )


def write_threshold_metrics_csv(
    path: Path,
    metrics: tuple[ThresholdMetric, ...],
) -> None:
    """Write exploratory threshold metrics CSV."""

    _validate_threshold_metrics(metrics)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "threshold",
                "far",
                "frr",
                "false_accepts",
                "false_rejects",
                "impostor_total",
                "genuine_total",
            ),
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "threshold": _format_float(metric.threshold),
                    "far": _format_float(metric.far),
                    "frr": _format_float(metric.frr),
                    "false_accepts": metric.false_accepts,
                    "false_rejects": metric.false_rejects,
                    "impostor_total": metric.impostor_total,
                    "genuine_total": metric.genuine_total,
                }
            )


def write_summary_csv(path: Path, summary: FeasibilityReportSummary) -> None:
    """Write key-value summary CSV without sensitive row-level fields."""

    _validate_summary(summary)
    rows = [
        ("label", summary.label),
        ("label_criteria_version", summary.label_criteria_version),
        ("total_samples", str(summary.total_samples)),
        ("generated_pairs", str(summary.generated_pairs)),
        ("evaluated_scores", str(summary.evaluated_scores)),
        ("genuine_count", str(summary.genuine_distribution.count)),
        ("impostor_count", str(summary.impostor_distribution.count)),
        ("overlap_low", _format_optional_float(summary.overlap_low)),
        ("overlap_high", _format_optional_float(summary.overlap_high)),
    ]
    for code, count in sorted(summary.invalid_counts.items()):
        rows.append((f"invalid_count.{code}", str(count)))

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("key", "value"))
        writer.writerows(rows)


def write_html_report(
    path: Path,
    *,
    summary: FeasibilityReportSummary,
    scores: tuple[ScoreRecord, ...],
    threshold_metrics: tuple[ThresholdMetric, ...],
) -> None:
    """Write a static exploratory HTML report."""

    _validate_summary(summary)
    _validate_score_records(scores)
    _validate_threshold_metrics(threshold_metrics)
    genuine_scores = tuple(
        score.score
        for score in scores
        if score.comparison_class is CalibrationComparisonClass.GENUINE
    )
    impostor_scores = tuple(
        score.score
        for score in scores
        if score.comparison_class is CalibrationComparisonClass.IMPOSTOR
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VoiceID Feasibility Probe Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; }}
    table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
    th, td {{ border: 1px solid #d0d7de; padding: 0.4rem 0.55rem; text-align: left; }}
    th {{ background: #f6f8fa; }}
    .note {{ color: #57606a; }}
  </style>
</head>
<body>
  <h1>VoiceID Feasibility Probe Report</h1>
  <p class="note">
    Exploratory raw cosine similarity report. Scores are not probability,
    confidence, production thresholds, or biometric identity verdicts.
  </p>
  <h2>Summary</h2>
  {_summary_table(summary)}
  <h2>Score Distributions</h2>
  {_distribution_table(summary)}
  <h2>Histogram</h2>
  {_histogram_table(genuine_scores, impostor_scores)}
  <h2>Exploratory Threshold Metrics</h2>
  {_threshold_table(threshold_metrics)}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _summary_table(summary: FeasibilityReportSummary) -> str:
    invalid_rows = "".join(
        f"<tr><th>{_html(code)}</th><td>{_html(str(count))}</td></tr>"
        for code, count in sorted(summary.invalid_counts.items())
    )
    if not invalid_rows:
        invalid_rows = "<tr><th>invalid_counts</th><td>0</td></tr>"
    rows = (
        ("label", summary.label),
        ("label_criteria_version", summary.label_criteria_version),
        ("total_samples", str(summary.total_samples)),
        ("generated_pairs", str(summary.generated_pairs)),
        ("evaluated_scores", str(summary.evaluated_scores)),
        ("overlap_low", _format_optional_float(summary.overlap_low)),
        ("overlap_high", _format_optional_float(summary.overlap_high)),
    )
    body = "".join(
        f"<tr><th>{_html(key)}</th><td>{_html(value)}</td></tr>" for key, value in rows
    )
    return f"<table><tbody>{body}{invalid_rows}</tbody></table>"


def _distribution_table(summary: FeasibilityReportSummary) -> str:
    rows = (
        ("GENUINE", summary.genuine_distribution),
        ("IMPOSTOR", summary.impostor_distribution),
    )
    body = "".join(
        "<tr>"
        f"<td>{_html(label)}</td>"
        f"<td>{_html(str(distribution.count))}</td>"
        f"<td>{_html(_format_optional_float(distribution.minimum))}</td>"
        f"<td>{_html(_format_optional_float(distribution.maximum))}</td>"
        f"<td>{_html(_format_optional_float(distribution.mean))}</td>"
        f"<td>{_html(_format_optional_float(distribution.median))}</td>"
        "</tr>"
        for label, distribution in rows
    )
    return (
        "<table><thead><tr><th>class</th><th>count</th><th>min</th>"
        "<th>max</th><th>mean</th><th>median</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _histogram_table(
    genuine_scores: tuple[float, ...],
    impostor_scores: tuple[float, ...],
) -> str:
    genuine_bins = _histogram_counts(genuine_scores)
    impostor_bins = _histogram_counts(impostor_scores)
    rows = []
    for index, (genuine_count, impostor_count) in enumerate(
        zip(genuine_bins, impostor_bins, strict=True)
    ):
        low = -1.0 + index * (2.0 / HISTOGRAM_BINS)
        high = -1.0 + (index + 1) * (2.0 / HISTOGRAM_BINS)
        rows.append(
            "<tr>"
            f"<td>{_html(f'[{_format_float(low)}, {_format_float(high)})')}</td>"
            f"<td>{_html(str(genuine_count))}</td>"
            f"<td>{_html(str(impostor_count))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>score_bin</th><th>genuine</th>"
        "<th>impostor</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _threshold_table(metrics: tuple[ThresholdMetric, ...]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{_html(_format_float(metric.threshold))}</td>"
        f"<td>{_html(_format_float(metric.far))}</td>"
        f"<td>{_html(_format_float(metric.frr))}</td>"
        f"<td>{_html(str(metric.false_accepts))}</td>"
        f"<td>{_html(str(metric.false_rejects))}</td>"
        "</tr>"
        for metric in metrics
    )
    return (
        "<table><thead><tr><th>threshold</th><th>FAR</th><th>FRR</th>"
        "<th>false_accepts</th><th>false_rejects</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _histogram_counts(scores: tuple[float, ...]) -> tuple[int, ...]:
    counts = [0] * HISTOGRAM_BINS
    for score in scores:
        clipped = min(1.0, max(-1.0, score))
        index = int((clipped + 1.0) / 2.0 * HISTOGRAM_BINS)
        if index == HISTOGRAM_BINS:
            index -= 1
        counts[index] += 1
    return tuple(counts)


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else _format_float(value)


def _format_float(value: float) -> str:
    return f"{value:.{PUBLIC_SCORE_DECIMALS}f}"


def _html(value: str) -> str:
    if type(value) is not str:
        raise ValueError("Invalid report value.")
    return html.escape(value, quote=True)


def _validate_score_records(scores: tuple[ScoreRecord, ...]) -> None:
    if type(scores) is not tuple:
        raise ValueError("Invalid report scores.")
    if any(type(score) is not ScoreRecord for score in scores):
        raise ValueError("Invalid report scores.")


def _validate_threshold_metrics(metrics: tuple[ThresholdMetric, ...]) -> None:
    if type(metrics) is not tuple:
        raise ValueError("Invalid report threshold metrics.")
    if any(type(metric) is not ThresholdMetric for metric in metrics):
        raise ValueError("Invalid report threshold metrics.")


def _validate_invalid_counts(invalid_counts: Counter[str]) -> None:
    if type(invalid_counts) is not Counter:
        raise ValueError("Invalid report summary.")
    for code, count in invalid_counts.items():
        if (
            type(code) is not str
            or not code
            or code.strip() != code
            or not set(code).issubset(_SAFE_COUNT_CODE_CHARACTERS)
        ):
            raise ValueError("Invalid report summary.")
        _validate_count_int(count)


def _validate_count_int(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("Invalid report count.")


def _validate_optional_score_float(value: float | None) -> None:
    if value is None:
        return
    _validate_score_float(value)


def _validate_score_float(value: float) -> None:
    if type(value) is not float or not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("Invalid report score.")


def _validate_rate_float(value: float) -> None:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Invalid report rate.")
