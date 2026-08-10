"""Tests for the public Phase 5A similarity contracts."""

from __future__ import annotations

import math

import pytest

from voiceid.similarity.contracts import (
    COSINE_SIMILARITY_METRIC,
    SIMILARITY_COMPARISON_VERSION,
    SimilarityErrorCode,
    SimilarityIssue,
    SimilarityStatus,
    SpeakerSimilarityMetadata,
    SpeakerSimilarityResult,
    build_invalid_similarity_result,
)


def test_valid_similarity_result_contract_and_serialization() -> None:
    result = SpeakerSimilarityResult(
        status=SimilarityStatus.VALID,
        similarity=0.8123456789,
        metadata=_metadata(),
    )

    assert result.is_valid
    assert type(result.similarity) is float
    assert result.to_dict() == {
        "status": "VALID",
        "similarity": 0.8123456789,
        "metadata": _metadata().to_dict(),
        "errors": [],
    }
    assert "0.812346" in repr(result)
    assert str(result) == repr(result)


def test_invalid_similarity_result_has_one_safe_error() -> None:
    result = build_invalid_similarity_result(
        SimilarityErrorCode.INCOMPATIBLE_EMBEDDINGS
    )

    assert result.status == SimilarityStatus.INVALID
    assert result.similarity is None
    assert result.metadata is None
    assert len(result.errors) == 1
    assert result.errors[0].code == "INCOMPATIBLE_EMBEDDINGS"
    assert result.to_dict()["metadata"] == {}


@pytest.mark.parametrize(
    ("case_name", "match"),
    [
        ("status_string", "SimilarityStatus"),
        ("valid_without_similarity", "Python float"),
        ("valid_with_integer_similarity", "Python float"),
        ("valid_with_nan", "finite"),
        ("valid_with_positive_infinity", "finite"),
        ("valid_below_range", r"\[-1.0, 1.0\]"),
        ("valid_above_range", r"\[-1.0, 1.0\]"),
        ("valid_without_metadata", "requires metadata"),
        ("valid_with_wrong_metadata_type", "requires metadata"),
        ("valid_with_errors", "cannot contain errors"),
        ("valid_with_empty_error_list", "cannot contain errors"),
        ("invalid_with_similarity", "cannot contain a score"),
        ("invalid_with_metadata", "cannot contain metadata"),
        ("invalid_without_error", "exactly one error"),
        ("invalid_with_two_errors", "exactly one error"),
        ("invalid_with_wrong_error_type", "exactly one error"),
    ],
)
def test_similarity_result_rejects_contradictory_states(
    case_name: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        SpeakerSimilarityResult(**_invalid_result_kwargs(case_name))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"metric": "dot_product"}, "cosine_similarity"),
        ({"comparison_version": "2"}, "version"),
        ({"comparison_version": None}, "version"),
        ({"comparison_version": ""}, "version"),
        ({"comparison_version": "   "}, "version"),
        ({"comparison_version": b"1"}, "version"),
        ({"comparison_version": True}, "version"),
        ({"comparison_version": 1}, "version"),
        ({"comparison_version": object()}, "version"),
        ({"embedding_dimension": 191}, "192"),
        ({"embedding_dimension": 192.0}, "192"),
        ({"normalized": 0}, "boolean"),
    ],
)
def test_similarity_metadata_rejects_invalid_contract_values(
    overrides: dict[str, object],
    match: str,
) -> None:
    values = _metadata_values()
    values.update(overrides)
    with pytest.raises(ValueError, match=match):
        SpeakerSimilarityMetadata(**values)  # type: ignore[arg-type]


def test_similarity_issue_rejects_unknown_code_and_arbitrary_message() -> None:
    with pytest.raises(ValueError, match="code"):
        SimilarityIssue(code="UNKNOWN", message="safe")
    with pytest.raises(ValueError, match="stable and safe"):
        SimilarityIssue(
            code=SimilarityErrorCode.COMPARISON_ERROR.value,
            message="raw internal exception",
        )


def test_similarity_result_accepts_range_boundaries() -> None:
    lower = SpeakerSimilarityResult(
        status=SimilarityStatus.VALID,
        similarity=-1.0,
        metadata=_metadata(),
    )
    upper = SpeakerSimilarityResult(
        status=SimilarityStatus.VALID,
        similarity=1.0,
        metadata=_metadata(),
    )

    assert math.isfinite(lower.similarity) and lower.similarity == -1.0
    assert math.isfinite(upper.similarity) and upper.similarity == 1.0


def _metadata() -> SpeakerSimilarityMetadata:
    return SpeakerSimilarityMetadata(**_metadata_values())


def _metadata_values() -> dict[str, object]:
    return {
        "metric": COSINE_SIMILARITY_METRIC,
        "comparison_version": SIMILARITY_COMPARISON_VERSION,
        "embedding_dimension": 192,
        "normalized": False,
    }


def _issue() -> SimilarityIssue:
    return build_invalid_similarity_result(SimilarityErrorCode.COMPARISON_ERROR).errors[
        0
    ]


def _invalid_result_kwargs(case_name: str) -> dict[str, object]:
    cases = {
        "status_string": {
            "status": "VALID",
            "similarity": 0.5,
            "metadata": _metadata(),
            "errors": (),
        },
        "valid_without_similarity": {
            "status": SimilarityStatus.VALID,
            "similarity": None,
            "metadata": _metadata(),
            "errors": (),
        },
        "valid_with_integer_similarity": {
            "status": SimilarityStatus.VALID,
            "similarity": 1,
            "metadata": _metadata(),
            "errors": (),
        },
        "valid_with_nan": {
            "status": SimilarityStatus.VALID,
            "similarity": float("nan"),
            "metadata": _metadata(),
            "errors": (),
        },
        "valid_with_positive_infinity": {
            "status": SimilarityStatus.VALID,
            "similarity": float("inf"),
            "metadata": _metadata(),
            "errors": (),
        },
        "valid_below_range": {
            "status": SimilarityStatus.VALID,
            "similarity": -1.0001,
            "metadata": _metadata(),
            "errors": (),
        },
        "valid_above_range": {
            "status": SimilarityStatus.VALID,
            "similarity": 1.0001,
            "metadata": _metadata(),
            "errors": (),
        },
        "valid_without_metadata": {
            "status": SimilarityStatus.VALID,
            "similarity": 0.5,
            "metadata": None,
            "errors": (),
        },
        "valid_with_wrong_metadata_type": {
            "status": SimilarityStatus.VALID,
            "similarity": 0.5,
            "metadata": object(),
            "errors": (),
        },
        "valid_with_errors": {
            "status": SimilarityStatus.VALID,
            "similarity": 0.5,
            "metadata": _metadata(),
            "errors": (_issue(),),
        },
        "valid_with_empty_error_list": {
            "status": SimilarityStatus.VALID,
            "similarity": 0.5,
            "metadata": _metadata(),
            "errors": [],
        },
        "invalid_with_similarity": {
            "status": SimilarityStatus.INVALID,
            "similarity": 0.5,
            "metadata": None,
            "errors": (_issue(),),
        },
        "invalid_with_metadata": {
            "status": SimilarityStatus.INVALID,
            "similarity": None,
            "metadata": _metadata(),
            "errors": (_issue(),),
        },
        "invalid_without_error": {
            "status": SimilarityStatus.INVALID,
            "similarity": None,
            "metadata": None,
            "errors": (),
        },
        "invalid_with_two_errors": {
            "status": SimilarityStatus.INVALID,
            "similarity": None,
            "metadata": None,
            "errors": (_issue(), _issue()),
        },
        "invalid_with_wrong_error_type": {
            "status": SimilarityStatus.INVALID,
            "similarity": None,
            "metadata": None,
            "errors": ("unsafe",),
        },
    }
    return cases[case_name]
