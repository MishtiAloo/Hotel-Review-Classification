"""Tests for the data validation and deduplication rules in src/data/prepare.py."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))

from src.data.labels import ASPECT_COLUMNS
from src.data.prepare import normalize_text, remove_duplicates, validate_rows


def make_row(row_id, review, positive=0, negative=0, neutral=0, **aspect_overrides):
    """Build one dataset-shaped row with all aspects defaulting to zero."""
    row = {"id": row_id, "review": review, "positive": positive, "negative": negative, "neutral": neutral}
    for aspect in ASPECT_COLUMNS:
        row[aspect] = aspect_overrides.get(aspect, 0)
    return row


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  The   room   was\tclean.  ") == "The room was clean."


def test_validate_rows_rejects_non_binary_label():
    row = make_row(1, "Nice hotel.", positive=1)
    row["Clean"] = "sigma"  # not 0 or 1, mirrors the real HRAST.csv row 4634
    dataframe = pd.DataFrame([row])

    valid, rejected = validate_rows(dataframe)

    assert len(valid) == 0
    assert len(rejected) == 1
    assert "Clean is not binary" in rejected[0]["reasons"][0]


def test_validate_rows_rejects_multiple_sentiment_labels():
    dataframe = pd.DataFrame([make_row(1, "Nice hotel.", positive=1, negative=1)])

    valid, rejected = validate_rows(dataframe)

    assert len(valid) == 0
    assert rejected[0]["reasons"] == ["sentiment columns do not contain exactly one 1"]


def test_validate_rows_accepts_a_clean_row():
    dataframe = pd.DataFrame([make_row(1, "Nice hotel.", positive=1, Staff=1)])

    valid, rejected = validate_rows(dataframe)

    assert len(rejected) == 0
    assert len(valid) == 1
    assert valid[0]["sentiment"] == "positive"


def test_remove_duplicates_keeps_one_copy_of_matching_rows():
    dataframe = pd.DataFrame(
        [
            make_row(1, "Great stay.", positive=1),
            make_row(2, "Great stay.", positive=1),
        ]
    )
    valid, rejected = validate_rows(dataframe)

    cleaned, extras, conflicting_groups = remove_duplicates(valid, rejected)

    assert len(cleaned) == 1
    assert extras == 1
    assert conflicting_groups == 0


def test_remove_duplicates_quarantines_conflicting_rows():
    dataframe = pd.DataFrame(
        [
            make_row(1, "Great stay.", positive=1),
            make_row(2, "Great stay.", negative=1),
        ]
    )
    valid, rejected = validate_rows(dataframe)

    # remove_duplicates appends newly quarantined rows into this same list.
    cleaned, extras, conflicting_groups = remove_duplicates(valid, rejected)

    assert len(cleaned) == 0
    assert conflicting_groups == 1
    rejected_ids = {record["id"] for record in rejected}
    assert rejected_ids == {1, 2}
