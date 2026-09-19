"""Validate, clean, deduplicate, and split the HRAST dataset."""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from labels import ASPECT_COLUMNS, SENTIMENT_COLUMNS, get_sentiment_name


def make_logger(log_path):
    """Create a logger that writes to both the terminal and a file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("prepare")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def normalize_text(text):
    """Strip the text and replace repeated whitespace with one space."""
    return re.sub(r"\s+", " ", str(text)).strip()


def is_binary(value):
    """Return True for numeric or text forms of zero and one."""
    return value in (0, 1, "0", "1")


def file_sha256(path):
    """Return the SHA-256 hash of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            block = file_handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_rows(dataframe):
    """Return valid records and a list of rejected records with reasons."""
    valid_records = []
    rejected_records = []
    label_columns = SENTIMENT_COLUMNS + ASPECT_COLUMNS

    for _, row in dataframe.iterrows():
        reasons = []
        review = normalize_text(row["review"])

        if not review:
            reasons.append("empty review")

        for column in label_columns:
            value = row[column]
            if not is_binary(value):
                reasons.append(f"{column} is not binary: {value!r}")

        # Only add the values after their binary form has been checked.
        sentiment_is_binary = True
        for column in SENTIMENT_COLUMNS:
            if not is_binary(row[column]):
                sentiment_is_binary = False

        if sentiment_is_binary:
            sentiment_total = 0
            for column in SENTIMENT_COLUMNS:
                sentiment_total += int(row[column])
            if sentiment_total != 1:
                reasons.append("sentiment columns do not contain exactly one 1")

        if reasons:
            rejected_records.append(
                {
                    "id": int(row["id"]),
                    "review": review,
                    "reasons": reasons,
                }
            )
            continue

        record = row.to_dict()
        record["review"] = review
        for column in label_columns:
            record[column] = int(record[column])
        record["id"] = int(record["id"])
        record["sentiment"] = get_sentiment_name(record)
        valid_records.append(record)

    return valid_records, rejected_records


def remove_duplicates(records, rejected_records):
    """Keep matching duplicates once and reject all conflicting duplicates."""
    groups = {}
    model_columns = SENTIMENT_COLUMNS + ASPECT_COLUMNS

    for record in records:
        review = record["review"]
        if review not in groups:
            groups[review] = []
        groups[review].append(record)

    cleaned_records = []
    duplicate_extras = 0
    conflicting_groups = 0

    for review, group in groups.items():
        first_labels = []
        for column in model_columns:
            first_labels.append(group[0][column])

        labels_match = True
        for record in group[1:]:
            current_labels = []
            for column in model_columns:
                current_labels.append(record[column])
            if current_labels != first_labels:
                labels_match = False

        if not labels_match:
            conflicting_groups += 1
            for record in group:
                rejected_records.append(
                    {
                        "id": record["id"],
                        "review": review,
                        "reasons": ["duplicate review has conflicting labels"],
                    }
                )
            continue

        cleaned_records.append(group[0])
        duplicate_extras += len(group) - 1

    return cleaned_records, duplicate_extras, conflicting_groups


def make_splits(dataframe, seed):
    """Create shared 70/15/15 splits using all sentiment and aspect labels."""
    stratify_columns = ASPECT_COLUMNS + SENTIMENT_COLUMNS
    labels = dataframe[stratify_columns].to_numpy()

    first_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed,
    )
    train_indices, temporary_indices = next(first_split.split(dataframe, labels))

    temporary = dataframe.iloc[temporary_indices].reset_index(drop=True)
    temporary_labels = temporary[stratify_columns].to_numpy()
    second_split = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed,
    )
    validation_indices, test_indices = next(
        second_split.split(temporary, temporary_labels)
    )

    train = dataframe.iloc[train_indices].reset_index(drop=True)
    validation = temporary.iloc[validation_indices].reset_index(drop=True)
    test = temporary.iloc[test_indices].reset_index(drop=True)
    return train, validation, test


def count_labels(dataframe):
    """Return simple split statistics for the audit report."""
    sentiment_counts = dataframe["sentiment"].value_counts().to_dict()
    aspect_counts = {}
    for column in ASPECT_COLUMNS:
        aspect_counts[column] = int(dataframe[column].sum())

    return {
        "rows": len(dataframe),
        "sentiment_counts": sentiment_counts,
        "aspect_counts": aspect_counts,
    }


def assert_no_overlap(train, validation, test):
    """Stop if the same normalized review appears in two splits."""
    train_reviews = set(train["review"])
    validation_reviews = set(validation["review"])
    test_reviews = set(test["review"])

    assert train_reviews.isdisjoint(validation_reviews)
    assert train_reviews.isdisjoint(test_reviews)
    assert validation_reviews.isdisjoint(test_reviews)


def parse_args():
    """Read command-line paths and the reproducibility seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/HRAST.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--log-file", type=Path, default=Path("logs/prepare.log"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-rows", type=int, default=23095)
    return parser.parse_args()


def main():
    """Run the complete deterministic data-preparation pipeline."""
    args = parse_args()
    logger = make_logger(args.log_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s", args.input)
    dataframe = pd.read_csv(args.input, encoding="utf-8-sig")
    raw_rows = len(dataframe)
    logger.info("Raw rows: %d", raw_rows)

    # The source contains one entirely empty, unnamed column.
    unnamed_columns = []
    for column in dataframe.columns:
        if str(column).startswith("Unnamed"):
            unnamed_columns.append(column)
    dataframe = dataframe.drop(columns=unnamed_columns)
    logger.info("Dropped unnamed columns: %s", unnamed_columns)

    required_columns = ["id", "review", "Aspect"]
    required_columns += SENTIMENT_COLUMNS + ASPECT_COLUMNS
    missing_columns = []
    for column in required_columns:
        if column not in dataframe.columns:
            missing_columns.append(column)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    valid_records, rejected_records = validate_rows(dataframe)
    malformed_count = len(rejected_records)
    logger.info("Malformed rows quarantined: %d", malformed_count)

    clean_records, duplicate_extras, conflicting_groups = remove_duplicates(
        valid_records,
        rejected_records,
    )
    logger.info("Matching duplicate extras removed: %d", duplicate_extras)
    logger.info("Conflicting duplicate groups quarantined: %d", conflicting_groups)

    clean = pd.DataFrame(clean_records)
    clean = clean.sort_values("id").reset_index(drop=True)

    if len(clean) != args.expected_rows:
        message = f"Expected {args.expected_rows} clean rows, found {len(clean)}"
        raise ValueError(message)

    train, validation, test = make_splits(clean, args.seed)
    assert_no_overlap(train, validation, test)

    # Do not train on the unreliable source-level Aspect flag.
    output_columns = ["id", "review", "sentiment"]
    output_columns += SENTIMENT_COLUMNS + ASPECT_COLUMNS
    clean = clean[output_columns]
    train = train[output_columns]
    validation = validation[output_columns]
    test = test[output_columns]

    clean.to_csv(args.output_dir / "hrast_clean.csv", index=False)
    train.to_csv(args.output_dir / "train.csv", index=False)
    validation.to_csv(args.output_dir / "validation.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)

    # Record how badly the source Aspect flag differs from the actual topic columns.
    numeric_aspects = dataframe[ASPECT_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    aspect_presence = numeric_aspects.eq(1).any(axis=1).astype(int)
    aspect_flag = pd.to_numeric(dataframe["Aspect"], errors="coerce")
    aspect_flag_mismatches = int(aspect_flag.ne(aspect_presence).sum())

    audit = {
        "source": str(args.input),
        "source_sha256": file_sha256(args.input),
        "seed": args.seed,
        "raw_rows": raw_rows,
        "clean_rows": len(clean),
        "unnamed_columns_removed": [str(value) for value in unnamed_columns],
        "malformed_rows_quarantined": malformed_count,
        "matching_duplicate_extras_removed": duplicate_extras,
        "conflicting_duplicate_groups_quarantined": conflicting_groups,
        "aspect_flag_mismatches": aspect_flag_mismatches,
        "rejected_records": rejected_records,
        "splits": {
            "train": count_labels(train),
            "validation": count_labels(validation),
            "test": count_labels(test),
        },
    }

    audit_path = args.output_dir / "audit.json"
    with audit_path.open("w", encoding="utf-8") as file_handle:
        json.dump(audit, file_handle, indent=2, ensure_ascii=False)

    logger.info("Clean rows: %d", len(clean))
    logger.info(
        "Split rows: train=%d, validation=%d, test=%d",
        len(train),
        len(validation),
        len(test),
    )
    logger.info("Aspect flag mismatches: %d", aspect_flag_mismatches)
    logger.info("Wrote audit report to %s", audit_path)
    logger.info("Data preparation completed successfully")


if __name__ == "__main__":
    main()
