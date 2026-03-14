from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build IAA subset files from the QA validation set."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/qa_validation_seed.jsonl"),
        help="Path to the canonical QA JSONL file in reference format.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/qa_validation_metadata.jsonl"),
        help="Path to the QA metadata JSONL file.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/qa_validation_iaa_subset.jsonl"),
        help="Path for the IAA subset QA JSONL in reference format.",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("data/qa_validation_iaa_metadata.jsonl"),
        help="Path for the IAA subset metadata JSONL.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/qa_validation_iaa_subset_blind.csv"),
        help="Path for the blind second-annotator CSV.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_question_index(
    rows: list[dict[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    index: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        question = str(row["question"])
        url = str(row["url"])
        index[(question, url)] = row
    return index


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_blind_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "question",
        "url",
        "second_annotator_answer",
        "second_annotator_evidence",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "url": row["url"],
                    "second_annotator_answer": "",
                    "second_annotator_evidence": "",
                    "notes": "",
                }
            )


def main() -> None:
    args = parse_args()
    question_rows = load_rows(args.questions)
    metadata_rows = load_rows(args.metadata)
    question_index = build_question_index(question_rows)

    iaa_metadata_rows = [
        row for row in metadata_rows if row.get("needs_second_annotation")
    ]
    iaa_question_rows: list[dict[str, object]] = []

    for metadata_row in iaa_metadata_rows:
        question = str(metadata_row["question"])
        url = str(metadata_row["url"])
        question_row = question_index.get((question, url))
        if question_row is None:
            raise ValueError(
                "Missing canonical QA row for metadata entry "
                f"{metadata_row.get('id', '<unknown>')}."
            )
        iaa_question_rows.append(question_row)

    write_jsonl(iaa_question_rows, args.output_jsonl)
    write_jsonl(iaa_metadata_rows, args.output_metadata)
    write_blind_csv(iaa_metadata_rows, args.output_csv)


if __name__ == "__main__":
    main()
