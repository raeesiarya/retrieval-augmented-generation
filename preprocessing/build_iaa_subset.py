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
        "--input",
        type=Path,
        default=Path("data/qa_validation_seed.jsonl"),
        help="Path to the full QA validation JSONL file.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/qa_validation_iaa_subset.jsonl"),
        help="Path for the IAA subset answer-key JSONL.",
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
        "source_url",
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
                    "source_url": row["source_url"],
                    "second_annotator_answer": "",
                    "second_annotator_evidence": "",
                    "notes": "",
                }
            )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    iaa_rows = [row for row in rows if row.get("needs_second_annotation")]
    write_jsonl(iaa_rows, args.output_jsonl)
    write_blind_csv(iaa_rows, args.output_csv)


if __name__ == "__main__":
    main()
