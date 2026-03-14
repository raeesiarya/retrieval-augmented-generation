from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export question text from a QA JSONL file to a plain txt file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/qa_validation_seed.jsonl"),
        help="Input QA JSONL file with question/answer/url rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("questions_validation.txt"),
        help="Output txt file with one question per line.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    questions: list[str] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            if not question:
                raise ValueError(f"Missing question on line {line_number} in {args.input}")
            questions.append(question)

    with args.output.open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(question + "\n")

    print(f"Exported {len(questions)} questions to {args.output}")


if __name__ == "__main__":
    main()
