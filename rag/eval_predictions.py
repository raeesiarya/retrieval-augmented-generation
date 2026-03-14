from __future__ import annotations

import argparse
import json
import string
import sys
from collections import Counter
from pathlib import Path


PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    return " ".join(text.lower().translate(PUNCT_TABLE).split())


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(
    metric_fn, prediction: str, ground_truths: list[str]
) -> float:
    return max(metric_fn(prediction, ground_truth) for ground_truth in ground_truths)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_squad_dataset(obj: object) -> bool:
    return (
        isinstance(obj, dict)
        and "data" in obj
        and "version" in obj
        and isinstance(obj["data"], list)
    )


def evaluate_squad(dataset: dict[str, object], predictions: dict[str, str]) -> dict[str, float]:
    f1 = 0.0
    exact_match = 0.0
    total = 0

    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    print(
                        f"Unanswered question {qa['id']} will receive score 0.",
                        file=sys.stderr,
                    )
                    continue

                ground_truths = [answer["text"] for answer in qa["answers"]]
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths
                )
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    total = max(total, 1)
    return {
        "count": total,
        "exact_match": exact_match / total,
        "f1": f1 / total,
        "exact_match_percent": round(100.0 * exact_match / total, 2),
        "f1_percent": round(100.0 * f1 / total, 2),
    }


def load_reference_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            url = str(row.get("url", "")).strip()
            if not question or not answer:
                raise ValueError(
                    f"Missing question/answer in reference JSONL at line {line_number}."
                )
            rows.append({"question": question, "answer": answer, "url": url})
    return rows


def load_prediction_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def split_answers(answer_field: str) -> list[str]:
    answers = [part.strip() for part in answer_field.split("|")]
    return [answer for answer in answers if answer]


def evaluate_reference_jsonl(
    references: list[dict[str, str]],
    predictions: list[str],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    if len(references) != len(predictions):
        raise ValueError(
            "Prediction/reference length mismatch: "
            f"{len(predictions)} predictions vs {len(references)} references."
        )

    total_em = 0.0
    total_f1 = 0.0
    details: list[dict[str, object]] = []

    for reference_row, prediction in zip(references, predictions):
        gold_answers = split_answers(reference_row["answer"])
        em = metric_max_over_ground_truths(exact_match_score, prediction, gold_answers)
        f1 = metric_max_over_ground_truths(f1_score, prediction, gold_answers)
        total_em += em
        total_f1 += f1
        details.append(
            {
                "question": reference_row["question"],
                "url": reference_row["url"],
                "prediction": prediction,
                "gold_answers": reference_row["answer"],
                "exact_match": em,
                "f1": f1,
            }
        )

    total = max(1, len(references))
    summary = {
        "count": len(references),
        "exact_match": total_em / total,
        "f1": total_f1 / total,
        "exact_match_percent": round(100.0 * total_em / total, 2),
        "f1_percent": round(100.0 * total_f1 / total, 2),
    }
    return summary, details


def write_details(details: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate either SQuAD-style JSON + prediction JSON, or "
            "reference.jsonl + line-by-line predictions."
        )
    )
    parser.add_argument("dataset_file", nargs="?", default=None)
    parser.add_argument("prediction_file", nargs="?", default=None)
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--details-out", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    references = args.references or (
        Path(args.dataset_file) if args.dataset_file is not None else None
    )
    predictions = args.predictions or (
        Path(args.prediction_file) if args.prediction_file is not None else None
    )

    if references is None or predictions is None:
        raise ValueError(
            "Provide either positional dataset/prediction files or "
            "--references/--predictions."
        )

    return references, predictions


def main() -> None:
    args = parse_args()
    references_path, predictions_path = resolve_paths(args)

    dataset_obj = load_json(references_path) if references_path.suffix == ".json" else None
    predictions_obj = (
        load_json(predictions_path) if predictions_path.suffix == ".json" else None
    )

    if dataset_obj is not None and is_squad_dataset(dataset_obj):
        if not isinstance(predictions_obj, dict):
            raise ValueError("SQuAD evaluation expects predictions to be a JSON object.")
        results = evaluate_squad(dataset_obj, predictions_obj)
        print(json.dumps(results, indent=2))
        return

    references = load_reference_jsonl(references_path)
    predictions = load_prediction_lines(predictions_path)
    results, details = evaluate_reference_jsonl(references, predictions)
    if args.details_out is not None:
        write_details(details, args.details_out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
