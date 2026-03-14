from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\+?\d[\d()\-\s]{7,}\d")
DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\b|\b(?:19|20)\d{2}\b"
)
NUMBER_RE = re.compile(r"\b\d[\d,./:-]*\b")

KEYWORD_GROUPS = {
    "contact": ("email", "phone", "office", "contact", "hours", "hall"),
    "academics": ("admission", "application", "degree", "course", "program", "units"),
    "people": ("director", "chair", "professor", "student", "staff"),
    "rankings": ("rank", "ranking", "enrollment", "fellow", "award", "members"),
    "events": ("event", "colloquium", "celebrating", "program", "lecture"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a QA annotation worklist from the EECS crawl corpus."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/crawl_eecs_raw.jsonl"),
        help="JSONL file with {url, text} crawl documents.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/qa_annotation_worklist.csv"),
        help="CSV path for the annotation worklist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=45,
        help="Maximum number of candidate pages to include.",
    )
    return parser.parse_args()


def load_documents(path: Path) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            documents.append(json.loads(line))
    return documents


def infer_page_type(url: str) -> str:
    if "/contact" in url or "/visiting" in url:
        return "contact"
    if "/academics/" in url:
        return "academics"
    if "/people/" in url:
        return "people"
    if "/news/" in url or re.search(r"/20\d{2}/", url):
        return "news"
    if "/about/" in url:
        return "about"
    if "/connect/" in url:
        return "connect"
    if "/research/" in url:
        return "research"
    return "general"


def question_ideas(url: str, text: str) -> str:
    page_type = infer_page_type(url)
    ideas: list[str] = []
    lowered = text.lower()

    if page_type == "contact":
        ideas.append("contact email, office location, phone, office hours")
    if page_type == "academics":
        ideas.append("requirement, degree type, deadline, unit load")
    if page_type == "people":
        ideas.append("person-role, title, office, contact detail")
    if page_type == "news":
        ideas.append("award, honoree, date, count, affiliation")
    if page_type == "about":
        ideas.append("ranking, count, year, historical milestone")
    if page_type == "connect":
        ideas.append("support contact, donation program, named initiative")
    if page_type == "research":
        ideas.append("event name, archive year, research area")

    if (page_type in {"contact", "people", "connect"}) and (
        EMAIL_RE.search(text) or PHONE_RE.search(text)
    ):
        ideas.append("contact email, office location, phone, office hours")
    if any(
        word in lowered for word in ("admission", "degree", "course", "units", "deadline")
    ):
        ideas.append("requirement, degree type, deadline, unit load")
    if any(word in lowered for word in ("chair", "director", "professor", "manager", "fellow")):
        ideas.append("person-role, title, award, affiliation")
    if any(word in lowered for word in ("rank", "ranking", "number of", "enrollment")):
        ideas.append("ranking, count, total enrollment, year")
    if any(word in lowered for word in ("event", "lecture", "program", "ceremony", "announced")):
        ideas.append("event date, venue, month, named program")

    if not ideas:
        fallback = {
            "academics": "program comparison, requirement, degree structure",
            "people": "name, role, office, contact detail",
            "news": "award, honoree, date, count",
            "contact": "location, office hours, email, phone",
            "about": "history fact, count, ranking, milestone",
            "connect": "donation program, support contact, initiative",
            "research": "lab, area, event, archive detail",
            "general": "count, named entity, year, location",
        }
        ideas.append(fallback[page_type])

    deduped = list(dict.fromkeys(ideas))
    return "; ".join(deduped[:3])


def preview(text: str, limit: int = 260) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def score_document(document: dict[str, str]) -> tuple[float, Counter]:
    text = document["text"]
    lowered = text.lower()
    word_count = max(1, len(text.split()))

    counts = Counter(
        emails=len(EMAIL_RE.findall(text)),
        phones=len(PHONE_RE.findall(text)),
        dates=len(DATE_RE.findall(text)),
        numbers=len(NUMBER_RE.findall(text)),
    )
    for label, keywords in KEYWORD_GROUPS.items():
        counts[label] = sum(lowered.count(keyword) for keyword in keywords)

    score = (
        min(counts["emails"], 8) * 4
        + min(counts["phones"], 6) * 3
        + min(counts["dates"], 8) * 2
        + min(counts["numbers"], 20) * 0.5
        + min(counts["contact"], 10) * 0.8
        + min(counts["academics"], 10) * 0.8
        + min(counts["people"], 10) * 0.5
        + min(counts["rankings"], 10) * 0.6
        + min(counts["events"], 10) * 0.6
    )
    score = score / (1 + word_count / 900)
    return score, counts


def select_documents(documents: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    scored = []
    for document in documents:
        score, counts = score_document(document)
        scored.append((score, counts, document))

    scored.sort(key=lambda item: item[0], reverse=True)

    selected: list[dict[str, str]] = []
    by_type: Counter[str] = Counter()
    for score, counts, document in scored:
        page_type = infer_page_type(document["url"])
        if by_type[page_type] >= 8:
            continue
        selected.append(
            {
                "url": document["url"],
                "text": document["text"],
                "page_type": page_type,
                "score": f"{score:.1f}",
                "counts": counts,
            }
        )
        by_type[page_type] += 1
        if len(selected) >= limit:
            break

    return selected


def write_worklist(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "url",
        "page_type",
        "score",
        "suggested_question_types",
        "text_preview",
        "status",
        "question",
        "answer",
        "evidence_text",
        "needs_second_annotation",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "item_id": f"page_{index:03d}",
                    "url": row["url"],
                    "page_type": row["page_type"],
                    "score": row["score"],
                    "suggested_question_types": question_ideas(row["url"], row["text"]),
                    "text_preview": preview(row["text"]),
                    "status": "todo",
                    "question": "",
                    "answer": "",
                    "evidence_text": "",
                    "needs_second_annotation": "no",
                }
            )


def main() -> None:
    args = parse_args()
    documents = load_documents(args.input)
    rows = select_documents(documents, limit=args.limit)
    write_worklist(rows, args.output)


if __name__ == "__main__":
    main()
