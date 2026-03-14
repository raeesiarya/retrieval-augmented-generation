from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict


WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(
    r"(?<!\.edu)(?<!\.org)(?<!\.gov)(?<!\.com)(?<!\.net)(?<=[.!?])\s+(?=[A-Z0-9])"
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\+?\d[\d()\-\s]{7,}\d")
URL_SLUG_RE = re.compile(r"[-_]+")

NAVIGATION_TAIL_MARKERS = [
    "People Alumni ",
    "Academics Courses ",
    "Connect Support ",
    "About Diversity ",
    "Research Department Colloquium Series ",
    "Our Students Student Organizations ",
    "Our Staff Staff Awards ",
]

INLINE_NOISE_PHRASES = [
    "In This Page",
    "Directory",
    "Staff contact quick list",
    "Department email lists (login required)",
    "Check System Status",
    "Cory and Soda room reservations",
    "Questions About Becoming a Student",
    "Questions About the Department",
    "Questions by Department Members",
]

NOISE_PREFIXES = (
    "categories ",
    "related links ",
    "book an appointment",
    "view photos",
    "view lectures",
    "watch the program",
    "see the winners",
    "submit an event",
    "map it",
    "map it -",
    "map it -",
    "read more",
    "read about",
    "learn more about",
    "more information about",
)

NOISE_EXACT = {
    "in this page",
    "directory",
    "quick links",
    "home",
    "menu",
    "search",
    "skip to main content",
    "staff contact quick list",
    "check system status",
}


class RawDocument(TypedDict):
    url: str
    text: str


class CleanDocument(TypedDict):
    url: str
    title: str
    text: str
    word_count: int


class ChunkDocument(TypedDict):
    chunk_id: str
    url: str
    title: str
    text: str
    chunk_index: int
    word_count: int


def normalize_space(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\xa0": " ",
        "\x00": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return WHITESPACE_RE.sub(" ", text).strip()


def load_raw_documents(path: Path) -> list[RawDocument]:
    rows: list[RawDocument] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            url = normalize_space(str(data.get("url", "")))
            text = str(data.get("text", ""))
            if not url or not text:
                raise ValueError(f"Invalid row at line {line_number}: missing url/text.")
            rows.append({"url": url, "text": text})
    return rows


def strip_navigation_tails(text: str) -> str:
    cleaned = text
    min_offset = 220
    for marker in NAVIGATION_TAIL_MARKERS:
        index = cleaned.find(marker)
        if index >= min_offset:
            cleaned = cleaned[:index].rstrip()
    return cleaned


def strip_inline_noise_phrases(text: str) -> str:
    cleaned = text
    for phrase in INLINE_NOISE_PHRASES:
        cleaned = cleaned.replace(phrase, " ")
    return normalize_space(cleaned)


def split_sentences(text: str) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]


def looks_like_noise(sentence: str) -> bool:
    normalized = normalize_space(sentence)
    lowered = normalized.casefold()

    if not normalized:
        return True
    if lowered in NOISE_EXACT:
        return True
    if lowered.startswith(NOISE_PREFIXES):
        return True
    if len(normalized.split()) <= 2 and not (EMAIL_RE.search(normalized) or PHONE_RE.search(normalized)):
        return True
    return False


def remove_duplicate_sentences(sentences: list[str]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for sentence in sentences:
        key = normalize_space(sentence).casefold()
        if key in seen:
            continue
        seen.add(key)
        kept.append(normalize_space(sentence))
    return kept


def infer_title(url: str, text: str) -> str:
    sentences = split_sentences(text)
    if sentences:
        first = sentences[0].strip(" -")
        if (
            1 <= len(first.split()) <= 12
            and not EMAIL_RE.search(first)
            and not PHONE_RE.search(first)
        ):
            return first

    path = url.rstrip("/").split("/")[-1]
    if not path or path == "eecs.berkeley.edu":
        return "EECS"
    return URL_SLUG_RE.sub(" ", path).strip().title()


def clean_document(document: RawDocument) -> CleanDocument | None:
    text = normalize_space(document["text"])
    text = strip_navigation_tails(text)
    text = strip_inline_noise_phrases(text)
    sentences = split_sentences(text)
    sentences = [sentence for sentence in sentences if not looks_like_noise(sentence)]
    sentences = remove_duplicate_sentences(sentences)

    cleaned_text = normalize_space(" ".join(sentences))
    if len(cleaned_text) < 80:
        return None

    return {
        "url": document["url"],
        "title": infer_title(document["url"], cleaned_text),
        "text": cleaned_text,
        "word_count": len(cleaned_text.split()),
    }


def chunk_document(
    document: CleanDocument,
    min_words: int = 45,
    max_words: int = 140,
    overlap_sentences: int = 1,
) -> list[ChunkDocument]:
    sentences = split_sentences(document["text"])
    if not sentences:
        return []

    chunks: list[ChunkDocument] = []
    start = 0

    while start < len(sentences):
        end = start
        current_words = 0

        while end < len(sentences):
            sentence_words = len(sentences[end].split())
            if current_words >= min_words and current_words + sentence_words > max_words:
                break
            current_words += sentence_words
            end += 1

        if end == start:
            end += 1

        chunk_text = normalize_space(" ".join(sentences[start:end]))
        if chunk_text:
            words = chunk_text.split()
            word_windows: list[str]
            if len(words) > max_words + 30:
                word_windows = []
                word_start = 0
                overlap_words = 25
                while word_start < len(words):
                    word_end = min(len(words), word_start + max_words)
                    word_windows.append(" ".join(words[word_start:word_end]))
                    if word_end >= len(words):
                        break
                    word_start = max(word_start + 1, word_end - overlap_words)
            else:
                word_windows = [chunk_text]

            for window_text in word_windows:
                chunk_index = len(chunks)
                chunks.append(
                    {
                        "chunk_id": f"{document['url']}#chunk-{chunk_index}",
                        "url": document["url"],
                        "title": document["title"],
                        "text": window_text,
                        "chunk_index": chunk_index,
                        "word_count": len(window_text.split()),
                    }
                )

        if end >= len(sentences):
            break

        start = max(start + 1, end - overlap_sentences)

    return chunks


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the EECS crawl into page-level and chunk-level retrieval corpora."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/crawl_eecs_raw.jsonl"),
        help="Path to the raw crawl JSONL file.",
    )
    parser.add_argument(
        "--output-docs",
        type=Path,
        default=Path("data/eecs_corpus_clean.jsonl"),
        help="Path for the cleaned page-level corpus.",
    )
    parser.add_argument(
        "--output-chunks",
        type=Path,
        default=Path("data/eecs_corpus_chunks.jsonl"),
        help="Path for the cleaned chunk-level corpus.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_documents = load_raw_documents(args.input)

    clean_documents: list[CleanDocument] = []
    chunk_documents: list[ChunkDocument] = []

    for raw_document in raw_documents:
        cleaned = clean_document(raw_document)
        if not cleaned:
            continue
        clean_documents.append(cleaned)
        chunk_documents.extend(chunk_document(cleaned))

    write_jsonl(clean_documents, args.output_docs)
    write_jsonl(chunk_documents, args.output_chunks)

    stats = Counter()
    stats["raw_documents"] = len(raw_documents)
    stats["clean_documents"] = len(clean_documents)
    stats["chunks"] = len(chunk_documents)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
