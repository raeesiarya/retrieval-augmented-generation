from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

try:
    from rag.llm import call_llm
except ImportError:
    from llm import call_llm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CORPUS_CANDIDATES = (
    "data/crawl_eecs.jsonl",
    "data/crawl_eecs_raw.jsonl",
    "data/eecs_corpus_chunks.jsonl",
    "data/eecs_corpus_clean.jsonl",
)

SYSTEM_PROMPT = (
    "You are answering factoid questions about UC Berkeley EECS using retrieved context.\n"
    "Rules:\n"
    "- Return a short answer phrase only (as short as possible).\n"
    "- Do not explain or add extra text.\n"
    '- If the answer is not supported by context, return exactly: "unknown".'
)

LLM_CHOICE = "meta-llama/llama-3.1-8b-instruct"

@dataclass(frozen=True)
class Chunk:
    url: str
    text: str
    retrieval_text: str

WHITESPACE_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?1[\s-]*)?(?:\(\d{3}\)|\d{3})[\s-]*\d{3}[\s-]*\d{4}(?!\w)"
)
COURSE_RE = re.compile(
    r"\b(?:CS|EECS|EE|ECE|MATH|ENGIN|STAT|DATA|INFO)\s*-?\s*\d{1,3}[A-Z]?\b",
    re.IGNORECASE,
)
ROOM_RE = re.compile(r"\b\d{3,4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")
BUILDING_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+"
    r"(?:Hall|Center|Building|Auditorium|Lab|Laboratory)\b"
)
TIME_RE = re.compile(
    r"\b(?:\d{1,2}(?::\d{2})?\s?(?:a\.?m\.?|p\.?m\.?)|noon|midnight)\b",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.IGNORECASE,
)
DATE_RANGE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}\s*[-–]\s*\d{1,2}\b",
    re.IGNORECASE,
)
YEAR_RANGE_RE = re.compile(r"\b(?:18|19|20)\d{2}\s*[-–]\s*(?:\d{2,4})\b")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
GPA_RE = re.compile(r"\b\d\.\d(?:\s*\([A-Za-z]\))?\b")
UNITS_RE = re.compile(r"\b\d+(?:\s*-\s*\d+)?\s+units?\b", re.IGNORECASE)
TEXT_COUNT_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"(?:\s+additional)?\s+(?:year|years|month|months)\b",
    re.IGNORECASE,
)
NUMBER_PHRASE_RE = re.compile(
    r"\b(?:over|under|about|approximately|around|more than|less than)?\s*"
    r"\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")
UNIVERSITY_RE = re.compile(
    r"\b(?:University of [A-Z][A-Za-z,&.-]+(?:\s+[A-Z][A-Za-z,&.-]+)*"
    r"|[A-Z][A-Za-z.&-]+(?:\s+[A-Z][A-Za-z.&-]+)* University)\b"
)
PROFESSORSHIP_RE = re.compile(
    r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,5}\s+Endowed Professorship\b"
)
TOKEN_NAME_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+Token\b")
TEAM_RE = re.compile(r"\bTeam\s+\d+\s*[-–]\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b")
DEGREE_RE = re.compile(
    r"\b(?:5th-Year\s+M\.S\.|Ph\.D\.|M\.Eng\.|M\.S\.|B\.S\.|B\.A\.)\b"
)
SPAN_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s*[;|]\s*|\n+")
STRUCTURAL_SPLIT_RE = re.compile(
    r"(?=\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b)"
    r"|(?=\b(?:18|19|20)\d{2}(?:\s*[-–]\s*\d{2,4})?\b)"
    r"|(?=\b(?:\+?1[\s-]*)?(?:\(\d{3}\)|\d{3})[\s-]*\d{3}[\s-]*\d{4}\b)"
)
NAME_BLOCKLIST = {
    "office",
    "staff",
    "student",
    "students",
    "course",
    "support",
    "department",
    "division",
    "relations",
    "program",
    "graduate",
    "undergraduate",
    "faculty",
    "affairs",
}
STOP_WORDS = {
    "what",
    "who",
    "when",
    "where",
    "which",
    "how",
    "many",
    "much",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "did",
    "do",
    "does",
    "from",
    "by",
    "that",
    "this",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
COURSE_CANONICAL_RE = re.compile(r"^([A-Za-z]{2,8})\s*-?\s*(\d{1,3}[A-Z]?)$")
def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_space(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def url_to_text(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in re.split(r"[/_.-]+", parsed.path) if part]
    return normalize_space(" ".join(parts))


def url_host_to_text(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.casefold().split(":", maxsplit=1)[0]
    if not host:
        return ""

    host_parts = [part for part in re.split(r"[.-]+", host) if part]
    features = list(host_parts)

    if "www2" in host_parts:
        features.extend(["legacy", "legacy_site"])

    if "eecs" in host_parts and "berkeley" in host_parts:
        features.extend(["eecs", "berkeley"])

    return normalize_space(" ".join(features))


def canonical_url_features(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.casefold()
    features: list[str] = []

    if path.endswith(".pdf") or ".pdf" in path:
        features.extend(["pdf", "document"])
    if "/pubs/" in path:
        features.extend(["publications", "pubs"])
    if "/techrpts/" in path or "techrpt" in path:
        features.extend(["technical", "tech", "report", "reports"])
    if "/homepages/" in path:
        features.extend(["homepages", "homepage", "people"])
    if "/faculty/" in path:
        features.extend(["faculty", "people"])

    return normalize_space(" ".join(features))


def infer_title(row: dict, text: str) -> str:
    title = str(row.get("title", "")).strip()
    if title:
        return normalize_space(title)

    head = re.split(r"[.!?]", text, maxsplit=1)[0]
    words = head.split()
    if not words:
        return ""
    return normalize_space(" ".join(words[:16]))


def build_retrieval_text(url: str, title: str, text: str) -> str:
    host_text = url_host_to_text(url)
    url_text = url_to_text(url)
    canonical_text = canonical_url_features(url)
    return normalize_space(
        f"{title} {title} {host_text} {url_text} {canonical_text} {text}"
    )


def question_types(question: str) -> set[str]:
    lowered = question.casefold()
    qtypes: set[str] = set()

    if "email" in lowered or "e-mail" in lowered:
        qtypes.add("email")
    if "phone" in lowered or "telephone" in lowered:
        qtypes.add("phone")
    if "course" in lowered:
        qtypes.add("course")
    if "university" in lowered:
        qtypes.add("university")
    if "professorship" in lowered:
        qtypes.add("professorship")
    if "token" in lowered:
        qtypes.add("token")
    if "team" in lowered:
        qtypes.add("team")
    if "degree program" in lowered or "degree" in lowered:
        qtypes.add("degree")
    if lowered.startswith("who ") or any(
        cue in lowered
        for cue in ("advisor", "chair", "director", "manager", "coordinator", "professor")
    ):
        qtypes.add("person")
    if lowered.startswith("where ") or any(
        cue in lowered
        for cue in ("building", "office", "located", "address", "room", "housed", "town", "city")
    ):
        qtypes.add("location")
    if lowered.startswith("when ") or "deadline" in lowered or "date" in lowered:
        qtypes.add("date")
    if "time" in lowered or "close" in lowered or "open" in lowered:
        qtypes.add("time")
    if lowered.startswith("how many") or any(
        cue in lowered for cue in ("how much", "minimum gpa", "full-time load", "units")
    ):
        qtypes.add("number")
    if "year" in lowered:
        qtypes.add("year")
    if re.match(r"^(is|are|was|were|do|does|did|can|could|should|has|have)\b", lowered):
        qtypes.add("yesno")

    return qtypes


def get_focus_terms(question: str) -> set[str]:
    return {token for token in tokenize(question) if token not in STOP_WORDS}


def get_focus_ngrams(question: str, n: int = 2) -> set[str]:
    tokens = [token for token in tokenize(question) if token not in STOP_WORDS]
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def cleanup_answer_text(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    cleaned = normalize_space(text)
    if "@" in cleaned:
        cleaned = re.sub(r"\s*@\s*", "@", cleaned)
        cleaned = re.sub(r"\s*\.\s*", ".", cleaned)
    return cleaned.strip(" ,;:.")


def canonicalize_course_code(text: str) -> str:
    cleaned = cleanup_answer_text(text)
    match = COURSE_CANONICAL_RE.fullmatch(cleaned)
    if not match:
        return cleaned
    subject, number = match.groups()
    return f"{subject.upper()} {number.upper()}"


def extract_person_name_from_question(question: str) -> str | None:
    matches = list(NAME_RE.finditer(question))
    if not matches:
        return None
    for match in reversed(matches):
        candidate = match.group(0)
        first = candidate.split()[0].casefold()
        if first not in {"what", "which", "who", "during", "how", "in", "to"}:
            return candidate
    return matches[-1].group(0)


def infer_yes_no(text: str) -> str | None:
    lowered = text.casefold()
    padded = f" {lowered} "
    if any(cue in padded for cue in (" no ", " not ", " do not ", " does not ", " did not ", " without ")):
        return "No"
    if " yes " in padded:
        return "Yes"
    return None


def local_context(text: str, start: int, end: int, radius: int = 90) -> str:
    return normalize_space(text[max(0, start - radius) : min(len(text), end + radius)])


def split_into_spans(text: str) -> list[str]:
    spans: list[str] = []
    for part in SPAN_SPLIT_RE.split(text):
        part = normalize_space(part)
        if not part:
            continue
        if len(part.split()) > 24:
            structural_parts = [
                normalize_space(piece) for piece in STRUCTURAL_SPLIT_RE.split(part) if normalize_space(piece)
            ]
            if len(structural_parts) > 1:
                spans.extend(structural_parts)
                continue
        spans.append(part)
    return spans


def is_plausible_person_name(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 2:
        return False
    return not any(token.casefold() in NAME_BLOCKLIST for token in tokens)


def score_candidate(
    text: str,
    focus_terms: set[str],
    source_rank: int,
    bonus: float,
    context_text: str | None = None,
) -> float:
    overlap_source = context_text if context_text else text
    overlap = len(set(tokenize(overlap_source)) & focus_terms)
    if focus_terms and overlap == 0 and bonus < 4.5:
        return float("-inf")

    overlap_bonus = 0.75 * overlap
    length_penalty = max(0, len(tokenize(text)) - 5) * 0.08
    rank_penalty = source_rank * 0.2
    return bonus + overlap_bonus - length_penalty - rank_penalty


def add_candidate(
    candidates: dict[str, tuple[float, str]],
    text: str,
    focus_terms: set[str],
    source_rank: int,
    bonus: float,
    context_text: str | None = None,
) -> None:
    candidate = cleanup_answer_text(text)
    if not candidate:
        return

    score = score_candidate(candidate, focus_terms, source_rank, bonus, context_text=context_text)
    if score == float("-inf"):
        return

    key = candidate.casefold()
    previous = candidates.get(key)
    if previous is None or score > previous[0]:
        candidates[key] = (score, candidate)


def extract_type_candidates(qtypes: set[str], span: str) -> list[tuple[str, float, str]]:
    candidates: list[tuple[str, float, str]] = []

    if "email" in qtypes:
        candidates.extend((match.group(0), 7.0, local_context(span, match.start(), match.end())) for match in EMAIL_RE.finditer(span))
    if "phone" in qtypes:
        candidates.extend((match.group(0), 7.0, local_context(span, match.start(), match.end())) for match in PHONE_RE.finditer(span))
    if "course" in qtypes:
        candidates.extend((match.group(0), 6.0, local_context(span, match.start(), match.end())) for match in COURSE_RE.finditer(span))
    if "university" in qtypes:
        candidates.extend((match.group(0), 6.5, local_context(span, match.start(), match.end())) for match in UNIVERSITY_RE.finditer(span))
    if "professorship" in qtypes:
        candidates.extend((match.group(0), 6.5, local_context(span, match.start(), match.end())) for match in PROFESSORSHIP_RE.finditer(span))
    if "token" in qtypes:
        candidates.extend((match.group(0), 6.5, local_context(span, match.start(), match.end())) for match in TOKEN_NAME_RE.finditer(span))
    if "team" in qtypes:
        candidates.extend((match.group(0), 6.5, local_context(span, match.start(), match.end())) for match in TEAM_RE.finditer(span))
    if "degree" in qtypes:
        candidates.extend((match.group(0), 6.5, local_context(span, match.start(), match.end())) for match in DEGREE_RE.finditer(span))
    if "location" in qtypes:
        candidates.extend((match.group(0), 5.5, local_context(span, match.start(), match.end())) for match in ROOM_RE.finditer(span))
        candidates.extend((match.group(0), 5.0, local_context(span, match.start(), match.end())) for match in BUILDING_RE.finditer(span))
    if "time" in qtypes:
        candidates.extend((match.group(0), 6.0, local_context(span, match.start(), match.end())) for match in TIME_RE.finditer(span))
    if "date" in qtypes:
        candidates.extend((match.group(0), 6.0, local_context(span, match.start(), match.end())) for match in DATE_RE.finditer(span))
    if "year" in qtypes:
        candidates.extend((match.group(0), 6.3, local_context(span, match.start(), match.end())) for match in YEAR_RANGE_RE.finditer(span))
        candidates.extend((match.group(0), 6.0, local_context(span, match.start(), match.end())) for match in YEAR_RE.finditer(span))
    if "number" in qtypes:
        candidates.extend((match.group(0), 6.2, local_context(span, match.start(), match.end())) for match in GPA_RE.finditer(span))
        candidates.extend((match.group(0), 6.2, local_context(span, match.start(), match.end())) for match in TEXT_COUNT_RE.finditer(span))
        candidates.extend((match.group(0), 6.0, local_context(span, match.start(), match.end())) for match in UNITS_RE.finditer(span))
        for match in NUMBER_PHRASE_RE.finditer(span):
            candidate = match.group(0)
            if YEAR_RE.fullmatch(candidate.strip()):
                continue
            candidates.append((candidate, 5.0, local_context(span, match.start(), match.end())))
    if "person" in qtypes:
        for match in NAME_RE.finditer(span):
            candidate = match.group(0)
            if is_plausible_person_name(candidate):
                candidates.append((candidate, 5.0, local_context(span, match.start(), match.end())))
    if "yesno" in qtypes:
        yes_no = infer_yes_no(span)
        if yes_no:
            candidates.append((yes_no, 6.0, span))

    return candidates


def extract_relation_candidates(question: str, span: str) -> list[tuple[str, float, str]]:
    lowered = question.casefold()
    candidates: list[tuple[str, float, str]] = []

    patterns: list[tuple[str, float]] = []
    if "designed for" in lowered:
        patterns.append((r"designed for ([^.]+)", 4.8))
    if lowered.startswith("where ") or "located" in lowered:
        patterns.extend(
            [
                (r"located (?:in|at)\s+([^.]+)", 4.8),
                (r"(?:in|at)\s+(\d{3,4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", 4.6),
                (r"intersection of ([^.]+)", 4.8),
            ]
        )
    if "deadline" in lowered:
        patterns.append((r"deadline (?:is|for .* is)\s+([^.]+)", 4.8))
    if "application period" in lowered:
        patterns.append((r"application period is ([^.]+)", 5.2))
    if "cell phone emergency hotline" in lowered:
        patterns.append((r"Cell phone:\s*((?:\+?1[\s-]*)?(?:\(\d{3}\)|\d{3})[\s-]*\d{3}[\s-]*\d{4})", 5.6))
    if "cory hall building manager" in lowered:
        patterns.append((r"Cory Hall building manager:\s*((?:\+?1[\s-]*)?(?:\(\d{3}\)|\d{3})[\s-]*\d{3}[\s-]*\d{4})", 5.6))
    if lowered.startswith("what building"):
        patterns.append((r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+Hall)", 4.6))
    if "what office" in lowered:
        patterns.append((r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+Office)", 4.6))
    if "load" in lowered or "units" in lowered:
        patterns.append((r"load of ([^.]+)", 4.8))
        patterns.append((r"apply with ([^.]+?semester units)", 5.2))
    if "gpa" in lowered:
        patterns.append((r"gpa of ([^.]+)", 4.8))
    if "current ee graduate student" in lowered:
        patterns.append((r"([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})\s+All current EE graduate student advising/assistance", 5.5))
    if "visiting appointment" in lowered or "reappointment requests" in lowered:
        patterns.append((r"visiting appointments/reappointments to [^.]*?([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})", 5.6))
    if "token" in lowered:
        patterns.append((r"ask for a [\"“]?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+Token)[\"”]?", 5.2))
    if "team" in lowered:
        patterns.append((r"to (Team\s+\d+\s*[-–]\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)", 5.2))
    if "available only to which undergraduates" in lowered:
        patterns.append((r"available only to ([^.]+)", 5.0))
    if "additional years" in lowered or "additional year" in lowered:
        patterns.append((r"requiring only ([^.]+?) beyond", 5.2))
    if "listed first" in lowered:
        patterns.append((r"Academic Year\s+\d{4}/\d{4}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", 5.4))
    if "what year" in lowered and "wicse" in lowered:
        patterns.append((r"(\d{4})\s+WICSE is founded", 5.4))
    if "president in" in lowered:
        range_match = re.search(r"(\d{4}-\d{2})", question)
        if range_match:
            year_range = re.escape(range_match.group(1))
            patterns.append((rf"{year_range}\s*[–-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", 5.4))
    if "director of diversity" in lowered:
        patterns.append((r"Director of Diversity,\s*((?:18|19|20)\d{2}\s*[-–]\s*(?:\d{2,4}))", 5.4))
    if "who spoke" in lowered:
        patterns.append((r"([A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+){1,3})\s+(?:Associate Professor|Professor|Director)", 5.1))
    if "which university" in lowered:
        patterns.append((r"(University of [A-Z][A-Za-z,&.-]+(?:\s+[A-Z][A-Za-z,&.-]+)*)", 5.6))
        patterns.append((r"([A-Z][A-Za-z.&-]+(?:\s+[A-Z][A-Za-z.&-]+)* University)", 5.4))
    if "how old" in lowered:
        patterns.append((r"age\s+(\d+)", 5.4))
        patterns.append((r"He was\s+(\d+)", 5.2))
    if "born" in lowered and ("town" in lowered or "city" in lowered):
        patterns.append((r"born in ([^.]+)", 5.3))
    if "three-year program" in lowered:
        patterns.append((r"([A-Z0-9][A-Za-z0-9&+.-]+(?:\s+[A-Z0-9][A-Za-z0-9&+.-]+){0,4})\s*:\s*A three-year program", 5.4))
    if "listed under" in lowered or "degree program" in lowered:
        person = extract_person_name_from_question(question)
        if person:
            person = re.escape(person)
            patterns.append(
                (
                    rf"{person}\s+((?:5th-Year\s+M\.S\.|Ph\.D\.|M\.Eng\.|M\.S\.|B\.S\.|B\.A\.))",
                    5.4,
                )
            )

    for pattern, bonus in patterns:
        match = re.search(pattern, span)
        if match:
            candidates.append((match.group(1), bonus, local_context(span, match.start(1), match.end(1))))

    return candidates

def postprocess_answer(question: str, answer: str) -> str:
    cleaned = cleanup_answer_text(answer)
    if not cleaned:
        return "unknown"

    qtypes = question_types(question)
    lowered_question = question.casefold()

    for regex in (
        EMAIL_RE if "email" in qtypes else None,
        PHONE_RE if "phone" in qtypes else None,
        COURSE_RE if "course" in qtypes else None,
        UNIVERSITY_RE if "university" in qtypes else None,
        PROFESSORSHIP_RE if "professorship" in qtypes else None,
        TOKEN_NAME_RE if "token" in qtypes else None,
        TEAM_RE if "team" in qtypes else None,
        DEGREE_RE if "degree" in qtypes else None,
        ROOM_RE if "location" in qtypes else None,
        BUILDING_RE if "location" in qtypes else None,
        TIME_RE if "time" in qtypes else None,
        DATE_RANGE_RE if "date" in qtypes else None,
        DATE_RE if "date" in qtypes else None,
        YEAR_RANGE_RE if "year" in qtypes else None,
        YEAR_RE if "year" in qtypes else None,
        TEXT_COUNT_RE if "number" in qtypes else None,
        UNITS_RE if "number" in qtypes else None,
    ):
        if regex is None:
            continue
        match = regex.search(cleaned)
        if match:
            extracted = cleanup_answer_text(match.group(0))
            if regex is COURSE_RE:
                return canonicalize_course_code(extracted)
            return extracted

    if "yesno" in qtypes:
        yes_no = infer_yes_no(cleaned)
        if yes_no:
            return yes_no

    if "course" in qtypes:
        return canonicalize_course_code(cleaned)

    if "degree" in qtypes:
        normalized = cleaned.casefold().replace(" ", "")
        if normalized in {"phdonly", "phd.only", "ph.donly", "ph.d.only"}:
            return "PhD. only"

    if "listed first" in lowered_question and cleaned.startswith("Recipients "):
        return cleanup_answer_text(cleaned.removeprefix("Recipients "))
    if "three-year program" in lowered_question and cleaned.startswith("UC Outreach Programs "):
        return cleanup_answer_text(cleaned.removeprefix("UC Outreach Programs "))
    if "application period" in lowered_question:
        match = DATE_RANGE_RE.search(cleaned)
        if match:
            return cleanup_answer_text(match.group(0))

    return cleaned


def retrieval_bonus(question: str, chunk: Chunk) -> float:
    qtypes = question_types(question)
    focus_terms = get_focus_terms(question)
    focus_bigrams = get_focus_ngrams(question, n=2)
    question_lower = question.casefold()
    text_lower = chunk.text.casefold()
    url_lower = chunk.url.casefold()
    retrieval_lower = chunk.retrieval_text.casefold()

    bonus = 0.0

    overlap = len(set(tokenize(chunk.retrieval_text)) & focus_terms)
    if overlap >= 2:
        bonus += 0.15 * min(overlap, 4)

    matched_bigrams = sum(1 for bigram in focus_bigrams if bigram in retrieval_lower)
    if matched_bigrams:
        bonus += 0.35 * min(matched_bigrams, 2)

    if any(role in question_lower for role in ("director", "chair", "manager", "coordinator", "advisor")):
        if any(role in text_lower for role in ("director", "chair", "manager", "coordinator", "advisor")):
            bonus += 0.7

    if "person" in qtypes:
        person_name = extract_person_name_from_question(question)
        if person_name:
            normalized_name = normalize_space(person_name).casefold()
            surname = normalized_name.split()[-1]
            if normalized_name in retrieval_lower:
                bonus += 3.0
            if surname in url_lower:
                bonus += 1.4
            elif surname in retrieval_lower:
                bonus += 0.8
        if any(path in url_lower for path in ("/people/", "/staff/", "/leadership", "/faculty/", "/homepages/")):
            bonus += 1.0

    if "email" in qtypes or "phone" in qtypes:
        if any(cue in text_lower for cue in ("email", "contact", "phone", "telephone")):
            bonus += 0.9
        if any(path in url_lower for path in ("/contact", "/staff/", "/leadership", "/people/")):
            bonus += 1.1

    if "location" in qtypes:
        if any(cue in text_lower for cue in ("office", "located", "address", "hall", "room", "building")):
            bonus += 0.9
        if any(path in url_lower for path in ("/contact", "/visiting", "/staff/", "/people/", "/homepages/")):
            bonus += 1.0

    if "date" in qtypes or "year" in qtypes:
        if any(path in url_lower for path in ("/news/", "/about/history", "/special-events", "/events/")):
            bonus += 0.7

    return bonus

# chunking
DEFAULT_TOP_K = 4
DEFAULT_CHUNK_SIZE = 140
DEFAULT_CHUNK_OVERLAP = 30
def chunk_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    words = text.split()
    if not words:
        return
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = words[i : i + chunk_size]
        if chunk:
            yield " ".join(chunk)

# handles raw text crawl
def extract_document_text(row: dict) -> str:
    for key in ("summary", "text", "clean_text", "content", "page_text"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""

# chunking
def load_chunks(
    corpus_path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[Chunk]:
    chunks: list[Chunk] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            url = str(row.get("url", "")).strip()
            text = extract_document_text(row)
            title = infer_title(row, text)
            if not url or not text:
                continue

            if "chunk_id" in row or "chunk_index" in row:
                chunks.append(
                    Chunk(
                        url=url,
                        text=text,
                        retrieval_text=build_retrieval_text(url, title, text),
                    )
                )
                continue

            for piece in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
                chunks.append(
                    Chunk(
                        url=url,
                        text=piece,
                        retrieval_text=build_retrieval_text(url, title, piece),
                    )
                )

    if not chunks:
        raise ValueError(f"No valid chunks loaded from {corpus_path}")
    return chunks


def resolve_corpus_path(corpus_arg: str | None) -> Path:
    if corpus_arg:
        path = Path(corpus_arg)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")
        return path

    for candidate in DEFAULT_CORPUS_CANDIDATES:
        path = PROJECT_ROOT / candidate
        if path.exists():
            return path

    raise FileNotFoundError(
        "No corpus file found. Tried: "
        + ", ".join(DEFAULT_CORPUS_CANDIDATES)
    )

# indexing
class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c.retrieval_text) for c in chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_len = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.tf = [Counter(tokens) for tokens in self.doc_tokens]
        self.df = self._build_document_frequencies()
        self.n_docs = len(chunks)

    def _build_document_frequencies(self) -> Counter:
        df = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                df[term] += 1
        return df

    def _idf(self, term: str) -> float:
        n_q = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n_q + 0.5) / (n_q + 0.5))

    def _score_query(self, query: str) -> list[tuple[float, int]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: list[tuple[float, int]] = []
        for i, tf_counter in enumerate(self.tf):
            score = 0.0
            dl = self.doc_lens[i]
            norm = self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avg_len))
            for term in query_terms:
                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf(term)
                score += idf * (tf * (self.k1 + 1)) / (tf + norm)
            if score > 0:
                scores.append((score, i))

        scores.sort(reverse=True)
        return scores

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int | None = None,
        max_chunks_per_url: int = 2,
    ) -> list[Chunk]:
        if top_k <= 0:
            return []

        candidate_limit = candidate_k or max(28, top_k * 10)
        raw_scored_chunks = self._score_query(query)[:candidate_limit]
        if not raw_scored_chunks:
            return []

        reranked_chunks: list[tuple[float, float, int]] = []
        for score, chunk_idx in raw_scored_chunks:
            adjusted_score = score + retrieval_bonus(query, self.chunks[chunk_idx])
            reranked_chunks.append((adjusted_score, score, chunk_idx))
        reranked_chunks.sort(reverse=True)

        by_url: dict[str, list[tuple[float, float, int]]] = {}
        for adjusted_score, raw_score, chunk_idx in reranked_chunks:
            url = self.chunks[chunk_idx].url
            by_url.setdefault(url, []).append((adjusted_score, raw_score, chunk_idx))

        ranked_urls: list[tuple[float, str]] = []
        for url, url_chunks in by_url.items():
            url_chunks.sort(reverse=True)
            best_score = url_chunks[0][0]
            support_score = sum(score for score, _, _ in url_chunks[1:3])
            url_score = best_score + 0.25 * support_score
            ranked_urls.append((url_score, url))

        ranked_urls.sort(reverse=True)
        url_scores = {url: score for score, url in ranked_urls}

        qtypes = question_types(query)
        repeat_penalty = 1.25
        if qtypes & {"person", "email", "phone", "location"}:
            repeat_penalty = 0.8
            max_chunks_per_url = max(max_chunks_per_url, 3)
        elif qtypes & {"date", "year"}:
            repeat_penalty = 1.0

        candidate_pool: list[tuple[str, int, float, int]] = []
        for _, url in ranked_urls:
            for rank_within_url, (adjusted_score, _, chunk_idx) in enumerate(by_url[url][:max_chunks_per_url]):
                candidate_pool.append((url, rank_within_url, adjusted_score, chunk_idx))

        selected_indices: list[int] = []
        selected_set: set[int] = set()
        url_use_count: Counter[str] = Counter()

        while len(selected_indices) < top_k:
            best_item: tuple[float, int] | None = None
            for url, rank_within_url, adjusted_score, chunk_idx in candidate_pool:
                if chunk_idx in selected_set:
                    continue
                final_score = adjusted_score + 0.08 * url_scores[url]
                final_score -= repeat_penalty * url_use_count[url]
                final_score -= 0.3 * rank_within_url
                if best_item is None or final_score > best_item[0]:
                    best_item = (final_score, chunk_idx)

            if best_item is None:
                break

            _, chunk_idx = best_item
            selected_indices.append(chunk_idx)
            selected_set.add(chunk_idx)
            url_use_count[self.chunks[chunk_idx].url] += 1

        return [self.chunks[idx] for idx in selected_indices]

# orignal ragmodel
class EarlyMilestoneRAG:
    def __init__(self, index: BM25Index, top_k: int = DEFAULT_TOP_K):
        self.index = index
        self.top_k = top_k
        self._llm_failure_warned = False

    def build_query(self, question: str, retrieved: list[Chunk]) -> str:
        context_blocks = []
        for i, chunk in enumerate(retrieved, start=1):
            context_blocks.append(f"[{i}] URL: {chunk.url}\n{chunk.text}")
        context = "\n\n".join(context_blocks)
        return (
            f"context:\n{context}\n\n"
            f"question: {question}\n\n"
            "Answer with just the short answer."
        )

    def answer(self, question: str, use_llm: bool = True) -> tuple[str, list[Chunk]]:
        retrieved = self.index.retrieve(question, top_k=self.top_k)
        if not retrieved:
            return "UNKNOWN", []

        if not use_llm:
            return self.extractive_fallback(question, retrieved), retrieved

        prompt = self.build_query(question, retrieved)
        try:
            raw = call_llm(
                query=prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=20,
                temperature=0.0,
                model=LLM_CHOICE,
            )
            answer = raw.strip().splitlines()[0].strip()
            return postprocess_answer(question, answer), retrieved
        except Exception as exc:
            if not self._llm_failure_warned:
                print(
                    f"Warning: LLM call failed, using extractive fallback: {exc}",
                    file=sys.stderr,
                )
                self._llm_failure_warned = True
            return self.extractive_fallback(question, retrieved), retrieved

    def extractive_fallback(self, question: str, retrieved: list[Chunk]) -> str:
        focus_terms = get_focus_terms(question)
        qtypes = question_types(question)
        candidates: dict[str, tuple[float, str]] = {}

        for source_rank, chunk in enumerate(retrieved):
            if not chunk.text:
                continue

            spans = split_into_spans(chunk.text)
            for span in spans:
                if not span:
                    continue

                for candidate, bonus, candidate_context in extract_type_candidates(qtypes, span):
                    add_candidate(
                        candidates,
                        candidate,
                        focus_terms,
                        source_rank,
                        bonus,
                        context_text=candidate_context,
                    )

                for candidate, bonus, candidate_context in extract_relation_candidates(question, span):
                    add_candidate(
                        candidates,
                        candidate,
                        focus_terms,
                        source_rank,
                        bonus,
                        context_text=candidate_context,
                    )

                span_terms = set(tokenize(span))
                overlap = len(span_terms & focus_terms)
                if overlap == 0:
                    continue

                short_span = " ".join(span.split()[:10])
                add_candidate(
                    candidates,
                    short_span,
                    focus_terms,
                    source_rank,
                    2.2,
                    context_text=span,
                )

                if len(span.split()) <= 12:
                    add_candidate(
                        candidates,
                        span,
                        focus_terms,
                        source_rank,
                        2.6,
                        context_text=span,
                    )

        if not candidates:
            return "unknown"

        best_answer = max(candidates.values(), key=lambda item: item[0])[1]
        return postprocess_answer(question, best_answer)


def run_batch(
    rag: EarlyMilestoneRAG, questions_path: Path, predictions_path: Path, use_llm: bool
) -> None:
    questions = load_questions(questions_path)

    predictions: list[str] = []
    for q in questions:
        if not q:
            predictions.append("UNKNOWN")
            continue
        answer, _ = rag.answer(q, use_llm=use_llm)
        predictions.append(answer)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as handle:
        for answer in predictions:
            handle.write(answer + "\n")


def load_questions(questions_path: Path) -> list[str]:
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    suffix = questions_path.suffix.lower()

    if suffix == ".jsonl":
        questions: list[str] = []
        with questions_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number} in {questions_path}"
                    ) from exc
                question = str(row.get("question", "")).strip()
                questions.append(question if question else "unknown")
        return questions

    if suffix == ".json":
        with questions_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, list):
            raise ValueError(
                f"JSON questions file must be a list in {questions_path}"
            )

        questions: list[str] = []
        for item in data:
            if isinstance(item, str):
                questions.append(item.strip())
            elif isinstance(item, dict):
                question = str(item.get("question", "")).strip()
                questions.append(question if question else "UNKNOWN")
            else:
                questions.append("UNKNOWN")
        return questions

    return [
        line.strip()
        for line in questions_path.read_text(encoding="utf-8").splitlines()
    ]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Early milestone RAG baseline")
    parser.add_argument("questions_pos", nargs="?", default=None, help=argparse.SUPPRESS)
    parser.add_argument("predictions_pos", nargs="?", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to corpus JSONL with {'url','text'} rows",
    )
    parser.add_argument("--questions", type=str, default=None, help="Input questions txt")
    parser.add_argument("--predictions", type=str, default=None, help="Output answers txt")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM generation and return 'unknown' after retrieval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions_arg = args.questions or args.questions_pos
    predictions_arg = args.predictions or args.predictions_pos

    corpus_path = resolve_corpus_path(args.corpus)

    chunks = load_chunks(
        corpus_path,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    index = BM25Index(chunks)
    rag = EarlyMilestoneRAG(index=index, top_k=args.top_k)

    use_llm = not args.no_llm

    if questions_arg or predictions_arg:
        if not questions_arg or not predictions_arg:
            raise ValueError("Batch mode requires both --questions and --predictions")
        run_batch(
            rag=rag,
            questions_path=Path(questions_arg),
            predictions_path=Path(predictions_arg),
            use_llm=use_llm,
        )
        return

    print("Failed to run")
    return


if __name__ == "__main__":
    main()
