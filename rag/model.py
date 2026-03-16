from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# from rag.llm import call_llm
from llm import call_llm

DEFAULT_CORPUS_CANDIDATES = (
    #"data/crawl_eecs_summaries.jsonl",
    #"data/crawl_eecs_summary.jsonl",
    #"data/crawl_eecs_llm_cleanup.jsonl",
    #"data/crawl_eecs_cleaned.jsonl",
    "data/crawl_eecs_raw.jsonl",
    #"data/corpus.jsonl",
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

# modify our tokenizer, currently only considering alphanumerics
TOKEN_RE = re.compile(r"[a-z0-9]+")
def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

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
            if not url or not text:
                continue

            for piece in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
                chunks.append(Chunk(url=url, text=piece))

    if not chunks:
        raise ValueError(f"No valid chunks loaded from {corpus_path}")
    return chunks

# indexing
class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c.text) for c in chunks]
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

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[Chunk]:
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
        return [self.chunks[i] for _, i in scores[:top_k]]

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
            return answer if answer else "unknown", retrieved
        except Exception as exc:
            if not self._llm_failure_warned:
                print(
                    f"Warning: LLM call failed, using extractive fallback: {exc}",
                    file=sys.stderr,
                )
                self._llm_failure_warned = True
            return self.extractive_fallback(question, retrieved), retrieved

    def extractive_fallback(self, question: str, retrieved: list[Chunk]) -> str:
        question_tokens = [t for t in tokenize(question) if t]
        question_terms = set(question_tokens)
        stop_words = {
            "what",
            "who",
            "when",
            "where",
            "which",
            "how",
            "many",
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
        }
        focus_terms = {t for t in question_terms if t not in stop_words}

        best_phrase = "UNKNOWN"
        best_score = -1.0

        for chunk in retrieved:
            text = chunk.text
            if not text:
                continue
            spans = re.split(r"(?<=[.!?])\s+|\s*[;|]\s*", text)
            for span in spans:
                span = span.strip()
                if not span:
                    continue
                span_tokens = tokenize(span)
                if not span_tokens:
                    continue
                overlap = len(set(span_tokens) & focus_terms)
                if overlap == 0:
                    continue

                # favor compact, answer-like spans.
                length_penalty = min(len(span_tokens), 14) * 0.08
                score = overlap - length_penalty
                if score > best_score:
                    words = span.split()
                    best_phrase = " ".join(words[:10]).strip(" ,;:.")
                    best_score = score

        return best_phrase if best_phrase else "UNKNOWN"


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

    # corpus_path = resolve_corpus_path(args.corpus)
    corpus_path = Path("data/crawl_eecs_raw.jsonl") # hardcoded

    chunks = load_chunks(
        corpus_path,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    index = BM25Index(chunks)
    rag = EarlyMilestoneRAG(index=index, top_k=args.top_k)

    use_llm = not args.no_llm
    if use_llm and not os.environ.get("OPENROUTER_API_KEY", "").strip():
        print(
            "Warning: OPENROUTER_API_KEY is not set; using extractive fallback.",
            file=sys.stderr,
        )

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
