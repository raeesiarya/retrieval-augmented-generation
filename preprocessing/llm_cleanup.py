from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypedDict
from huggingface_hub import get_token

HF_TOKEN = get_token()

try:
    from preprocessing.crawl_eecs import get_urls, process_urls
except ImportError:
    from crawl_eecs import get_urls, process_urls


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MAX_INPUT_TOKENS = 4000
DEFAULT_MAX_NEW_TOKENS = 120
DEFAULT_TEMPERATURE = 0.1
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"

DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_DIR))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

WHITESPACE_RE = re.compile(r"\s+")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*", re.MULTILINE)
INLINE_LIST_ITEM_RE = re.compile(r"(?:^|\s)(?:[-*]|\d+[.)])\s+(?=[A-Z0-9])")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
SUMMARY_PREFIX_RE = re.compile(
    r"^(?:summary|three-sentence summary|here(?:'s| is) (?:the )?summary)\s*:\s*",
    re.IGNORECASE,
)
BOILERPLATE_LINE_RE = re.compile(
    r"^(?:home|menu|search|skip to main content|breadcrumb|breadcrumbs|open menu|close menu)$",
    re.IGNORECASE,
)
EMPTY_SUMMARY = (
    "This page contains limited substantive EECS information. "
    "Most of the available text appears to be navigational, repetitive, or administrative material. "
    "No additional page-specific details could be extracted reliably from the crawl output."
)

SYSTEM_PROMPT = (
    "You are an expert information extraction assistant building a retrieval corpus for the "
    "UC Berkeley EECS website. Produce concise factual summaries that keep only the main "
    "informational content of the page."
)

USER_PROMPT_TEMPLATE = """URL: {url}

Write exactly THREE sentences describing the main purpose and content of this page.

Ignore navigation menus, repeated lists, newsletter archives, event archives, breadcrumbs, headers, footers, UI text, and raw link collections.

Focus on meaningful informational content such as EECS programs, faculty, research, departmental initiatives, announcements, events with explanations, and resources for students or faculty.

Return only the summary in plain prose. Do not use bullets or numbering.

Webpage text:
{page_text}"""


class RawDocument(TypedDict):
    url: str
    text: str


class SummaryDocument(TypedDict):
    url: str
    summary: str


def normalize_space(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_webpage_text(text: str) -> str:
    text = text.replace("\x00", " ")
    lines = text.splitlines()

    if len(lines) <= 1:
        return normalize_space(text)

    cleaned_lines: list[str] = []
    seen: set[str] = set()

    for raw_line in lines:
        line = normalize_space(raw_line)
        if not line or BOILERPLATE_LINE_RE.fullmatch(line):
            continue

        line_key = line.casefold()
        if line_key in seen:
            continue

        seen.add(line_key)
        cleaned_lines.append(line)

    return normalize_space(" ".join(cleaned_lines))


def clean_generated_summary(text: str) -> str:
    text = SUMMARY_PREFIX_RE.sub("", text.strip(), count=1)
    text = LIST_ITEM_RE.sub("", text)
    text = text.replace("\n", " ")
    text = INLINE_LIST_ITEM_RE.sub(" ", text)
    text = normalize_space(text)
    return text.strip(" \"'")


def split_sentences(text: str) -> list[str]:
    cleaned = clean_generated_summary(text)
    if not cleaned:
        return []

    sentences: list[str] = []
    for fragment in SENTENCE_SPLIT_RE.split(cleaned):
        sentence = fragment.strip(" \"'")
        if not sentence:
            continue
        if sentence[-1] not in ".!?":
            sentence = sentence.rstrip(",;:") + "."
        sentences.append(sentence)

    return sentences


def coerce_to_three_sentences(text: str) -> str:
    sentences = split_sentences(text)

    if len(sentences) >= 3:
        return " ".join(sentences[:3])

    if len(sentences) == 2:
        return " ".join(
            sentences
            + [
                "Navigation elements, repeated lists, and boilerplate were omitted from this summary."
            ]
        )

    if len(sentences) == 1:
        return " ".join(
            [
                sentences[0],
                "The remaining page text was limited or noisy, so only the clearest information was retained.",
                "Navigation elements, repeated lists, and boilerplate were omitted from this summary.",
            ]
        )

    return EMPTY_SUMMARY


def ensure_document_schema(document: dict[str, Any]) -> RawDocument:
    url = normalize_space(str(document.get("url", "")))
    text = str(document.get("text", ""))

    if not url:
        raise ValueError("Each document must include a non-empty 'url'.")

    return {"url": url, "text": text}


@dataclass(slots=True)
class LlamaPageSummarizer:
    model_name: str = MODEL_NAME
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    cache_dir: Path = DEFAULT_CACHE_DIR
    hf_token: str | None = None
    tokenizer: Any = None
    model: Any = None

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.cache_dir))

        self.hf_token = (
            self.hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or HF_TOKEN
        )

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            token=self.hf_token,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            token=self.hf_token,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        self.input_device = next(self.model.parameters()).device

        self.terminators = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id >= 0:
            self.terminators.append(eot_id)

    def _build_prompt(self, url: str, page_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(url=url, page_text=page_text),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _truncate_page_text(self, url: str, text: str) -> str:
        cleaned_text = normalize_webpage_text(text)
        if not cleaned_text:
            return ""

        prompt_without_text = self._build_prompt(url, "")
        prompt_tokens = self.tokenizer(
            prompt_without_text,
            add_special_tokens=False,
        )["input_ids"]
        available_tokens = max(256, self.max_input_tokens - len(prompt_tokens))

        text_tokens = self.tokenizer(
            cleaned_text,
            add_special_tokens=False,
        )["input_ids"]
        if len(text_tokens) <= available_tokens:
            return cleaned_text

        truncated_text = self.tokenizer.decode(
            text_tokens[:available_tokens],
            skip_special_tokens=True,
        )
        return normalize_space(truncated_text)

    def _generate_summary(self, url: str, page_text: str) -> str:
        prompt = self._build_prompt(url, page_text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {name: tensor.to(self.input_device) for name, tensor in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.terminators,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def summarize_document(self, document: RawDocument) -> SummaryDocument:
        url = document["url"]
        truncated_text = self._truncate_page_text(url, document["text"])

        if not truncated_text:
            return {"url": url, "summary": EMPTY_SUMMARY}

        raw_summary = self._generate_summary(url, truncated_text)
        summary = coerce_to_three_sentences(raw_summary)

        return {"url": url, "summary": summary}

    def summarize_documents(
        self,
        documents: Iterable[dict[str, Any]],
        show_progress: bool = True,
    ) -> list[SummaryDocument]:
        normalized_documents = [
            ensure_document_schema(document) for document in documents
        ]
        summaries: list[SummaryDocument] = []
        iterator = tqdm(
            normalized_documents,
            total=len(normalized_documents),
            desc="Summarizing pages",
            disable=not show_progress,
        )

        for document in iterator:
            try:
                summaries.append(self.summarize_document(document))
            except Exception as exc:
                print(
                    f"Warning: failed to summarize {document['url']}: {exc}",
                    file=sys.stderr,
                )
                summaries.append({"url": document["url"], "summary": EMPTY_SUMMARY})

        return summaries


def summarize_documents(
    documents: Iterable[dict[str, Any]],
    model_name: str = MODEL_NAME,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    hf_token: str | None = None,
    show_progress: bool = True,
) -> list[SummaryDocument]:
    summarizer = LlamaPageSummarizer(
        model_name=model_name,
        cache_dir=Path(cache_dir),
        hf_token=hf_token,
    )
    return summarizer.summarize_documents(documents, show_progress=show_progress)


def load_documents(input_path: str | Path) -> list[RawDocument]:
    path = Path(input_path)
    raw_documents: Any

    if path.suffix == ".jsonl":
        raw_documents = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_documents.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} in {path}."
                    ) from exc
    else:
        with path.open("r", encoding="utf-8") as handle:
            raw_documents = json.load(handle)

    if isinstance(raw_documents, dict):
        raw_documents = raw_documents.get("documents", [])

    if not isinstance(raw_documents, list):
        raise ValueError("Input data must be a JSON array or JSONL file of documents.")

    return [ensure_document_schema(document) for document in raw_documents]


def write_summaries(summaries: list[SummaryDocument], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for summary in summaries:
                handle.write(json.dumps(summary, ensure_ascii=True) + "\n")
        return

    with path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize UC Berkeley EECS webpages with Meta-Llama-3.1-8B-Instruct.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a JSON or JSONL file containing documents with 'url' and 'text' fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for JSON or JSONL summaries. Defaults to stdout.",
    )
    parser.add_argument(
        "--crawl-limit",
        type=int,
        default=10,
        help="Number of pages to crawl when --input is not provided.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Hugging Face cache directory for model files.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. Falls back to HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:
        documents = load_documents(args.input)
    else:
        urls = get_urls(limit=args.crawl_limit)
        documents = process_urls(urls)

    summaries = summarize_documents(
        documents,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
    )

    if args.output:
        write_summaries(summaries, args.output)
        return

    print(json.dumps(summaries, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
