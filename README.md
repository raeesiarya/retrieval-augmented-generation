<h2 align="center">RAG-EECS: Retrieval-Augmented Question Answering over the UC Berkeley EECS Website</h2>

<p align="center">
  <strong>A retrieval-augmented generation system that answers natural language questions about UC Berkeley's EECS department by combining BM25 retrieval with an LLM reader and an extractive fallback.</strong>
</p>

<p align="center">
  Hanna Roed, Arya Raeesi, Rohan Bijukumar<br>
  <a href="mailto:hanna.roed@berkeley.edu">hanna.roed@berkeley.edu</a>,
  <a href="mailto:aryaraeesi@berkeley.edu">aryaraeesi@berkeley.edu</a>,
  <a href="mailto:rohanbijukumar@berkeley.edu">rohanbijukumar@berkeley.edu</a>
</p>

<p align="center">
  <a href="./RAG_Model_REPORT.pdf">Project Report</a>
</p>

## Overview

This repository implements a retrieval-augmented generation (RAG) pipeline for short-form question answering grounded in a crawl of the UC Berkeley EECS website. Given a free-form question, the system retrieves the most relevant pages from a local corpus, prompts an LLM with that evidence, and falls back to an extractive heuristic when the LLM is unavailable or unhelpful.

The system is designed for the CS 288 RAG assignment, where predictions are scored against a held-out reference set using exact match and F1. The focus of the project is twofold:

- build a competitive end-to-end pipeline on the visible validation distribution
- understand *where* the system is actually weak by separately measuring retrieval quality, reader quality, and fallback quality

The repository therefore includes both the production pipeline used by the autograder and a research harness used to diagnose retrieval and answer-extraction errors.

## Core Idea

Short-form QA over a web crawl can succeed for at least three reasons, and they are not all equally robust:

| Source of correctness | What it means | What it predicts about robustness |
| --- | --- | --- |
| Retrieval-grounded answer | The gold page is in top-`k` and the LLM reads it | Generalizes when the corpus and crawl shift |
| Extractive fallback | An answer is recovered directly from retrieved text without the LLM | Survives LLM behavior shifts |
| LLM prior knowledge | The LLM answers from parametric memory regardless of retrieval | Brittle under unfamiliar pages or different LLMs |

We instrument the pipeline so that each of these can be measured independently. This is what motivates the structure of `rag/eval_predictions.py` and `research/research_metrics.py`: end-to-end F1 alone hides which component is doing the work.

## Pipeline

The end-to-end flow used by the autograder is:

1. Load the crawl corpus from `data/` (default: `data/crawl_eecs.jsonl`).
2. Split each page into chunks and build a BM25 index over chunk text augmented with page titles and URL tokens.
3. For each input question, retrieve the top-`k` most relevant chunks.
4. Prompt the LLM (`rag/llm.py`) with the question and the retrieved context.
5. If the LLM call fails, times out, or returns an empty answer, fall back to an extractive heuristic that pulls the most likely span from the retrieved text. Type-aware extraction is used for emails, dates, names, numbers, buildings, and locations.
6. Normalize and shorten the final answer so it matches the short-answer style expected by the grader.

The autograder substitutes its own `rag/llm.py` at evaluation time, so all logic that affects scoring lives in `rag/model.py`.

## Project Goals

- Build a retrieval-augmented short-answer system that performs well on the visible CS 288 validation set.
- Avoid overfitting to that visible set by constructing additional local holdout sets that stress legacy pages, PDF-backed pages, and underrepresented page types.
- Decompose end-to-end accuracy into retrieval recall, reader accuracy given retrieved evidence, and fallback-only accuracy.
- Provide a reproducible research harness (`research/research_metrics.py`) for tracking these metrics across changes.
- Reduce dependence on the local LLM so that evaluation under a different graded LLM does not collapse.

## Repository Layout

```
.
├── rag/
│   ├── model.py              # main retrieval and QA pipeline (autograder entrypoint)
│   ├── llm.py                # LLM call helper (replaced by the autograder)
│   └── eval_predictions.py   # exact-match and F1 scoring
├── preprocessing/            # crawler, HTML cleaning, IAA subset construction
├── research/                 # metric harness for retrieval / reader / fallback diagnostics
├── data/
│   ├── crawl_eecs.jsonl              # raw crawl used as the default corpus
│   ├── eecs_text_bs_rewritten.jsonl  # cleaned variant
│   ├── qa_validation_seed.jsonl      # visible validation set
│   ├── qa_holdout_mini*.jsonl        # additional local holdout sets
│   └── reference.jsonl               # reference annotations
├── questions_validation.txt          # visible validation questions
├── questions_holdout_mini*.txt       # local holdout question files
├── run.sh                            # autograder entrypoint
├── RAG_Model_REPORT.pdf              # project report
├── pyproject.toml
└── README.md
```

## How to Run

Install dependencies with [uv](https://docs.astral.sh/uv/) (Python 3.10+):

```bash
uv sync
```

Run the model on the validation questions:

```bash
uv run python rag/model.py questions_validation.txt predictions.txt
```

Score the predictions:

```bash
uv run python rag/eval_predictions.py \
  --references data/qa_validation_seed.jsonl \
  --questions questions_validation.txt \
  --predictions predictions.txt
```

Run via the autograder entrypoint:

```bash
./run.sh questions_validation.txt predictions.txt
```

Useful environment variables consumed by `run.sh`:

- `RAG_CORPUS_PATH` — override the default corpus file (defaults to `data/crawl_eecs.jsonl`).
- `RAG_NO_LLM=1` — disable the LLM and force extractive-only answers (used to measure fallback-only performance).

## Evaluation Methodology

Every change is evaluated against three classes of metric, not just final F1:

- **End-to-end:** exact match and F1 against the visible validation set and the local holdout sets.
- **Retrieval:** `URL Recall@k` and `answer_in_top_k` to measure whether the right evidence is even reaching the reader.
- **Conditional reader:** `F1 | gold URL retrieved` and `F1 | answer in top-k` to measure how well the model uses correct evidence.
- **Fallback-only:** F1 with the LLM disabled, to estimate robustness to a different graded LLM.

The interpretation rules used in this project are:

- If `URL Recall@k` is low, the bottleneck is retrieval.
- If `URL Recall@k` is high but F1 is low, the bottleneck is answer extraction or formatting.
- If full-model F1 is high but fallback-only F1 collapses, the pipeline is leaning too hard on the local LLM.

These are the same rules driving the prioritization in the project report.

## Findings

The diagnostic metrics surface several patterns that are not visible from end-to-end F1 alone:

- Retrieval, not reading, is the dominant source of error. Conditional F1 given the gold URL is materially higher than overall F1.
- The visible validation set is concentrated on a small number of URLs and a single domain, which means local F1 likely overestimates robustness on the hidden grader.
- Legacy `www2` pages, PDF-backed pages, and unusual page formats are a consistent weakness across all metrics.
- Fallback-only performance is much weaker than full-model performance, which means the system is currently over-reliant on the local LLM.

The full breakdown — including per-category numbers and the prioritized next steps that follow from them — lives in [`RAG_Model_REPORT.pdf`](./RAG_Model_REPORT.pdf).

## Why This Matters

A RAG system that scores well on a familiar validation slice can still fail badly on a held-out distribution if it is implicitly relying on a specific LLM or on lexical overlap with frequently-seen pages. By instrumenting retrieval quality, conditional reader quality, and fallback quality separately, this project tries to make those failure modes visible *before* the hidden evaluation rather than after.

## Notes

- The default corpus is the raw crawl rather than the cleaned variant; in our experiments it consistently retrieved better.
- The autograder replaces `rag/llm.py` at evaluation time. **Do not modify `rag/llm.py`.** All scoring-relevant logic must live in `rag/model.py`.
- The local holdout sets (`qa_holdout_mini*.jsonl`) were constructed to stress page types that are underrepresented in the visible validation set.

## Acknowledgments

This project was built for UC Berkeley CS 288. We thank the course staff for the assignment, the evaluation harness, and the EECS crawl that the corpus is derived from.
