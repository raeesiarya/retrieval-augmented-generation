#!/usr/bin/env bash
set -euo pipefail

QUESTIONS_PATH="$1"
PREDICTIONS_PATH="$2"
CORPUS_PATH="${RAG_CORPUS_PATH:-data/crawl_eecs.jsonl}"

if [ "${RAG_NO_LLM:-0}" = "1" ]; then
  python3 rag/model.py --no-llm --corpus "$CORPUS_PATH" "$QUESTIONS_PATH" "$PREDICTIONS_PATH"
else
  python3 rag/model.py --corpus "$CORPUS_PATH" "$QUESTIONS_PATH" "$PREDICTIONS_PATH"
fi
