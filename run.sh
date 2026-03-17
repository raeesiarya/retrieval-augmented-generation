#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash run.sh <questions_txt_path> <predictions_out_path>" >&2
  exit 1
fi

QUESTIONS_PATH="$1"
PREDICTIONS_PATH="$2"
CORPUS_PATH="${RAG_CORPUS_PATH:-data/crawl_eecs.jsonl}"

if [ ! -f "$QUESTIONS_PATH" ]; then
  echo "Questions file not found: $QUESTIONS_PATH" >&2
  exit 1
fi

if [ ! -f "$CORPUS_PATH" ]; then
  echo "Corpus file not found: $CORPUS_PATH" >&2
  exit 1
fi

if [ "${RAG_NO_LLM:-0}" = "1" ]; then
  python3 rag/model.py --no-llm --corpus "$CORPUS_PATH" "$QUESTIONS_PATH" "$PREDICTIONS_PATH"
else
  python3 rag/model.py --corpus "$CORPUS_PATH" "$QUESTIONS_PATH" "$PREDICTIONS_PATH"
fi
