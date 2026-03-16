#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash run.sh <questions_txt_path> <predictions_out_path>" >&2
  exit 1
fi

QUESTIONS_PATH="$1"
PREDICTIONS_PATH="$2"

if [ ! -f "$QUESTIONS_PATH" ]; then
  echo "Questions file not found: $QUESTIONS_PATH" >&2
  exit 1
fi

if [ -n "${RAG_CORPUS_PATH:-}" ]; then
  if [ "${RAG_NO_LLM:-0}" = "1" ]; then
    python3 rag/model.py --no-llm --corpus "$RAG_CORPUS_PATH" "$QUESTIONS_PATH" "$PREDICTIONS_PATH"
  else
    python3 rag/model.py --corpus "$RAG_CORPUS_PATH" "$QUESTIONS_PATH" "$PREDICTIONS_PATH"
  fi
  
else
  if [ "${RAG_NO_LLM:-0}" = "1" ]; then
    python3 rag/model.py --no-llm "$QUESTIONS_PATH" "$PREDICTIONS_PATH"
  else
    python3 rag/model.py "$QUESTIONS_PATH" "$PREDICTIONS_PATH"
  fi
fi
