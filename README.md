# RAG Project

This project is a retrieval augmented generation system for the UC Berkeley EECS website.

## How it works

1. The model reads the crawl data from `data/`.
2. It splits pages into chunks and builds a BM25 retrieval index.
3. For each question, it retrieves the most relevant chunks.
4. It sends the question and retrieved context to the LLM.
5. If the LLM fails or times out, it uses an extractive fallback method to pull an answer from the retrieved text.
6. It cleans the final answer so it matches the short answer style used by the grader.

## What changed

- Improved retrieval by using chunk text, page titles, and URL words
- Fixed corpus loading so `rag/model.py` uses the correct data file
- Improved answer cleanup so outputs are shorter and closer to exact answers
- Improved the fallback for emails, dates, names, numbers, buildings, and locations
- Built extra unseen holdout sets to test the model on questions about pages it hasn't seen before

## Main files

- `rag/model.py`: main retrieval and QA pipeline
- `rag/llm.py`: LLM call helper
- `rag/eval_predictions.py`: scoring script for exact match and F1
- `run.sh`: script used by the autograder
- `data/qa_validation_seed.jsonl`: visible validation set
- `data/qa_holdout_mini.jsonl`, `data/qa_holdout_mini2.jsonl`, `data/qa_holdout_mini3.jsonl`: extra local holdout sets

## How to run

Run the model on the validation questions:

```bash
python3 rag/model.py questions_validation.txt predictions.txt
```

Score the predictions:

```bash
python3 rag/eval_predictions.py --references data/qa_validation_seed.jsonl --questions questions_validation.txt --predictions predictions.txt
```

## Notes

- The default corpus is currently the raw crawl because it worked better than the cleaned version in our tests.
- The autograder replaces `rag/llm.py`, so the main logic that matters most is in `rag/model.py`. DO NOT CHANGE `rag/llm.py`.
- We added holdout question sets so we can test on "unseen" pages.
