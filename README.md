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

# Research Notes

This file summarizes what the current metrics suggest about the system and what to do next.

It is based on the outputs from `research/research_metrics.py`, especially
`research/research_metrics_output.json`.

## Main Takeaways

### 1. Retrieval is still the main bottleneck

On the validation set:

- `F1 = 0.758`
- `URL Recall@4 = 0.73`
- `answer_in_top_k = 0.81`

But conditional performance is much higher:

- `F1 | gold URL retrieved = 0.898`
- `F1 | answer in top-k = 0.921`

Interpretation:

- The model is often fine once it sees the right evidence.
- The bigger problem is that it does not consistently retrieve the right page/chunk.

### 2. The benchmark is still too easy and too narrow

The visible validation set has:

- `100` questions
- only `33` unique URLs
- only `1` domain
- `55%` of questions coming from the top `10` URLs
- `38` questions from `people` pages alone

Interpretation:

- Local validation likely overestimates robustness.
- The current benchmark is concentrated on a familiar part of the site and does not stress retrieval enough.

### 3. Legacy / weird-format pages are a severe weakness

On `reference_legacy`:

- `F1 = 0.28`
- `URL Recall@4 = 0.0`
- `unknown_rate = 0.5`

Interpretation:

- The system is not handling legacy `www2` pages, PDFs, and odd page formats well.
- This is the clearest proxy for hidden autograder failure.

### 4. Person / email / location questions are weaker than they look

On validation:

- `person` questions: `URL Recall@4 = 0.57`
- `email` questions: `URL Recall@4 = 0.56`
- `location` questions: `URL Recall@4 = 0.61`
- `who` questions: `URL Recall@4 = 0.57`
- `where` questions: `URL Recall@4 = 0.50`
- `people` pages: `URL Recall@4 = 0.63`

Interpretation:

- These questions are not actually "solved."
- They look decent only when the right page is found.

### 5. The current system relies heavily on the LLM

Fallback-only performance is much worse:

- validation fallback-only `F1 = 0.292`
- holdout_mini2 fallback-only `F1 = 0.321`
- holdout_mini3 fallback-only `F1 = 0.259`
- reference_legacy fallback-only `F1 = 0.046`

Interpretation:

- The extractive fallback is not strong enough to carry the pipeline.
- If the autograder LLM behaves differently from our local one, score drop is expected.

## What This Means

The current hidden-set gap is most likely caused by:

1. retrieval miss rate on broader page distributions
2. weak handling of legacy / PDF / unusual pages
3. overreliance on the local LLM to clean up weak retrieval or weak extraction
4. local evaluation sets that are too concentrated to expose these problems

This does **not** look like a simple "the model answers slightly wrong" problem.

## What We Should Do Next

### Priority 1: Build harder evaluation sets

We should create new local evaluation sets that intentionally stress:

- legacy `www2` pages
- PDF-backed questions
- pages with only one question per URL
- more unseen URLs
- more diverse page types
- more paraphrased wording
- more multi-hop-ish phrasing where lexical overlap is weaker

Goal:

- make local retrieval recall look more like the hidden grader
- stop optimizing only for the current validation distribution

### Priority 2: Track retrieval metrics every time

When we evaluate changes, we should always track:

- overall `EM` / `F1`
- `URL Recall@k`
- `answer_in_top_k`
- `F1 | gold URL retrieved`
- `F1 | answer in top-k`
- fallback-only `EM` / `F1`

Goal:

- tell whether a change improved retrieval, answer extraction, or just local LLM behavior

### Priority 3: Improve retrieval coverage, not just answer cleanup

Given the metrics, the highest-value improvements are retrieval-side.

Focus areas:

- better indexing for legacy / `www2` / PDF-like content
- more robust handling of URL/domain variation
- better retrieval for person/location/contact pages
- reducing dependence on exact lexical overlap

Goal:

- increase `URL Recall@k` and `answer_in_top_k`, especially on diverse sets

### Priority 4: Treat fallback quality as a real weakness

The fallback should not be ignored just because the local LLM helps.

Focus areas:

- better extraction from retrieved spans
- stronger postprocessing for person/location/entity answers
- less brittle question-type assumptions

Goal:

- reduce the risk of catastrophic score drops when LLM behavior shifts

## Recommended Workflow

For each future change:

1. run `research/research_metrics.py`
2. compare validation, holdout, and legacy results together
3. look first at `URL Recall@k`
4. then check `answer_in_top_k`
5. only after that interpret the final `F1`

Rule of thumb:

- if `URL Recall@k` is low, fix retrieval
- if `URL Recall@k` is high but `F1` is low, fix answer extraction / formatting
- if full-model `F1` is high but fallback-only `F1` is awful, we are leaning too hard on the LLM

## Short Version

The system is strongest on familiar modern EECS pages with high lexical overlap.
It is weakest on:

- legacy / `www2` / PDF-like pages
- broader page diversity
- person / location / contact retrieval
- situations where the LLM cannot rescue weak retrieval or weak extraction

So the next step is not just "write harder questions."
The next step is:

- build harder and more diverse local benchmarks
- optimize retrieval against those benchmarks
- measure retrieval recall separately from end-to-end `F1`
