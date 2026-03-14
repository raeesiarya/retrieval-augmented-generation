from tqdm import tqdm
from crawl_eecs import get_urls, process_urls
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

PROMPT = """
You are cleaning scraped webpage text for a Retrieval-Augmented Generation (RAG) system.

Your goal is to extract ONLY meaningful informational content.

Remove completely:
- navigation menus
- breadcrumbs
- repeated text
- repeated links
- newsletter lists
- event listings
- page headers and footers
- link collections like "< Page > < Page >"
- duplicated phrases
- UI elements
- contact information blocks
- newsletter archives

Rules:
- Remove duplicate sentences.
- Remove repeated phrases.
- If a sentence appears many times, keep only one.
- Remove fragments that do not form meaningful sentences.

Keep ONLY:
- factual information
- descriptions
- biographies
- research descriptions
- names of people
- dates
- organizations
- locations

Rewrite the remaining content into clean paragraphs.

If the page contains no meaningful information, return an empty string.

TEXT:
"""


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200):
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def basic_cleanup(text: str) -> str:
    lines = text.splitlines()

    cleaned = []
    seen = set()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # remove duplicates
        if line in seen:
            continue

        seen.add(line)
        cleaned.append(line)

    return " ".join(cleaned)


def clean_text(text: str) -> str:
    prompt = PROMPT + text

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1] :]

    return tokenizer.decode(generated, skip_special_tokens=True)


def clean_documents(documents: list) -> list:

    cleaned_docs = []

    # count total chunks for accurate progress bar
    total_chunks = sum(len(chunk_text(doc["text"])) for doc in documents)

    pbar = tqdm(total=total_chunks, desc="LLM cleaning")

    for doc in documents:
        cleaned_input = basic_cleanup(doc["text"])
        chunks = chunk_text(cleaned_input)
        cleaned_chunks = []

        for chunk in chunks:
            cleaned = clean_text(chunk)
            cleaned_chunks.append(cleaned)

            pbar.update(1)

        combined = "\n".join(cleaned_chunks)

        cleaned_docs.append({"url": doc["url"], "text": combined})

    pbar.close()

    return cleaned_docs


if __name__ == "__main__":
    urls = get_urls(limit=3)
    documents = process_urls(urls)

    cleaned_docs = clean_documents(documents)

    print("Pages scraped:", len(documents))

    for doc in cleaned_docs:
        print("\nURL:", doc["url"])
        print(doc["text"])
