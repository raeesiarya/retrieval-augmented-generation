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
Clean the following webpage text so it can be used in a question answering system.

Remove:
- navigation menus
- repeated UI elements
- newsletter lists
- event listings
- footer text

Keep:
- factual information
- descriptions
- names
- dates
- numbers

Rewrite the content as clear paragraphs.

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
        chunks = chunk_text(doc["text"])
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
    urls = get_urls(limit=10)
    documents = process_urls(urls)

    cleaned_docs = clean_documents(documents)

    print("Pages scraped:", len(documents))

    for doc in cleaned_docs:
        print("\nURL:", doc["url"])
        print(doc["text"])
