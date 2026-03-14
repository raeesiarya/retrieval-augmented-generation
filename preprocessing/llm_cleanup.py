from transformers import pipeline
from tqdm import tqdm
from crawl_eecs import get_urls, process_urls
import torch

cleaner = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    torch_dtype=torch.float16,
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


def clean_text(text: str) -> str:
    prompt = PROMPT + text

    output = cleaner(prompt)

    return output[0]["generated_text"]


def clean_documents(documents: list) -> list:

    cleaned_docs = []

    pbar = tqdm(total=len(documents), desc="LLM cleaning")

    for doc in documents:
        cleaned = clean_text(doc["text"])

        cleaned_docs.append({"url": doc["url"], "text": cleaned})

        pbar.update(1)

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
