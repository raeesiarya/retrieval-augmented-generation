from transformers import pipeline
from tqdm import tqdm
from crawl_eecs import get_urls, process_urls

cleaner = pipeline("text-generation", model="google/flan-t5-base", max_new_tokens=512)

PROMPT = """
Clean the following webpage text so it is useful for a question answering system.

Remove:
- navigation menus
- repeated UI elements
- newsletter lists
- "View Open Faculty Positions"
- event listings

Keep:
- factual information
- descriptions
- names
- numbers
- dates

Text:
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
