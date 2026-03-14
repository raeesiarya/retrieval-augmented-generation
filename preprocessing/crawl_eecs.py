from collections import deque
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})


def get_urls(base_url: str = "https://eecs.berkeley.edu", limit: int = 50000) -> list:
    """
    Crawl the EECS website and return all internal URLs.
    """

    BAD_EXTENSIONS = (
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".mp4",
        ".ps",
        ".ps.gz",
        ".doc",
        ".zip",
        ".bin",
        ".xml",
        ".atom",
    )

    HEADERS = {"User-Agent": "Mozilla/5.0"}

    visited = set()
    queued = {base_url}
    to_visit = deque([base_url])

    pbar = tqdm(desc="Crawling pages")

    while to_visit and len(visited) < limit:
        url = to_visit.popleft()
        queued.remove(url)

        if url in visited:
            continue

        visited.add(url)

        pbar.update(1)
        pbar.set_postfix(queue=len(to_visit), visited=len(visited))

        try:
            response = SESSION.get(url, headers=HEADERS, timeout=5)
        except requests.RequestException:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a", href=True):
            full_url = urljoin(url, link["href"])

            parsed = urlparse(full_url)

            if parsed.path == "":
                full_url = parsed.scheme + "://" + parsed.netloc

            # normalize URL
            full_url = full_url.split("#")[0]
            full_url = full_url.split("?")[0]
            full_url = full_url.rstrip("/")

            parsed = urlparse(full_url)

            # only crawl the main EECS site
            if parsed.netloc not in {"eecs.berkeley.edu", "www.eecs.berkeley.edu"}:
                continue

            # skip mail links
            if full_url.startswith("mailto:"):
                continue

            # skip downloads / non-html
            if full_url.lower().endswith(BAD_EXTENSIONS):
                continue

            # skip CGI endpoints
            if "cgi-bin" in full_url or ".cgi" in full_url:
                continue

            if full_url not in visited and full_url not in queued:
                to_visit.append(full_url)
                queued.add(full_url)

    pbar.close()

    return list(visited)


def remove_repeated_lines(text: str) -> str:
    sentences = re.split(r"(?<=[.!?]) +", text)

    seen = set()
    filtered = []

    for s in sentences:
        s = s.strip()
        if not s or s in seen:
            continue

        seen.add(s)
        filtered.append(s)

    return " ".join(filtered)


def open_page(page_url: str) -> str:
    """Open the URL and return clean text from the page."""

    try:
        response = SESSION.get(page_url, timeout=5)
    except Exception:
        return ""

    if response.status_code != 200:
        return ""

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(
        ["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]
    ):
        tag.decompose()

    # try to extract main content
    content = soup.select_one(
        "main article, main .content, main .page-content, article"
    )

    if not content:
        content = soup.body if soup.body else soup

    text = content.get_text(separator=" ")
    text = " ".join(text.split())
    text = remove_repeated_lines(text)

    return text


def process_urls(urls: list) -> list:
    """Go through the list of urls and get the text of each page."""

    documents = []

    pbar = tqdm(total=len(urls), desc="Processing pages")

    for url in urls:
        text = open_page(url)

        pbar.update(1)

        if not text:
            continue

        documents.append({"url": url, "text": text})

    pbar.close()

    return documents


def save_documents(documents: list, output_file_path: str) -> None:
    """Save the documents to a JSONL file."""

    with open(output_file_path, "w", encoding="utf-8") as f:
        for doc in documents:
            json.dump(doc, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    urls = get_urls(limit=1000)
    documents = process_urls(urls)

    print("Pages scraped:", len(documents))

    for doc in documents:
        print("\nURL:", doc["url"])
        print(doc["text"])

    save_documents(documents, "data/crawl_eecs_raw.jsonl")
