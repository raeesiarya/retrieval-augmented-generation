from collections import deque
from tqdm import tqdm
import urllib.request
import urllib.error
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

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

ALLOWED_DOMAINS = {
    "eecs.berkeley.edu",
    "www.eecs.berkeley.edu",
}


def fetch_url(url: str) -> str | None:
    """Fetch HTML content from a URL."""

    req = urllib.request.Request(url, headers=HEADERS)

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "")

            if "text/html" not in content_type:
                return None

            html = response.read().decode("utf-8", errors="ignore")
            return html

    except Exception:
        return None


def get_urls(base_url: str = "https://eecs.berkeley.edu", limit: int = 50000) -> list:
    """Crawl the EECS website and collect internal URLs."""

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

        html = fetch_url(url)

        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")

        for link in soup.find_all("a", href=True):
            full_url = urljoin(url, link["href"])

            parsed = urlparse(full_url)

            # normalize URL
            full_url = full_url.split("#")[0]
            full_url = full_url.split("?")[0]
            full_url = full_url.rstrip("/")

            parsed = urlparse(full_url)

            # domain filter
            if parsed.netloc not in ALLOWED_DOMAINS:
                continue

            # skip email links
            if full_url.startswith("mailto:"):
                continue

            # skip file downloads
            if full_url.lower().endswith(BAD_EXTENSIONS):
                continue

            # skip CGI endpoints
            if "cgi-bin" in full_url or ".cgi" in full_url:
                continue

            # skip faculty home directories
            if "/~" in full_url:
                continue

            # skip wordpress pagination
            if re.search(r"/page/\d+", full_url):
                continue

            if full_url not in visited and full_url not in queued:
                to_visit.append(full_url)
                queued.add(full_url)

    pbar.close()

    return list(visited)


def remove_repeated_lines(text: str) -> str:
    """Remove duplicate sentences."""

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
    """Extract clean text from a webpage."""

    html = fetch_url(page_url)

    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # remove junk elements
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]
    ):
        tag.decompose()

    # try to locate main content
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
    """Extract text from all crawled URLs."""

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
    """Save documents to JSONL."""

    with open(output_file_path, "w", encoding="utf-8") as f:
        for doc in documents:
            json.dump(doc, f, ensure_ascii=False)

            f.write("\n")


if __name__ == "__main__":
    urls = get_urls(limit=50000)

    documents = process_urls(urls)

    print("Pages scraped:", len(documents))

    for doc in documents[:10]:
        print("\nURL:", doc["url"])
        print(doc["text"][:500])

    save_documents(documents, "data/crawl_eecs_raw.jsonl")
