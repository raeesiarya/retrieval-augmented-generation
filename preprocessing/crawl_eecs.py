from collections import deque
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def get_urls(base_url: str = "https://eecs.berkeley.edu") -> list:
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

    visited = set()
    queued = {base_url}
    to_visit = deque([base_url])

    pbar = tqdm(desc="Crawling pages")

    while to_visit:
        url = to_visit.pop()
        queued.remove(url)

        if url in visited:
            continue

        visited.add(url)

        pbar.update(1)
        pbar.set_postfix(queue=len(to_visit), visited=len(visited))

        try:
            response = requests.get(url, timeout=5)
        except Exception:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a", href=True):
            full_url = urljoin(base_url, link["href"])

            # normalize URL
            full_url = full_url.split("#")[0]
            full_url = full_url.split("?")[0]
            full_url = full_url.rstrip("/")

            parsed = urlparse(full_url)

            # stay inside EECS domain
            if "eecs.berkeley.edu" not in parsed.netloc:
                continue

            # skip wiki editing pages
            if "wiki.eecs.berkeley.edu" in parsed.netloc:
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


def open_page(page_url: str) -> str:
    """Open the URL and return clean text from the page."""

    try:
        response = requests.get(page_url, timeout=5)
    except Exception:
        return ""

    if response.status_code != 200:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    # remove junk elements
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    # extract visible text
    text = soup.get_text(separator=" ")

    # normalize whitespace
    text = " ".join(text.split())

    return text


def process_urls(urls: list):
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


if __name__ == "__main__":
    urls = get_urls()
    documents = process_urls(urls)
    print(documents)
