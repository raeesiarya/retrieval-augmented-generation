from collections import deque
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def get_urls(base_url: str = "https://eecs.berkeley.edu") -> list:
    """
    Crawl the EECS website and return all internal URLs.
    """

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
            full_url = full_url.split("#")[0]

            parsed = urlparse(full_url)

            if "eecs.berkeley.edu" not in parsed.netloc:
                continue

            if full_url.endswith((".pdf", ".jpg", ".png", ".zip")):
                continue

            if full_url not in visited and full_url not in queued:
                to_visit.append(full_url)
                queued.add(full_url)

    pbar.close()

    return list(visited)


def open_page(page_url: str) -> str:
    """Open the url and return the text of the page."""
    return


def process_urls(urls: list):
    """Go through the list of urls and get the text of each page."""
    return


if __name__ == "__main__":
    urls = get_urls()
    print(urls)
