from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def get_urls(base_url: str = "https://eecs.berkeley.edu") -> list:
    """
    Crawl the EECS website and return all internal URLs.
    """

    visited = set()
    to_visit = [base_url]

    pbar = tqdm(desc="Crawling pages")

    while to_visit:
        url = to_visit.pop()

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
            href = link["href"]

            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)

            if "eecs.berkeley.edu" not in parsed.netloc:
                continue

            if full_url.endswith((".pdf", ".jpg", ".png", ".zip")):
                continue

            if full_url not in visited and full_url not in to_visit:
                to_visit.append(full_url)

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
