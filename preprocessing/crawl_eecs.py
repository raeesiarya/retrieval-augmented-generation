from __future__ import annotations

from collections import deque
import json
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_URL = "https://eecs.berkeley.edu"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

ALLOWED_HOST_RE = re.compile(r"^(?:www\d*\.)?eecs\.berkeley\.edu$", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
TITLE_SUFFIX_RE = re.compile(
    r"\s*[|-]\s*(?:EECS(?: at Berkeley)?|Electrical Engineering and Computer Sciences.*"
    r"|University of California, Berkeley.*)$",
    re.IGNORECASE,
)

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
    ".docx",
    ".ppt",
    ".pptx",
    ".zip",
    ".bin",
    ".xml",
    ".atom",
)
BLOCKED_SCHEMES = ("mailto:", "tel:", "javascript:")
MAIN_CONTENT_SELECTORS = (
    "main",
    "[role='main']",
    "article",
    ".page-content",
    ".entry-content",
    ".node__content",
    ".layout-content",
    ".view-content",
    ".content",
)
NOISE_EXACT = {
    "directory",
    "home",
    "menu",
    "quick links",
    "search",
    "skip to main content",
    "staff contact quick list",
}
TAIL_NOISE_MARKERS = [
    " Categories ",
    " Tags ",
    " People Alumni ",
    " Academics Courses ",
    " Connect Support ",
    " About Diversity ",
    " Research Department Colloquium Series ",
    " Our Students Student Organizations ",
    " Our Staff Staff Awards ",
]
MIN_WORDS = 20


def normalize_space(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\xa0": " ",
        "\x00": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    if not path:
        return f"{scheme}://{netloc}"
    return f"{scheme}://{netloc}{path}"


def is_allowed_host(netloc: str) -> bool:
    return bool(ALLOWED_HOST_RE.fullmatch(netloc.split(":", 1)[0].lower()))


def should_skip_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return True
    if not is_allowed_host(parsed.netloc):
        return True

    lowered_path = parsed.path.lower()
    if lowered_path.endswith(BAD_EXTENSIONS):
        return True
    if "cgi-bin" in lowered_path or lowered_path.endswith(".cgi"):
        return True

    return False


def get_urls(base_url: str = BASE_URL, limit: int = 50000) -> list[str]:
    """Crawl reachable EECS HTML pages starting from the main site."""

    visited: set[str] = set()
    base_url = normalize_url(base_url)
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
            response = SESSION.get(url, timeout=10)
        except requests.RequestException:
            continue

        if response.status_code != 200:
            continue

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a", href=True):
            href = link["href"].strip()
            if not href or href.lower().startswith(BLOCKED_SCHEMES):
                continue

            full_url = urljoin(url, href)
            full_url = full_url.split("#", 1)[0]
            full_url = full_url.split("?", 1)[0]
            full_url = normalize_url(full_url)

            if should_skip_url(full_url):
                continue

            if full_url not in visited and full_url not in queued:
                to_visit.append(full_url)
                queued.add(full_url)

    pbar.close()
    return sorted(visited)


def clean_title_text(text: str) -> str:
    title = normalize_space(text)
    title = TITLE_SUFFIX_RE.sub("", title).strip(" -|")
    return normalize_space(title)


def fallback_title(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    if not slug or slug == "eecs.berkeley.edu":
        return "EECS"
    return normalize_space(re.sub(r"[-_]+", " ", slug)).title()


def extract_title(soup: BeautifulSoup, url: str) -> str:
    meta_title = soup.find("meta", attrs={"property": "og:title"})
    if meta_title and meta_title.get("content"):
        title = clean_title_text(meta_title["content"])
        if title:
            return title

    h1 = soup.find("h1")
    if h1:
        title = clean_title_text(h1.get_text(" ", strip=True))
        if title:
            return title

    if soup.title and soup.title.string:
        title = clean_title_text(soup.title.string)
        if title:
            return title

    return fallback_title(url)


def select_content_root(soup: BeautifulSoup):
    for selector in MAIN_CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node and node.get_text(" ", strip=True):
            return node
    return soup.body if soup.body else soup


def serialize_table(table) -> str:
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [
            normalize_space(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"])
        ]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def prune_noise_nodes(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(
        ["script", "style", "noscript", "svg", "iframe", "canvas", "form", "button"]
    ):
        tag.decompose()

    for tag in soup.find_all(True):
        attrs = " ".join(
            str(value)
            for key, value in tag.attrs.items()
            if key in {"class", "id", "role", "aria-label"}
        ).lower()
        if tag.name in {"nav", "footer"}:
            tag.decompose()
            continue
        if any(
            cue in attrs
            for cue in (
                "breadcrumb",
                "cookie",
                "menu",
                "pagination",
                "search",
                "share",
                "social",
                "subscribe",
            )
        ):
            tag.decompose()


def strip_tail_noise(text: str) -> str:
    cleaned = f" {text} "
    min_offset = 220
    for marker in TAIL_NOISE_MARKERS:
        index = cleaned.find(marker)
        if index >= min_offset:
            cleaned = cleaned[:index].rstrip()
    return normalize_space(cleaned)


def is_noise_line(line: str) -> bool:
    lowered = line.casefold()
    if lowered in NOISE_EXACT:
        return True
    if len(line.split()) <= 1 and not re.search(r"\d|@", line):
        return True
    return False


def dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for line in lines:
        key = normalize_space(line).casefold()
        if key in seen:
            continue
        seen.add(key)
        kept.append(normalize_space(line))
    return kept


def extract_clean_text(content) -> str:
    content_soup = BeautifulSoup(str(content), "html.parser")
    prune_noise_nodes(content_soup)

    # Tables are explicitly in scope for the assignment, so keep row structure readable.
    for table in content_soup.find_all("table"):
        table_text = serialize_table(table)
        if table_text:
            table.replace_with("\n" + table_text + "\n")
        else:
            table.decompose()

    lines = [
        normalize_space(line)
        for line in content_soup.get_text("\n").splitlines()
        if normalize_space(line)
    ]
    lines = [line for line in lines if not is_noise_line(line)]
    lines = dedupe_lines(lines)

    text = normalize_space(" ".join(lines))
    return strip_tail_noise(text)


def open_page(page_url: str) -> dict[str, str] | None:
    """Open a page and return retrieval-friendly text plus metadata."""

    try:
        response = SESSION.get(page_url, timeout=10)
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    title = extract_title(soup, page_url)
    content = select_content_root(soup)
    text = extract_clean_text(content)

    if len(text.split()) < MIN_WORDS:
        return None

    return {"url": page_url, "title": title, "text": text}


def process_urls(urls: list[str]) -> list[dict[str, str]]:
    """Fetch and clean the text for each crawled URL."""

    documents: list[dict[str, str]] = []
    pbar = tqdm(total=len(urls), desc="Processing pages")

    for url in urls:
        document = open_page(url)
        pbar.update(1)

        if not document:
            continue

        documents.append(document)

    pbar.close()
    return documents


def save_documents(documents: list[dict[str, str]], output_file_path: str) -> None:
    """Save the crawled corpus to a JSONL file."""

    with open(output_file_path, "w", encoding="utf-8") as handle:
        for doc in documents:
            json.dump(doc, handle, ensure_ascii=False)
            handle.write("\n")


if __name__ == "__main__":
    urls = get_urls(limit=50000)
    documents = process_urls(urls)
    print("Pages scraped:", len(documents))
    save_documents(documents, "data/crawl_eecs_raw.jsonl")
