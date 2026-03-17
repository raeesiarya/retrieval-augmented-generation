import json

from tqdm import tqdm


INPUT_PATH = "data/crawl_eecs_raw.jsonl"
OUTPUT_PATH = "data/crawl_eecs.jsonl"

# Add URL substrings here. If a row's URL contains any of these values,
# that row will be filtered out.
FILTER_URLS = [
    # "hkn.eecs.berkeley.edu",
    # "/category/news/resolved-incidents/",
    "https://hkn.eecs.berkeley.edu/about/cmembers/",
    "https://hkn.eecs.berkeley.edu/about/officers/1",
    "https://hkn.eecs.berkeley.edu/about/officers/2",
    "https://hkn.eecs.berkeley.edu/about/officers/3",
    "https://hkn.eecs.berkeley.edu/about/officers/4",
    "https://hkn.eecs.berkeley.edu/about/officers/5",
    "https://hkn.eecs.berkeley.edu/about/officers/6",
    "https://hkn.eecs.berkeley.edu/about/officers/7",
    "https://hkn.eecs.berkeley.edu/about/officers/8",
    "https://hkn.eecs.berkeley.edu/about/officers/9",
    "http://hkn.eecs.berkeley.edu/about/officers/1",
    "http://hkn.eecs.berkeley.edu/about/officers/2",
    "http://hkn.eecs.berkeley.edu/about/officers/3",
    "http://hkn.eecs.berkeley.edu/about/officers/4",
    "http://hkn.eecs.berkeley.edu/about/officers/5",
    "http://hkn.eecs.berkeley.edu/about/officers/6",
    "http://hkn.eecs.berkeley.edu/about/officers/7",
    "http://hkn.eecs.berkeley.edu/about/officers/8",
    "http://hkn.eecs.berkeley.edu/about/officers/9",
    "http://chisel.eecs.berkeley.edu",
    "https://chisel.eecs.berkeley.edu",
]


def should_filter(url: str, text: str) -> bool:
    if text == "":
        return True

    for filtered_url in FILTER_URLS:
        if filtered_url == "":
            continue
        if filtered_url in url:
            return True
    return False


def main() -> None:
    kept_rows = 0
    removed_rows = 0

    with (
        open(INPUT_PATH, "r", encoding="utf-8") as input_file,
        open(OUTPUT_PATH, "w", encoding="utf-8") as output_file,
    ):
        for line in tqdm(input_file, desc="Filtering rows"):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            url = row["url"]
            text = row.get("text", "")

            if should_filter(url, text):
                removed_rows += 1
                continue

            json.dump(row, output_file, ensure_ascii=False)
            output_file.write("\n")
            kept_rows += 1

    print(f"Wrote {kept_rows} rows to {OUTPUT_PATH}")
    print(f"Filtered out {removed_rows} rows")


if __name__ == "__main__":
    main()
