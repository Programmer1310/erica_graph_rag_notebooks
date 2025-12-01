import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

BASE_URL = "https://pantelis.github.io/courses/ai/"
RAW_DIR = "data/raw/course_site/"
PAGE_DIR = os.path.join(RAW_DIR, "pages")
ASSET_DIR = os.path.join(RAW_DIR, "assets")

os.makedirs(PAGE_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)

visited = set()

def is_internal_link(url):
    parsed = urlparse(url)
    return "pantelis.github.io" in parsed.netloc and "/courses/ai/" in parsed.path

def save_file(url, content, subpath):
    filepath = os.path.join(subpath, os.path.basename(url))
    with open(filepath, "wb") as f:
        f.write(content)
    return filepath

def scrape_page(url):
    if url in visited:
        return []
    visited.add(url)

    print(f"[scrape] fetching {url}")

    response = requests.get(url)
    if response.status_code != 200:
        print(f"[scrape] failed: {url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    # Save HTML
    filename = url.replace(BASE_URL, "").strip("/").replace("/", "_") or "index"
    html_path = os.path.join(PAGE_DIR, f"{filename}.html")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    # Extract text
    text = soup.get_text(separator="\n", strip=True)
    text_path = os.path.join(PAGE_DIR, f"{filename}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Collect links
    new_links = []
    for a in soup.find_all("a", href=True):
        full_url = urljoin(url, a["href"])
        
        # Skip external links
        if not is_internal_link(full_url):
            continue
        
        if full_url not in visited:
            new_links.append(full_url)

    return new_links

def scrape_course_site():
    print("[scrape] starting course site scrape...")
    to_visit = [BASE_URL]

    while to_visit:
        url = to_visit.pop()
        new_links = scrape_page(url)
        to_visit.extend(new_links)

    print(f"[scrape] done. Total pages: {len(visited)}")


#if __name__ == "__main__":
scrape_course_site()
