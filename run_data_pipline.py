import os
import glob
from course_scraper import scrape_course_site
from zoom_scraper import download_zoom_recordings
from build_vector_db import build_vector_db

CHUNK_DIR = "data/processed/chunks"
TEXT_DIR = "data/processed/text"

os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

def load_all_text():
    txt_files = glob.glob("data/raw/course_site/pages/*.txt")
    combined = []

    for fpath in txt_files:
        with open(fpath, "r", encoding="utf-8") as f:
            combined.append(f.read())

    all_text = "\n\n".join(combined)
    
    with open(os.path.join(TEXT_DIR, "full.txt"), "w", encoding="utf-8") as f:
        f.write(all_text)

    return all_text

def chunk_text(text, size=1200):
    chunks = []
    words = text.split()
    current = []

    for w in words:
        current.append(w)
        if len(current) >= size:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    # save chunks
    for i, c in enumerate(chunks):
        with open(os.path.join(CHUNK_DIR, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(c)

    return chunks

def main():
    print("[pipeline] scraping course site")
    scrape_course_site()

    print("[pipeline] combining text")
    all_text = load_all_text()

    print("[pipeline] chunking text")
    chunk_text(all_text)

    print("[pipeline] building vector DB")
    build_vector_db()

    print("[pipeline] done")

#if __name__ == "__main__":
#    main()
