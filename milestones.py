# milestone.ipynb

import os
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import networkx as nx
import numpy as np
import pickle
from langchain_ollama import OllamaEmbeddings

# -------------------- Directories --------------------
PAGE_DIR = "pages"
TEXT_DIR = "text"
CHUNK_DIR = "chunks"
DB_DIR = "vector_db"
BASE_URL = "https://pantelis.github.io"

visited = set()
embedder = OllamaEmbeddings(model="nomic-embed-text")

# -------------------- Scraper --------------------
def is_internal_link(url):
    parsed = urlparse(url)
    return "pantelis.github.io" in parsed.netloc

def scrape_page(url):
    url = url.rstrip("/")
    if url in visited:
        return []
    visited.add(url)
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []
    except:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    parsed_url = urlparse(url)
    path = parsed_url.path.strip("/") or "index.html"
    filename = "".join(c for c in path.replace("/", "_") if c.isalnum() or c in "._-")
    if not filename.endswith(".html"):
        filename += ".html"
    html_path = os.path.join(PAGE_DIR, filename)
    text_path = os.path.join(PAGE_DIR, filename.replace(".html", ".txt"))
    os.makedirs(PAGE_DIR, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    try:
        text = soup.get_text(separator="\n", strip=True)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
    except:
        pass
    new_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0]
        if not href or href.lower().startswith(('mailto:', 'javascript:', 'tel:')):
            continue
        full_url = urljoin(url, href).rstrip("/")
        if is_internal_link(full_url) and full_url not in visited:
            new_links.append(full_url)
    return new_links

def scrape_course_site():
    os.makedirs(PAGE_DIR, exist_ok=True)
    to_visit = [BASE_URL]
    while to_visit:
        url = to_visit.pop()
        try:
            new_links = scrape_page(url)
            to_visit.extend(new_links)
        except:
            continue
    print(f"Scraped {len(visited)} pages")

# -------------------- Load Chunks --------------------
def load_chunks():
    chunks = []
    for fname in os.listdir(PAGE_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(PAGE_DIR, fname), "r", encoding="utf-8") as f:
                chunks.append(f.read())
    return chunks

# -------------------- Build Knowledge Graph --------------------
def ask_ollama(prompt: str):
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:0.5b-instruct"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60
        )
        return result.stdout.decode() if result.returncode == 0 else ""
    except:
        return ""

def build_knowledge_graph(chunks):
    G = nx.Graph()
    for chunk in chunks:
        prompt = f"""
Extract key concepts/entities and relationships from the following text.
Return as pairs in the format: Entity1 --relation--> Entity2

Text:
{chunk}

Pairs:
"""
        pairs_text = ask_ollama(prompt)
        lines = [line.strip() for line in pairs_text.splitlines() if line.strip()]
        for line in lines:
            try:
                if "--" in line:
                    parts = line.split("--")
                    node1 = parts[0].strip()
                    relation = parts[1].split(">")[0].strip() if ">" in parts[1] else ""
                    node2 = parts[1].split(">")[1].strip() if ">" in parts[1] else ""
                    G.add_node(node1)
                    G.add_node(node2)
                    G.add_edge(node1, node2, relation=relation)
            except:
                continue
    return G

# -------------------- Save Artifacts --------------------
def save_artifacts(chunks, graph):
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open("artifacts/graph.pkl", "wb") as f:
        pickle.dump(graph, f)

# -------------------- Run --------------------
scrape_course_site()
chunks = load_chunks()
graph = build_knowledge_graph(chunks)
save_artifacts(chunks, graph)
print("Artifacts saved to ./artifacts/")
