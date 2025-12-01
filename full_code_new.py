"""
full_code.py - Cleaned end-to-end GraphRAG + RAG pipeline using Ollama (qwen2.5) + Chroma

Usage:
    # from your Flask frontend:
    from full_code import load_graph, answer_query
    load_graph(build_if_missing=False)   # ensures KG & collection are loaded (won't rebuild by default)
    answer = answer_query("What is a perceptron?")

    # From CLI to run whole pipeline (scrape, chunk, embed, build KG):
    python full_code.py --run-pipeline

Notes / Environment:
    - Ollama must be running locally (default http://127.0.0.1:11434).
      Pull model: `ollama pull qwen2.5:0.5b` or `ollama pull qwen2.5:1.5b` (0.5b is lighter).
    - Python deps (approx): requests, beautifulsoup4, markdownify, chromadb, networkx, nltk, tqdm
    - Make sure chromadb PersistentClient path is writable (DB_DIR).
    - This file will NOT run heavy tasks on import; call ensure_pipeline() or run CLI flag to build.
"""

import os
import time
import json
import glob
import math
import requests
import traceback
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm

# NLP
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

# Graph & DB
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import chromadb

# Default paths (change if you want)
BASE_URL = "https://pantelis.github.io/courses/ai/"
RAW_DIR = "data/raw/course_site"
PAGE_DIR = os.path.join(RAW_DIR, "pages")
ASSET_DIR = os.path.join(RAW_DIR, "assets")
TEXT_DIR = "data/processed/text"
CHUNK_DIR = "data/processed/chunks"
DB_DIR = "data/vector_db"
KG_JSON = "data/kg_course.json"
INGESTED_URLS = "data/ingested_urls.txt"

# Ollama settings
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")  # make sure model is pulled in ollama
ollama_session = requests.Session()

# Ensure directories exist
for d in (PAGE_DIR, ASSET_DIR, TEXT_DIR, CHUNK_DIR, DB_DIR, os.path.dirname(KG_JSON)):
    os.makedirs(d, exist_ok=True)

# --- Utilities & NLP helpers ---
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def cosine(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def extract_candidate_phrases(text):
    candidates = []
    for sent in sent_tokenize(text):
        tokens = word_tokenize(sent)
        tags = pos_tag(tokens)
        buf = []
        for word, tag in tags:
            if tag.startswith("NN") or tag.startswith("JJ") or tag == "NNP":
                buf.append(word)
            else:
                if len(buf) >= 1:
                    phrase = " ".join(buf).strip().lower()
                    phrase = phrase.strip(" ,.:;()[]\"'")
                    if len(phrase) > 1 and not phrase.isnumeric():
                        candidates.append(phrase)
                buf = []
        if buf:
            phrase = " ".join(buf).strip().lower()
            if len(phrase) > 1:
                candidates.append(phrase)
    return candidates

# --- Ollama HTTP wrappers (robust, retries) ---
def _ollama_post(path, json_payload, timeout=60, retries=4, backoff=2):
    url = f"{OLLAMA_BASE_URL.rstrip('/')}{path}"
    for attempt in range(1, retries + 1):
        try:
            r = ollama_session.post(url, json=json_payload, timeout=timeout)
            # Two possibilities: 200 is good, else print response and retry
            if r.status_code == 200:
                return r
            else:
                print(f"[ollama] HTTP {r.status_code} at {url}: {r.text[:200]}")
        except requests.exceptions.RequestException as e:
            print(f"[ollama] request error on {url}: {e}")
        if attempt < retries:
            time.sleep(backoff * attempt)
    raise RuntimeError(f"Ollama request failed after {retries} attempts: {url}")

def embed_ollama(text, retries=6):
    """
    Stable embedding call using Ollama's /api/embeddings with qwen2.5 model.
    Returns a list of floats (embedding) or raises on failure.
    """
    if not text:
        return []
    # truncate to keep Ollama happier on long texts
    if len(text) > 8000:
        text = text[:8000] + " [truncated]"

    payload = {
        "model": OLLAMA_MODEL,
        # Qwen expects "input" for embeddings
        "input": text
    }

    try:
        r = _ollama_post("/api/embeddings", payload, timeout=120, retries=retries)
        j = r.json()
        # possible shapes: {"embedding": [...]} or older responses
        emb = j.get("embedding") or j.get("embeddings") or j.get("data") or None
        # handle different structures
        if isinstance(emb, list) and all(isinstance(x, (int, float)) for x in emb):
            return emb
        # some responses put embeddings under data[0]["embedding"]
        if isinstance(j.get("data"), list) and j["data"] and isinstance(j["data"][0].get("embedding"), list):
            return j["data"][0]["embedding"]
        # fallback: try top-level keys
        if isinstance(j, dict) and "embedding" in j and isinstance(j["embedding"], list):
            return j["embedding"]
        # if we reach here, try to find numeric list anywhere
        def find_emb(obj):
            if isinstance(obj, list):
                if obj and isinstance(obj[0], (int, float)):
                    return obj
                for item in obj:
                    res = find_emb(item)
                    if res:
                        return res
            if isinstance(obj, dict):
                for v in obj.values():
                    res = find_emb(v)
                    if res:
                        return res
            return None
        emb = find_emb(j)
        if emb:
            return emb
        # not found
        print("[embed] embedding not found in response:", j)
        return []
    except Exception as e:
        print("[embed] failed:", e)
        return []

def ask_ollama(prompt, max_tokens=1024, retries=3):
    """
    Sends a prompt to Ollama /api/generate using the same model.
    Returns the string output (best-effort).
    """
    if not prompt:
        return ""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # you can add generation parameters if desired
        "max_tokens": max_tokens
    }

    try:
        r = _ollama_post("/api/generate", payload, timeout=180, retries=retries)
        j = r.json()
        # qwen style returns {'response': '...'} or {'content': '...'} or nested
        if isinstance(j, dict):
            # common keys
            for k in ("response", "content", "output", "text"):
                if k in j and isinstance(j[k], str):
                    return j[k]
            # sometimes 'choices' exist like OpenAI
            if "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                c = j["choices"][0]
                if isinstance(c, dict):
                    if "text" in c:
                        return c["text"]
                    if "message" in c and isinstance(c["message"], dict) and "content" in c["message"]:
                        return c["message"]["content"]
            # try to concatenate textual leaves
            def collect_text(obj):
                if isinstance(obj, str):
                    return obj
                if isinstance(obj, dict):
                    return " ".join(collect_text(v) for v in obj.values() if collect_text(v))
                if isinstance(obj, list):
                    return " ".join(collect_text(i) for i in obj if collect_text(i))
                return ""
            collected = collect_text(j)
            return collected.strip()
        return str(j)
    except Exception as e:
        print("[ask] failed:", e)
        return f"[ask error] {e}"

# --- Chroma collection helpers (persistent) ---
def get_chroma_collection(name="ai_course"):
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        col = client.get_collection(name)
    except Exception:
        # create if missing
        col = client.create_collection(name)
    return col

# --- Scraping / processing pipeline (kept simple) ---
def scrape_course_site(base_url=BASE_URL):
    """
    Crawl pantelis.github.io and save HTML and text pages into PAGE_DIR.
    Simple BFS crawl limited to domain.
    """
    visited = set()
    to_visit = [base_url.rstrip("/")]
    print("[scrape] starting course site scrape...")
    while to_visit:
        url = to_visit.pop(0)
        url = url.rstrip("/")
        if url in visited:
            continue
        visited.add(url)
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                print(f"[scrape] failed {url} status {r.status_code}")
                continue
        except Exception as e:
            print(f"[scrape] request error {url}: {e}")
            continue

        soup = BeautifulSoup(r.text, "html.parser")

        parsed = urlparse(url)
        path = parsed.path if parsed.path else "/"
        filename = path.strip("/").replace("/", "_") or "index"
        # keep extension if present
        if not (filename.endswith(".html") or filename.endswith(".txt") or filename.endswith(".pdf") or filename.endswith(".ipynb")):
            filename += ".html"
        cleanname = "".join([c for c in filename if c.isalnum() or c in "._-"])
        html_path = os.path.join(PAGE_DIR, cleanname)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(r.text)

        # Save plain text if HTML
        if html_path.endswith(".html"):
            text = soup.get_text(separator="\n", strip=True)
            txtpath = os.path.join(PAGE_DIR, cleanname[:-5] + ".txt")
            with open(txtpath, "w", encoding="utf-8") as f:
                f.write(text)

            # collect links
            for a in soup.find_all("a", href=True):
                href = a["href"].split("#")[0]
                if not href or href.startswith(("mailto:", "javascript:", "tel:")):
                    continue
                full = urljoin(url + "/", href).rstrip("/")
                if "pantelis.github.io" in urlparse(full).netloc and full not in visited:
                    to_visit.append(full)
        # small rate-limit
        time.sleep(0.01)

    # Save list
    pages = sorted(glob.glob(os.path.join(PAGE_DIR, "*.html")) + glob.glob(os.path.join(PAGE_DIR, "*.txt")))
    with open(INGESTED_URLS, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(p + "\n")
    print(f"[scrape] done. pages saved: {len(pages)}")
    return pages

def load_all_texts_from_pages():
    txt_files = sorted(glob.glob(os.path.join(PAGE_DIR, "**", "*.txt"), recursive=True))
    combined = []
    for path in txt_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                combined.append(f.read())
        except:
            continue
    all_text = "\n\n".join(combined)
    with open(os.path.join(TEXT_DIR, "full.txt"), "w", encoding="utf-8") as f:
        f.write(all_text)
    return all_text

def chunk_text(text, size=1200):
    """
    Very simple word-based chunker. Saves to CHUNK_DIR as chunk_#.txt
    """
    if not text:
        return []
    words = text.split()
    chunks = []
    cur = []
    for w in words:
        cur.append(w)
        if len(cur) >= size:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    # save
    for i, c in enumerate(chunks):
        with open(os.path.join(CHUNK_DIR, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(c)
    return chunks

def build_vector_db_from_chunks(batch_limit=None):
    """
    Reads files from CHUNK_DIR, obtains embeddings from Ollama and inserts into Chroma.
    Skips already-existing ids for idempotency.
    """
    col = get_chroma_collection()
    chunk_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "*.txt")))
    if batch_limit:
        chunk_files = chunk_files[:batch_limit]
    print(f"[db] embedding and inserting {len(chunk_files)} chunks (this may take a while)...")

    for i, path in enumerate(tqdm(chunk_files)):
        cid = f"chunk-{i}"
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()

        # get embedding
        emb = embed_ollama(txt)
        if not emb or not isinstance(emb, list) or len(emb) < 4:
            print(f"[db] ❌ Invalid embedding for chunk {i} (skipping)")
            continue

        try:
            # check if id exists already
            try:
                existing = col.get(ids=[cid], include=["ids"])
                if existing and existing.get("ids"):
                    # already present: skip or update — we skip
                    continue
            except Exception:
                # older chroma may raise; continue to add
                pass

            col.add(
                ids=[cid],
                embeddings=[emb],
                documents=[txt],
                metadatas=[{"source": os.path.basename(path)}]
            )
        except Exception as e:
            print(f"[db] embed/add failed for chunk {i}: {e}")

    print("[db] vector DB insertion complete.")

# --- GraphRAG Implementation (kept close to your original) ---
class GraphRAG:
    def __init__(self, collection):
        self.G = nx.DiGraph()
        self.collection = collection
        self._concept_embeddings = {}
        self._resource_embeddings = {}
        self._examples = {}

    def add_concept(self, concept_id, title, difficulty=None, aliases=None, definitions=None):
        self.G.add_node(concept_id, type="concept", title=title, difficulty=difficulty or "medium",
                        aliases=aliases or [], definitions=definitions or [])
        self._concept_embeddings[concept_id] = embed_ollama(title) or embed_ollama(title[:400])

    def add_resource(self, resource_id, doc_text, src_meta=None, rtype="web", span=None, timecodes=None):
        self.G.add_node(resource_id, type="resource", text=doc_text, metadata=src_meta or {}, rtype=rtype,
                        span=span or "", timecodes=timecodes or [])
        # try to get embedding from collection if the id exists
        try:
            items = self.collection.get(ids=[resource_id], include=["embeddings"])
            if items and items.get("embeddings"):
                emb = items["embeddings"][0]
            else:
                emb = embed_ollama(doc_text)
        except Exception:
            emb = embed_ollama(doc_text)
        self._resource_embeddings[resource_id] = emb

    def add_example(self, example_id, snippet, src_meta=None):
        self.G.add_node(example_id, type="example", snippet=snippet, metadata=src_meta or {})
        self._examples[example_id] = snippet

    def add_explains(self, resource_id, concept_id, score=0.0):
        self.G.add_edge(resource_id, concept_id, type="explains", score=score)

    def add_exemplifies(self, example_id, concept_id, score=0.0):
        self.G.add_edge(example_id, concept_id, type="exemplifies", score=score)

    def add_prereq(self, a, b, reason=""):
        self.G.add_edge(a, b, type="prereq_of", reason=reason)

    def add_near_transfer(self, a, b, score=0.0):
        self.G.add_edge(a, b, type="near_transfer", score=score)
        self.G.add_edge(b, a, type="near_transfer", score=score)

    def build_from_collection(self, max_concepts=40, candidate_top_k=200, concept_sim_threshold=0.68):
        print("[kg] loading chunks from collection")
        items = self.collection.get(include=["ids", "documents", "embeddings", "metadatas"])
        ids = items.get("ids", [])
        chunks = []
        for i, cid in enumerate(ids):
            chunks.append({
                "id": cid,
                "doc": items["documents"][i],
                "embedding": items["embeddings"][i],
                "metadata": items.get("metadatas", [{}])[i]
            })

        # candidate phrases
        phrase_counts = {}
        for c in chunks:
            cand = extract_candidate_phrases(c.get("doc",""))
            for p in cand:
                if len(p) < 3 or p.isnumeric():
                    continue
                phrase_counts[p] = phrase_counts.get(p, 0) + 1
        sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
        candidates = [p for p, _ in sorted_phrases[:candidate_top_k]]
        selected = candidates[:max_concepts]
        print(f"[kg] creating {len(selected)} concepts")
        for i, title in enumerate(selected):
            cid = f"concept_{i}"
            # ask LLM for concise definition
            prompt = f"Provide a one-sentence definition of the concept \"{title}\" in the context of an introductory AI course."
            definition = ask_ollama(prompt) or ""
            self.add_concept(cid, title=title, definitions=[definition])

        # create resource nodes from chunks
        for c in chunks:
            self.add_resource(c["id"], c["doc"], src_meta=c.get("metadata", {}), rtype="web")

        # link resources->concepts via cosine sim
        concept_ids = [n for n, d in self.G.nodes(data=True) if d.get("type") == "concept"]
        concept_embs = [self._concept_embeddings[cid] for cid in concept_ids]
        for rid, data in list(self.G.nodes(data=True)):
            if data.get("type") != "resource":
                continue
            res_emb = self._resource_embeddings.get(rid)
            if res_emb is None:
                continue
            for idx, ce in enumerate(concept_embs):
                sim = cosine(res_emb, ce)
                if sim >= concept_sim_threshold:
                    self.add_explains(rid, concept_ids[idx], score=sim)

        # naive example detection
        for rid, data in list(self.G.nodes(data=True)):
            if data.get("type") != "resource":
                continue
            doc = data.get("text", "").lower()
            if any(k in doc for k in ("example", "exercise", "demo", "proof")):
                ex_id = f"example_{rid}"
                snippet = doc[:800]
                self.add_example(ex_id, snippet, {"source": rid})
                ex_emb = embed_ollama(snippet)
                for cid in concept_ids:
                    sim = cosine(ex_emb, self._concept_embeddings[cid])
                    if sim >= concept_sim_threshold:
                        self.add_exemplifies(ex_id, cid, score=sim)

        # near transfer & prereq via neighbor similarity + LLM checks
        for i, cid in enumerate(concept_ids):
            emb_i = self._concept_embeddings[cid]
            sims = []
            for cj in concept_ids:
                if cid == cj:
                    continue
                sims.append((cj, cosine(emb_i, self._concept_embeddings[cj])))
            sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)[:8]
            for (cj, simval) in sims_sorted:
                if simval < 0.45:
                    continue
                if simval >= 0.70:
                    self.add_near_transfer(cid, cj, score=simval)
                if 0.45 <= simval < 0.90:
                    prompt = (f"In one short line, answer Yes or No: Is \"{self.G.nodes[cid]['title']}\" "
                              f"a prerequisite for \"{self.G.nodes[cj]['title']}\" in an introductory AI curriculum? If yes, "
                              f"add a short reason.")
                    resp = ask_ollama(prompt)
                    if resp and resp.lower().strip().startswith("yes"):
                        self.add_prereq(cid, cj, reason=resp.strip())

        print("[kg] finished building KG")

    def concepts_for_query(self, query, top_k=6):
        q_emb = embed_ollama(query)
        concept_nodes = [n for n, d in self.G.nodes(data=True) if d.get("type") == "concept"]
        sims = []
        for cid in concept_nodes:
            sims.append((cid, cosine(q_emb, self._concept_embeddings.get(cid, []))))
        return sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]

    def subgraph_for_concepts(self, concept_ids, prereq_depth=2, include_resources=True, include_examples=True):
        nodes_to_include = set()
        for cid in concept_ids:
            nodes_to_include.add(cid)
            stack = [cid]
            depth = 0
            while stack and depth < prereq_depth:
                newstack = []
                for node in stack:
                    for p in self.G.predecessors(node):
                        if self.G.edges[p, node].get("type") == "prereq_of":
                            nodes_to_include.add(p)
                            newstack.append(p)
                stack = newstack
                depth += 1
        # siblings via near_transfer
        for cid in list(nodes_to_include):
            for nbr in list(self.G.successors(cid)):
                if self.G.edges[cid, nbr].get("type") == "near_transfer":
                    nodes_to_include.add(nbr)
            for nbr in list(self.G.predecessors(cid)):
                if self.G.edges[nbr, cid].get("type") == "near_transfer":
                    nodes_to_include.add(nbr)
        # add resources/examples
        if include_resources or include_examples:
            for node in list(nodes_to_include):
                for u, v, ed in self.G.in_edges(node, data=True):
                    if ed.get("type") == "explains" and include_resources:
                        nodes_to_include.add(u)
                    if ed.get("type") == "exemplifies" and include_examples:
                        nodes_to_include.add(u)
        return self.G.subgraph(nodes_to_include).copy()

    def save_json(self, path=KG_JSON):
        data = json_graph.node_link_data(self.G)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("[kg] saved to", path)

    def load_json(self, path=KG_JSON):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.G = json_graph.node_link_graph(data)
        # rebuild embeddings best-effort
        self._concept_embeddings = {}
        self._resource_embeddings = {}
        for n, d in self.G.nodes(data=True):
            if d.get("type") == "concept":
                self._concept_embeddings[n] = embed_ollama(d.get("title", "")) or []
            elif d.get("type") == "resource":
                try:
                    items = self.collection.get(ids=[n], include=["embeddings"])
                    if items and items.get("embeddings"):
                        self._resource_embeddings[n] = items["embeddings"][0]
                    else:
                        self._resource_embeddings[n] = embed_ollama(d.get("text", "")) or []
                except Exception:
                    self._resource_embeddings[n] = embed_ollama(d.get("text", "")) or []
        print("[kg] loaded from", path)

# --- Higher-level orchestration & exposed functions ---

# Internal singletons (lazy loaded)
_collection_singleton = None
_kg_singleton = None

def load_collection():
    global _collection_singleton
    if _collection_singleton is None:
        _collection_singleton = get_chroma_collection()
    return _collection_singleton

def load_graph(build_if_missing=False, max_concepts=60):
    """
    Loads (and returns) GraphRAG instance and vector collection.
    If build_if_missing is True, it will:
        - scrape if pages missing
        - combine text & chunk if chunks missing
        - build vector DB if DB empty
        - build KG if kg json missing
    Return: (collection, kg)
    """
    global _collection_singleton, _kg_singleton

    col = load_collection()

    # if build_if_missing -> run pipeline parts
    if build_if_missing:
        # scrape if no pages
        txts = glob.glob(os.path.join(PAGE_DIR, "**", "*.txt"), recursive=True)
        if not txts:
            print("[pipeline] scraping (pages missing)")
            scrape_course_site()

        # combine & chunk if no chunks
        chunk_files = glob.glob(os.path.join(CHUNK_DIR, "*.txt"))
        if not chunk_files or len(chunk_files) < 10:
            print("[pipeline] combining & chunking text")
            all_text = load_all_texts_from_pages()
            chunk_text(all_text)

        # embed & build vector DB if empty
        try:
            items = col.get(include=["ids"])
            if not items.get("ids"):
                print("[pipeline] building vector DB (collection empty)")
                build_vector_db_from_chunks()
        except Exception:
            print("[pipeline] building vector DB (exception reading collection)")
            build_vector_db_from_chunks()

        # KG build if json missing
        if not os.path.exists(KG_JSON):
            print("[pipeline] building KG from collection")
            kg = GraphRAG(col)
            kg.build_from_collection(max_concepts=max_concepts)
            kg.save_json(KG_JSON)
    # load KG
    if _kg_singleton is None:
        kg = GraphRAG(col)
        if os.path.exists(KG_JSON):
            kg.load_json(KG_JSON)
        else:
            # optional: lazy lightweight KG (empty) so answer_query fails gracefully
            print("[kg] KG JSON missing, create with build_if_missing=True if you want to construct it now.")
        _kg_singleton = kg
    return col, _kg_singleton

def build_generation_prompt_for_subgraph(query, subgraph, ordered_prereq=True):
    """
    Build a scaffolded prompt for generation using the subgraph.
    """
    concept_nodes = [n for n, d in subgraph.nodes(data=True) if d.get("type") == "concept"]
    indeg = {n: subgraph.in_degree(n) for n in concept_nodes}
    ordered = sorted(concept_nodes, key=lambda x: indeg.get(x, 0))

    parts = [f"User query: {query}\n"]
    parts.append("Scaffold: concepts (simple -> complex):\n")
    for cid in ordered:
        node = subgraph.nodes[cid]
        parts.append(f"Concept: {node.get('title')}\nDefinition: {node.get('definitions')}\n")
        incoming = [(u, v, d) for u, v, d in subgraph.in_edges(cid, data=True) if d.get("type") == "explains"]
        if incoming:
            parts.append("References:\n")
            for u, v, d in incoming[:3]:
                rnode = subgraph.nodes[u]
                parts.append(f" - Resource: {rnode.get('metadata', {}).get('source','')} (span={rnode.get('span','')})\n")
    parts.append("\nAnswer using ONLY the above material, step-by-step, and cite references when relevant:\n")
    return "\n".join(parts)

def answer_query(query: str, top_k=6, prereq_depth=2):
    """
    Main function for external callers (frontend). Ensures collection & KG loaded,
    selects candidate concepts, builds subgraph and asks LLM to generate an answer.
    Returns a string (the LLM answer).
    """
    if not query or not query.strip():
        return "Please provide a question."

    col, kg = load_graph(build_if_missing=False)
    if kg is None or kg.G.number_of_nodes() == 0:
        return "Knowledge Graph not loaded. Run `load_graph(build_if_missing=True)` to create it."

    # find candidate concepts
    cand = kg.concepts_for_query(query, top_k=top_k)
    if not cand:
        return "I don't see this in the course material."

    concept_ids = [cid for cid, score in cand]
    sub = kg.subgraph_for_concepts(concept_ids, prereq_depth=prereq_depth)
    prompt = build_generation_prompt_for_subgraph(query, sub)
    answer = ask_ollama(prompt)
    return answer or "I don't see this in the course material."

# convenience wrapper to run the full pipeline (for CLI)
def ensure_pipeline(max_concepts=60, rebuild_vector_db=False):
    """
    Run the full pipeline: scrape -> combine -> chunk -> embed -> build KG
    Use with care (this takes time).
    """
    print("[ensure_pipeline] start")
    # Scrape if needed
    pages = glob.glob(os.path.join(PAGE_DIR, "**", "*.txt"), recursive=True)
    if not pages:
        print("[ensure_pipeline] scraping pages")
        scrape_course_site()
    # combine & chunk
    chunks = glob.glob(os.path.join(CHUNK_DIR, "*.txt"))
    if not chunks:
        print("[ensure_pipeline] combining text and chunking")
        all_text = load_all_texts_from_pages()
        chunk_text(all_text)
    # build db
    col = load_collection()
    try:
        ids = col.get(include=["ids"]).get("ids", [])
    except Exception:
        ids = []
    if rebuild_vector_db or not ids:
        print("[ensure_pipeline] building vector DB from chunks")
        build_vector_db_from_chunks()
    # build KG
    if not os.path.exists(KG_JSON):
        print("[ensure_pipeline] building KG")
        kg = GraphRAG(col)
        kg.build_from_collection(max_concepts=max_concepts)
        kg.save_json(KG_JSON)
    else:
        print("[ensure_pipeline] KG JSON already present:", KG_JSON)

# If run as script, allow --run-pipeline or --test-answer
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-pipeline", action="store_true", help="Run full ingestion -> vector DB -> KG pipeline")
    p.add_argument("--test-answer", type=str, default=None, help="Run a test query (after loading KG)")
    p.add_argument("--rebuild-db", action="store_true", help="Force rebuild of vector DB")
    args = p.parse_args()

    if args.run_pipeline:
        ensure_pipeline(rebuild_vector_db=args.rebuild_db)
        print("Pipeline complete. KG at:", KG_JSON)
    elif args.test_answer:
        # Ensure KG loaded
        load_graph(build_if_missing=False)
        ans = answer_query(args.test_answer)
        print("\n==== ANSWER ====\n")
        print(ans[:5000])
    else:
        print("Run with --run-pipeline to build everything or --test-answer 'your question' to test answering.")
