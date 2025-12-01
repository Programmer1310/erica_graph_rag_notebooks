import os
import time
import requests
from bs4 import BeautifulSoup
import chromadb
from langchain_ollama import OllamaEmbeddings

DB_DIR = "data/vector_db"
COLLECTION_NAME = "ai_course123"

# -------------------- Ollama Embeddings -------------------- #
embedder = OllamaEmbeddings(model="nomic-embed-text:latest")

def embed_text(text: str):
    return embedder.embed_query(text)

def embed_texts_in_batches(texts, batch_size=5):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_emb = [embed_text(t) for t in batch]
        embeddings.extend(batch_emb)
        time.sleep(0.1)
    return embeddings

# -------------------- Chroma DB -------------------- #
def get_chroma_client():
    return chromadb.PersistentClient(path=DB_DIR)

def load_collection():
    client = get_chroma_client()
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        return client.get_collection(COLLECTION_NAME)
    else:
        return client.create_collection(COLLECTION_NAME)

def build_vector_db_from_chunks(chunks: list[str]):
    collection = load_collection()
    embeddings = embed_texts_in_batches(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"Inserted {len(chunks)} chunks into vector DB.")

def query_vector_db(query: str, n_results=5):
    collection = load_collection()
    query_emb = embed_text(query)
    results = collection.query(query_embeddings=[query_emb], n_results=n_results)
    return results["documents"][0]

# -------------------- Web Scraping -------------------- #
def scrape_course_page(url: str):
    """Scrape a single course page and return text chunks."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts and styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Split into chunks of ~150 words
    chunks = []
    chunk_size = 150
    current_chunk = []
    word_count = 0
    for line in lines:
        words = line.split()
        word_count += len(words)
        current_chunk.append(line)
        if word_count >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# -------------------- Ollama LLM -------------------- #
def ask_ollama(prompt: str, model: str = "qwen2.5:0.5b-instruct"):
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.decode()}"
        return result.stdout.decode()
    except Exception as e:
        return f"Ollama error: {str(e)}"

def build_prompt(query: str, retrieved_chunks: list[str]):
    context = "\n\n".join(retrieved_chunks)
    return f"""
You are an AI tutor for an NYU AI course.

Use ONLY the following course material when answering.
If the answer is not found in the context, say:
"I don’t see this in the course material."

The name of the course is "Introduction to AI".

### Context:
{context}

### Question:
{query}

### Answer:
"""

def answer_query(query: str):
    chunks = query_vector_db(query)
    prompt = build_prompt(query, chunks)
    answer = ask_ollama(prompt)
    return answer

# -------------------- Main Pipeline -------------------- #
if __name__ == "__main__":
    # Example: scrape course page(s)
    course_urls = ["https://pantelis.github.io/courses/ai/"]
    all_chunks = []
    for url in course_urls:
        all_chunks.extend(scrape_course_page(url))
    
    # Build/update vector DB
    build_vector_db_from_chunks(all_chunks)
    
    # Query example
    query = "What is machine learning?"
    print("Question:", query)
    print("Answer:", answer_query(query))
