import os
import chromadb
from chromadb.config import Settings
from ollama import Client

TEXT_DIR = "data/processed/text"
CHUNK_DIR = "data/processed/chunks"
DB_DIR = "data/vector_db"

os.makedirs(DB_DIR, exist_ok=True)

client = Client()   # connects to http://localhost:11434

def embed_ollama(text):
    """Embed text using Ollama Python API (works on all versions)."""
    response = client.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]


def load_chunks():
    chunks = []
    for fname in os.listdir(CHUNK_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(CHUNK_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                chunks.append(f.read())
    return chunks


def build_vector_db():
    client_db = chromadb.PersistentClient(path=DB_DIR)
    collection = client_db.get_or_create_collection("ai_course")

    chunks = load_chunks()

    print(f"[db] embedding {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        emb = embed_ollama(chunk)

        collection.add(
            ids=[f"chunk-{i}"],
            embeddings=[emb],
            documents=[chunk]
        )

    print("[db] vector database built successfully.")


#if __name__ == "__main__":
build_vector_db()
