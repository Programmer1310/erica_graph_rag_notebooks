import chromadb
from langchain_ollama import OllamaEmbeddings

DB_DIR = "data/vector_db"

# Create Ollama embedder 
embedder = OllamaEmbeddings(model="nomic-embed-text")

def embed_query(text: str):
    #Embed a single query using the same embedding model used for the DB
    return embedder.embed_query(text)  # returns a list of floats

def load_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection("ai_course")

def query_vector_db(query: str, n_results=5):
    #Query the vector DB using embeddings from Ollama
    collection = load_collection()
    query_emb = embed_query(query)  # 768-dim embedding
    results = collection.query(
        query_embeddings=[query_emb],  # pass embedding, not raw text
        n_results=n_results
    )
    return results["documents"][0]

'''def ask_ollama(prompt: str):
    import subprocess
    result = subprocess.run(
        ["ollama", "run", "llama3.1"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode()'''
def ask_ollama(prompt: str):
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:0.5b-instruct"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60  # Add timeout
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
"I don’t see this in the course material.

The name of the course is "Introduction to AI".
"

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
