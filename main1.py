import subprocess
import sys

def install_package(package):
    """Install package using pip if not already installed."""
    try:
        __import__(package)
    except ImportError:
        print(f"[setup] Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure required packages
install_package("numpy")
install_package("scikit_learn")

from flask import Flask, request, jsonify, send_file
import pickle
import os
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings
import subprocess

app = Flask(__name__)

# -------------------- Load artifacts --------------------
ARTIFACT_DIR = "artifacts"
with open(os.path.join(ARTIFACT_DIR, "chunks.pkl"), "rb") as f:
    course_chunks = pickle.load(f)

with open(os.path.join(ARTIFACT_DIR, "graph.pkl"), "rb") as f:
    knowledge_graph = pickle.load(f)

embedder = OllamaEmbeddings(model="nomic-embed-text")

# -------------------- Utilities --------------------
def embed_text(text: str):
    return embedder.embed_query(text)

def ask_ollama(prompt: str):
    try:
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:0.5b-instruct"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.decode()}"
        return result.stdout.decode()
    except Exception as e:
        return f"Ollama error: {str(e)}"

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    return f"""
You are an AI tutor for an NYU AI course.

Use ONLY the following course material when answering.
If the answer is not found in the context, say: "I don’t see this in the course material."

### Context:
{context}

### Question:
{query}

### Answer:
"""

def query_graph_rag(query, G, chunks, top_k_nodes=5):
    '''query_emb = np.array(embed_text(query)).reshape(1, -1)
    node_texts = list(G.nodes)
    node_embs = np.array([embed_text(n) for n in node_texts])
    sims = cosine_similarity(query_emb, node_embs)[0]
    top_indices = sims.argsort()[-top_k_nodes:][::-1]
    top_nodes = [node_texts[i] for i in top_indices]
    relevant_chunks = [c for c in chunks if any(node in c for node in top_nodes)]
    prompt = build_prompt(query, relevant_chunks)
    answer = ask_ollama(prompt)
    return answer'''
    #def query_graph_rag(query, G, chunks, top_k_nodes=5):
    print(f"\n[INFO] Received query: {query}")

    # Embed the query
    query_emb = np.array(embed_text(query)).reshape(1, -1)

    # Embed all nodes in the graph
    node_texts = list(G.nodes)
    node_embs = np.array([embed_text(n) for n in node_texts])

    # Compute similarity
    sims = cosine_similarity(query_emb, node_embs)[0]

    # Pick top nodes
    top_indices = sims.argsort()[-top_k_nodes:][::-1]
    top_nodes = [node_texts[i] for i in top_indices]

    print(f"[INFO] Top nodes selected from graph: {top_nodes}")

    # Retrieve chunks that contain any of the top nodes
    relevant_chunks = [c for c in chunks if any(node in c for node in top_nodes)]
    print(f"[INFO] Number of chunks retrieved: {len(relevant_chunks)}")
    for i, c in enumerate(relevant_chunks):
        print(f"[INFO] Chunk {i+1}: {c[:200]}{'...' if len(c)>200 else ''}")  # print first 200 chars

    # Build prompt for LLM
    prompt = build_prompt(query, relevant_chunks)
    print(f"[INFO] Prompt sent to Ollama:\n{prompt}\n{'-'*60}")

    # Get answer
    answer = ask_ollama(prompt)
    print(f"[INFO] Answer from Ollama:\n{answer}\n{'-'*60}")
    return answer



# -------------------- Flask routes --------------------
@app.route("/")
def index():
    # Serve your frontend HTML
    return send_file("templates\index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    if not data.get("query"):
        return jsonify({"error": "query field missing"})

    mode = data.get("mode", "graph_rag")
    query = data["query"]

    if mode == "graph_rag":
        answer = query_graph_rag(query, knowledge_graph, course_chunks)

    elif mode == "simple_rag":
        query_emb = np.array(embed_text(query)).reshape(1, -1)
        chunk_embs = np.array([embed_text(c) for c in course_chunks])
        sims = cosine_similarity(query_emb, chunk_embs)[0]
        top_chunks = [course_chunks[i] for i in sims.argsort()[-5:][::-1]]

        print(f"[INFO] Top chunks for Simple RAG:")
        for i, c in enumerate(top_chunks):
            print(f"[INFO] Chunk {i+1}: {c[:200]}{'...' if len(c) > 200 else ''}")

        prompt = build_prompt(query, top_chunks)
        print(f"[INFO] Prompt sent to Ollama:\n{prompt}\n{'-'*60}'")

        answer = ask_ollama(prompt)
        print(f"[INFO] Answer from Ollama:\n{answer}\n{'='*80}")

    elif mode == "graph_deep":
        answer = query_graph_rag(query, knowledge_graph, course_chunks)

    else:
        answer = "Unknown mode"

    return jsonify({"answer": answer, "history": []})

'''def ask():
    data = request.json
    if not data.get("query"):
        return jsonify({"error": "query field missing"})
    
    mode = data.get("mode", "graph_rag")
    query = data["query"]

    if mode == "graph_rag":
        answer = query_graph_rag(query, knowledge_graph, course_chunks)
    elif mode == "simple_rag":
        # simple RAG: use top N chunks by embedding similarity
        query_emb = np.array(embed_text(query)).reshape(1, -1)
        chunk_embs = np.array([embed_text(c) for c in course_chunks])
        sims = cosine_similarity(query_emb, chunk_embs)[0]
        top_chunks = [course_chunks[i] for i in sims.argsort()[-5:][::-1]]
        prompt = build_prompt(query, top_chunks)
        answer = ask_ollama(prompt)
    elif mode == "graph_deep":
        # For now, same as graph_rag
        answer = query_graph_rag(query, knowledge_graph, course_chunks)
    else:
        answer = "Unknown mode"

    return jsonify({"answer": answer, "history": []})'''

@app.route("/graph_info")
def graph_info():
    total_nodes = knowledge_graph.number_of_nodes()
    total_edges = knowledge_graph.number_of_edges()
    # Simple type count
    node_types = {}
    for n in knowledge_graph.nodes:
        node_types[n] = node_types.get(n, 0) + 1
    return jsonify({
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "node_types": {"concept": total_nodes}  # simple placeholder
    })

@app.route("/concepts")
def concepts():
    # Return top 10 nodes as concepts
    top_nodes = list(knowledge_graph.nodes)[:10]
    concepts = [{"title": n, "definition": f"Definition for {n}", "connections": knowledge_graph.degree[n]} for n in top_nodes]
    return jsonify({"concepts": concepts})

@app.route("/modes")
def modes():
    return jsonify({
        "simple_rag": "Simple RAG",
        "graph_rag": "GraphRAG",
        "graph_deep": "GraphRAG (Deep)"
    })

# -------------------- Run Flask --------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
