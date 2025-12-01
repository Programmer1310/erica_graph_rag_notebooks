'''import subprocess
import sys

try:
    import sentence_transformers
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
'''


from flask import Flask, request, jsonify, render_template
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import Client
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import os


# INITIALIZE
app = Flask(__name__)
#G = nx.read_gpickle("knowledge_graph.gpickle")
import pickle
with open("erica_graph_rag_notebooks/knowledge_graph1.pkl", "rb") as f:
    print("opened knowledge graph!")
    G = pickle.load(f)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
ollama_client = Client(host="http://localhost:11434")
LLM_MODEL = "qwen2.5:7b"


# COSINE SIMILARITY
def cosine(a, b):
    #print("enter cosine")
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# FORMAT OUTPUT
import re
def format_llm_output(text):
    import re

    # Remove $$ and inline math delimiters
    text = re.sub(r'\$\$|\\\(|\\\)', '', text)

    # Inequalities
    text = text.replace(r'\leq', '<=').replace(r'\geq', '>=')
    text = text.replace(r'\lt', '<').replace(r'\gt', '>')

    # Replace \text{...}
    text = re.sub(r'\\text\{(.*?)\}', r'\1', text)

    # Convert \frac{a}{b} → (a)/(b)
    text = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1)/(\2)', text)

    # sqrt
    text = re.sub(r'\\sqrt\{(.*?)\}', r'sqrt(\1)', text)

    # Convert \mathbb{X} → X
    text = re.sub(r'\\mathbb\{(.*?)\}', r'\1', text)

    # Convert \mathcal{X} → X
    text = re.sub(r'\\mathcal\{(.*?)\}', r'\1', text)

    # Convert \mathbf{X} → X
    text = re.sub(r'\\mathbf\{(.*?)\}', r'\1', text)

    # Remove styling commands like \mathrm, \boldsymbol, etc.
    text = re.sub(r'\\(mathrm|boldsymbol|mathit|mathbf|mathsf)\{(.*?)\}', r'\2', text)

    # Subscripts and superscripts
    text = re.sub(r'_{(.*?)}', r'_\1', text)
    text = re.sub(r'\^{(.*?)}', r'^\1', text)

    # Dot operator
    text = text.replace(r'\cdot', '*')

    # Remove \left and \right
    text = text.replace(r'\left', '').replace(r'\right', '')

    # Remove *ALL remaining LaTeX commands* like \mathbbE, \alpha, \sum → alphabetic only kept
    text = re.sub(r'\\([A-Za-z]+)', r'\1', text)

    # Normalize spaces
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()





# RETRIEVE TOP-K KG NODES
def retrieve_relevant_nodes(query, top_k=5):
    print("retrieve relevant thingy")
    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    node_scores = []
    for n in G.nodes:
        if "embedding" in G.nodes[n] and G.nodes[n]["embedding"]:
            node_emb = np.array(G.nodes[n]["embedding"])
            sim = cosine(query_emb, node_emb)
            node_scores.append((n, sim))
    node_scores.sort(key=lambda x: x[1], reverse=True)
    return [n for n, s in node_scores[:top_k]]


# BUILD STRICT PROMPT
def build_prompt(query, nodes):
    context_text = ""
    print("built prompt!")
    for n in nodes:
        node_data = G.nodes[n]
        #context_text += f"Concept: {node_data['name']}\n"
        context_text += f"Concept: {n}\n" 
        if node_data.get("aliases"):
            context_text += f"Aliases: {', '.join(node_data['aliases'])}\n"
        if node_data.get("definition"):
            context_text += f"Definition: {node_data['definition']}\n"
        context_text += "\n"

    prompt = f"""

    You are Erica, the AI tutor for Prof. Pantelis Monogioudis's "Introduction to AI" course.

    RULES:
    1. First use the course material below if relevant
    2. If not enough in course material, add basic general knowledge
    3. Always show any mathematical formulas or notation that exist for the topic
    4. Keep explanations simple and clear
    5. When explaining a concept, FIRST explain its prerequisites and show their mathematical formulas
    6. Build up understanding step-by-step from foundational concepts to the current concept

    WHEN ANSWERING:
    - Start with: "Based on our course material"
    - If adding info: "In general AI"
    - Always show formulas: If topic has math, display it with $$like this$$
    - For prerequisites: Explain each prerequisite concept with its own mathematical notation before connecting them to the main concept

    COURSE MATERIAL:
    {context_text}

    QUESTION:
    {query} """
    
    print("Prompt Sent to Model:")
    print(prompt)
    print("============================")
    return prompt



def visualize_query_subgraph(nodes, filename="static/query_graph.png"):
    folder = os.path.dirname(filename)
    if folder != "" and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"Created missing folder: {folder}")

    subG = G.subgraph(nodes)

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(subG, seed=42)

    nx.draw_networkx_nodes(subG, pos, node_size=700, node_color="lightblue")
    nx.draw_networkx_edges(subG, pos, width=1.5, alpha=0.7)
    nx.draw_networkx_labels(
        subG, pos,
        labels={n: G.nodes[n].get("name", n) for n in subG.nodes},
        font_size=8
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved KG subgraph visualization at {filename}")


# ANSWER QUESTION
def answer_question(query):
    print("Inside answer def")
    nodes = retrieve_relevant_nodes(query)
    print("Retrieved KG Nodes:")
    print(nodes)
    print("==========================")
    

    visualize_query_subgraph(nodes)


    prompt = build_prompt(query, nodes)
    
    response = ollama_client.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    
    raw_answer = response.message.content
    clean_answer = format_llm_output(raw_answer) 
    return clean_answer



# ROUTES
@app.route("/")
def index():
    return render_template("am2_frontend.html") 

@app.route("/ask", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message.strip():
        return jsonify({"response": "Please enter a question."})
    answer = answer_question(user_message)
    return jsonify({"response": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
