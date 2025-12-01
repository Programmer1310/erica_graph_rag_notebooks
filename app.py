from flask import Flask, render_template, request, jsonify
import sys
import os

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import new unified backend
from full_code_new import load_graph, answer_query

app = Flask(__name__)

# -----------------------------------------
# INITIALIZE GRAPH + VECTOR DB AT STARTUP
# -----------------------------------------
print("🔧 Initializing GraphRAG backend...")

try:
    # Loads existing KG + vector DB
    load_graph(build_if_missing=False)
    print("✓ Knowledge Graph + Vector DB loaded")
except Exception as e:
    print(f"⚠ Could not load saved graph: {e}")
    print("→ Building from scratch (this may take time)...")
    load_graph(build_if_missing=True)
    print("✓ Knowledge Graph built")

# Chat memory
chat_history = []

@app.route("/")
def index():
    return render_template("latest_graphrag_front.html")

# -----------------------------------------
# MAIN QUESTION ENDPOINT
# -----------------------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_input = data.get("query", "")

    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    print(f"\n📩 New Query: {user_input}")

    try:
        answer = answer_query(user_input)   # <-- NEW unified call
    except Exception as e:
        print("Pipeline error:", e)
        answer = f"Sorry, something went wrong: {str(e)}"

    # Save to history
    chat_history.append({
        "user": user_input,
        "bot": answer
    })
    if len(chat_history) > 20:
        chat_history.pop(0)

    return jsonify({
        "answer": answer,
        "history": chat_history
    })

# -----------------------------------------
# OPTIONAL: GET CONCEPT LIST
# -----------------------------------------
@app.route("/concepts", methods=["GET"])
def concepts():
    from full_code_new import kg  # import here to avoid circular import

    if kg is None or len(kg.nodes()) == 0:
        return jsonify({"error": "Knowledge graph not built yet"}), 503

    concepts = []
    for node, data in kg.nodes(data=True):
        if data.get("type") == "concept":
            concepts.append({
                "id": node,
                "title": data.get("title", ""),
                "connections": len(kg[node]),
                "definition": data.get("definitions", [""])[0][:200] + "..."
            })

    concepts = sorted(concepts, key=lambda x: x["connections"], reverse=True)

    return jsonify({
        "total": len(concepts),
        "concepts": concepts[:50]
    })

# -----------------------------------------
# OPTIONAL: GRAPH STATS
# -----------------------------------------
@app.route("/graph_info", methods=["GET"])
def graph_info():
    from full_code_new import kg

    if kg is None:
        return jsonify({"error": "KG not ready"}), 503

    node_types = {}
    edge_types = {}

    for n, d in kg.nodes(data=True):
        node_types[d.get("type", "unknown")] = node_types.get(d.get("type", "unknown"), 0) + 1

    for u, v, d in kg.edges(data=True):
        edge_types[d.get("type", "unknown")] = edge_types.get(d.get("type", "unknown"), 0) + 1

    return jsonify({
        "nodes": kg.number_of_nodes(),
        "edges": kg.number_of_edges(),
        "node_types": node_types,
        "edge_types": edge_types
    })

# -----------------------------------------
# RUN SERVER
# -----------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
