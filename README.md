# ERICA — Graph RAG AI Tutor

NYU Artificial Intelligence, Fall 2025

ERICA is an AI tutor for the course that answers questions using a knowledge graph of AI/ML
concepts built from the course website. It retrieves the concepts most relevant to a question,
pulls in related and prerequisite concepts from the graph, and has a local LLM answer from
that context. It also draws the subgraph used to answer each question.

## How it works

1. **Scrape** — the course site is crawled and its pages saved as text
   (`erica_graph_rag_notebooks/01_scrape_data.ipynb`).
2. **Build the knowledge graph** — text is chunked; a local LLM (Ollama, `qwen2.5:7b`) extracts
   concepts with definitions, aliases and difficulty; concepts are embedded with
   `all-MiniLM-L6-v2`, linked by cosine similarity, and given LLM-inferred prerequisite edges
   (`erica_graph_rag_notebooks/02_build_graph.ipynb`).
3. **Answer** — `app1.py` (Flask) embeds the question, retrieves the top matching concept nodes,
   builds a prompt from them and their neighbours, queries the LLM, and renders the answer plus
   a picture of the query subgraph (`static/query_graph.png`).

`app.py` is an earlier version of the app that uses `full_code_new.py`, an alternative
end-to-end pipeline that adds a Chroma vector store alongside the graph (GraphRAG + RAG).

## Layout

| Path | Contents |
| --- | --- |
| `app1.py` | Flask app (main entry point) |
| `templates/am2_frontend.html` | Chat front end used by `app1.py` |
| `static/query_graph.png` | Subgraph image generated for the latest question |
| `erica_graph_rag_notebooks/` | Scraping + knowledge-graph notebooks; `knowledge_graph1.pkl` is the graph `app1.py` loads |
| `app.py`, `full_code_new.py`, `templates/latest_graphrag_front.html` | Earlier GraphRAG + Chroma version |
| `data/kg_course.json`, `data/ingested_urls.txt` | Knowledge graph and URL list used by the earlier version |
| `dockerfile`, `docker-compose.yml` | Container setup (app + ChromaDB) |
| `*_Outputs.pdf`, `Q1_OP1.jpg`, `QA1_Terminal_OP1.jpg` | Example answers and outputs |

## Running

1. Install and start [Ollama](https://ollama.com), then `ollama pull qwen2.5:7b`.
2. `pip install -r requirements.txt flask sentence-transformers ollama matplotlib`
   and install [nano-graphrag](https://github.com/gusye1234/nano-graphrag).
3. `python app1.py` and open http://localhost:5000 — or `docker compose up`.

Scraped course pages, intermediate chunks and the vector database are not included; the
scraping and pipeline scripts regenerate them. Scrapers that need a logged-in session read it
from a local `.env` file, which is never committed.
