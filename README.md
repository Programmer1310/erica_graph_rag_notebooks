# ERICA Graph RAG Notebooks

NYU Artificial Intelligence, Fall 2025

Builds a knowledge graph of AI/ML concepts from the course website, for use in a graph-based
retrieval-augmented generation (Graph RAG) tutor.

## Pipeline

1. **`01_scrape_data.ipynb`** — crawls the course site (pantelis.github.io), saves the links to
   `raw_data/links.json`, and scrapes each page's text into `scraped_content.txt`.
2. **`02_build_graph.ipynb`** — chunks the scraped text, uses a local LLM (Ollama, `qwen2.5:7b`)
   to extract concepts (name, definition, aliases, difficulty), embeds them with
   `all-MiniLM-L6-v2`, links related concepts by cosine similarity, asks the LLM to infer
   prerequisite edges, and saves the resulting NetworkX graph.

## Outputs

| File | Contents |
| --- | --- |
| `knowledge_graph1.pkl` | Final knowledge graph (NetworkX `DiGraph`) |
| `knowledge_graph.pkl`, `knowledge_graph - Copy.pkl` | Earlier graph builds |
| `knowledge_graph.gpickle`, `dummy_knowledge_graph.gpickle` | Test graphs from the prototype section |

## Running

Requires a local [Ollama](https://ollama.com) server with `qwen2.5:7b`, plus
`requests beautifulsoup4 sentence-transformers networkx numpy ollama pandas`.
`scraped_content.txt` is not included; run `01_scrape_data.ipynb` to regenerate it.
