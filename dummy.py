from langchain_ollama import OllamaEmbeddings

embedder = OllamaEmbeddings(model="nomic-embed-text:latest")
print(embedder.embed_query("Hello world"))
