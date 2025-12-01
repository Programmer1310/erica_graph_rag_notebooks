

'''import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


from nltk.tokenize import sent_tokenize'''

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from markdownify import markdownify
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

# Ollama imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings

import chromadb
from chromadb.utils import embedding_functions
from chromadb import PersistentClient

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# ------------------------
# Config
# ------------------------
COURSE_URL = "https://pantelis.github.io/courses/ai/"
OLLAMA_MODEL = "qwen2.5"
VECTOR_DB_PATH = "vector_db"

WEBSITE_FOLDER = "website_data"
PROCESSED_TEXT_FOLDER = "processed_text"
ZOOM_FOLDER = "zoom_recordings"
TRANSCRIPTS_FOLDER = "zoom_transcripts"

os.makedirs(WEBSITE_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_TEXT_FOLDER, exist_ok=True)
os.makedirs(ZOOM_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)

nltk.download("punkt_tab", quiet=True)

from langchain_ollama import OllamaEmbeddings

#embedder = OllamaEmbeddings(model="qwen2.5") #used during

from types import SimpleNamespace
import ollama

EMBED_MODEL = "nomic-embed-text"   # or nomic-embed depending on your system

def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts with Ollama."""
    embeddings = []
    for t in texts:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        embeddings.append(resp["embeddings"])
    return embeddings

def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embeddings"]

# ➤ Wrap into a SimpleNamespace (Chroma requires obj with these 2 methods)
embedding_fn = SimpleNamespace(
    embed_documents=embed_documents,
    embed_query=embed_query
)


# Minimal class wrapper — only 3 lines
'''class OllamaEmbed:
    def __call__(self, input: list[str]) -> list[list[float]]:
        return [embedder.embed_query(t) for t in input]

ef = OllamaEmbed()

from types import SimpleNamespace

OLLAMA_MODEL = "nomic-embed-text"   # MUST be an embedding model

embedder = OllamaEmbeddings(model=OLLAMA_MODEL)

def _embed_batch(texts: list[str]) -> list[list[float]]:
    return [embedder.embed_query(t) for t in texts]

OllamaEmbed = SimpleNamespace(
    embed_documents=_embed_batch,
    embed_query=_embed_batch
)'''


'''from langchain_ollama import OllamaEmbeddings

embedder = OllamaEmbeddings(model="qwen2.5")

# Chroma 0.4+ expects an object with .embed_query method
class OllamaEmbed:
    def embed_query(self, input: list[str]) -> list[list[float]]:
        # input is a list of strings
        return [embedder.embed_query(t) for t in input]

ef = OllamaEmbed()'''

# ------------------------
# Step 1: Scrape website
# ------------------------
def scrape_website(base_url):
    visited = set()
    to_visit = [base_url]
    os.makedirs("website_data", exist_ok=True)

    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)
        try:
            r = requests.get(url)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")

            filename = url.replace(base_url, "").replace("/", "_") or "index"
            with open(f"website_data/{filename}.html", "w", encoding="utf-8") as f:
                f.write(r.text)

            for link in soup.find_all("a"):
                href = link.get("href")
                if href and href.endswith(".html"):
                    new_url = urljoin(url, href)
                    if base_url in new_url:
                        to_visit.append(new_url)
        except Exception as e:
            print("Error scraping:", e)

# ------------------------
# Step 2: Convert HTML to text
# ------------------------
def html_to_text():
    os.makedirs("processed_text", exist_ok=True)
    for file in os.listdir("website_data"):
        if file.endswith(".html"):
            html = open(os.path.join("website_data", file), encoding="utf-8").read()
            text = markdownify(html)
            out_file = os.path.join("processed_text", file.replace(".html", ".txt"))
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(text)

# ------------------------
# Step 3: Chunk text
# ------------------------
'''def chunk_text(chunk_size=500):
    all_chunks = []
    for file in os.listdir("processed_text"):
        if file.endswith(".txt"):
            text = open(os.path.join("processed_text", file), encoding="utf-8").read()
            sentences = sent_tokenize(text)
            chunks = []
            temp = ""
            for s in sentences:
                if len(temp.split()) + len(s.split()) > chunk_size:
                    chunks.append(temp.strip())
                    temp = s
                else:
                    temp += " " + s
            if temp:
                chunks.append(temp.strip())
            all_chunks.extend(chunks)
    return all_chunks'''

def chunk_text(chunk_size=500):
    all_chunks = []

    # Website chunks
    for file in os.listdir(PROCESSED_TEXT_FOLDER):
        if file.endswith(".txt"):
            text = open(os.path.join(PROCESSED_TEXT_FOLDER, file), encoding="utf-8").read()
            sentences = sent_tokenize(text)
            temp = ""
            for s in sentences:
                if len(temp.split()) + len(s.split()) > chunk_size:
                    all_chunks.append(temp.strip())
                    temp = s
                else:
                    temp += " " + s
            if temp:
                all_chunks.append(temp.strip())

    # Zoom transcripts chunks
    for file in os.listdir(TRANSCRIPTS_FOLDER):
        if file.endswith(".txt"):
            text = open(os.path.join(TRANSCRIPTS_FOLDER, file), encoding="utf-8").read()
            sentences = sent_tokenize(text)
            temp = ""
            for s in sentences:
                if len(temp.split()) + len(s.split()) > chunk_size:
                    all_chunks.append(temp.strip())
                    temp = s
                else:
                    temp += " " + s
            if temp:
                all_chunks.append(temp.strip())

    return all_chunks

# ------------------------
# Step 4: Wrap Ollama embeddings for Chroma
# ------------------------
#embedder = OllamaEmbeddings(model=OLLAMA_MODEL)

'''def embed_texts(texts):
    # texts: list of strings
    embeddings = []
    for t in texts:
        vec = embedder.embed_query(t)
        embeddings.append(vec)
    return embeddings'''

# ------------------------
# Step 5: Build local vector DB
# ------------------------

def download_zoom_recordings(zoom_links):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # remove for visible browser
    chrome_options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    for url in zoom_links:
        driver.get(url)
        time.sleep(5)
        input("Complete SSO login in browser, then press Enter here...")

        try:
            download_btn = driver.find_element(By.XPATH, "//a[contains(text(),'Download')]")
            download_url = download_btn.get_attribute("href")
            filename = download_url.split("/")[-1]
            r = requests.get(download_url)
            with open(os.path.join(ZOOM_FOLDER, filename), "wb") as f:
                f.write(r.content)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    driver.quit()

# ------------------------
# Step 6: Transcribe Zoom recordings using Whisper
# ------------------------
from faster_whisper import WhisperModel

whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

def transcribe_zoom_recordings():
    for file in os.listdir(ZOOM_FOLDER):
        if file.endswith(".mp4"):
            audio_path = os.path.join(ZOOM_FOLDER, file)
            transcript_path = os.path.join(TRANSCRIPTS_FOLDER, file.replace(".mp4", ".txt"))

            if os.path.exists(transcript_path):
                continue

            print(f"Transcribing {file} ...")

            segments, info = whisper_model.transcribe(audio_path)

            with open(transcript_path, "w", encoding="utf-8") as f:
                for segment in segments:
                    f.write(segment.text + "\n")

            print(f"Transcribed {file}")


# ------------------------
# Step 7: Build local vector DB
# ------------------------
def build_vector_db_local(chunks):
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    #client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    #import chromadb

    chroma_client = chromadb.PersistentClient(path="vector_db")

    collection = chroma_client.get_or_create_collection(
    name="ai_project_full",
    embedding_function=embedding_fn)


    # Check for existing collection
    '''try:
        collection = client.get_collection("ai_project_full")
        print("Using existing collection 'ai_project_full'")
    except:
        collection = client.create_collection(
            "ai_project_full",
            embedding_function=ef
        )
        print("Created new collection 'ai_project_full'")
        for i, chunk in enumerate(tqdm(chunks)):
            collection.add(
                documents=[chunk],
                metadatas=[{"source": f"chunk_{i}"}],
                ids=[str(i)]
            )'''
    return collection

'''def build_vector_db_local(chunks):
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    # Check if collection exists, otherwise create
    try:
        collection = client.get_collection("ai_project5")
        print("Using existing collection 'ai_project5'")
    except:
        collection = client.create_collection(
            "ai_project5",
            embedding_function=ef
        )
        print("Created new collection 'ai_project5'")
        # Only add documents if newly created
        for i, chunk in enumerate(tqdm(chunks)):
            collection.add(
                documents=[chunk],
                metadatas=[{"source": f"chunk_{i}"}],
                ids=[str(i)]
            )

    return collection'''

'''def build_vector_db_local(chunks):
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.create_collection(
    "ai_project5",
    embedding_function=ef  # <- use the function we defined
    )

    for i, chunk in enumerate(tqdm(chunks)):
        collection.add(
            documents=[chunk],
            metadatas=[{"source": f"chunk_{i}"}],
            ids=[str(i)]
        )
    return collection'''

# ------------------------
# Step 6: RAG query locally using Ollama
# ------------------------
def rag_query(collection, query):
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url="http://localhost:11434")
    results = collection.query(query_texts=[query], n_results=3)
    context = " ".join([doc for doc in results['documents'][0]])
    prompt = f"Answer using only the context below:\n{context}\n\nQuestion: {query}"
    return llm.invoke(prompt)

        # Start interactive chat
def chat_loop(collection):
    print("\n=== Erica AI Tutor ===")
    print("Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = rag_query(collection, query)
        print(f"\nErica: {answer}")


# ------------------------
# Main pipeline
# ------------------------
if __name__ == "__main__":
    print("Scraping website...")
    scrape_website(COURSE_URL)
    print("Converting HTML to text...")
    html_to_text()

    print("Downloading Zoom recordings...")
    zoom_links = []  # put Brightspace Zoom recording URLs here
    download_zoom_recordings(zoom_links)

    print("Transcribing Zoom recordings...")
    transcribe_zoom_recordings()

    print("Chunking text...")
    chunks = chunk_text()
    print(f"Total chunks: {len(chunks)}")

    print("Building vector DB with Ollama embeddings locally...")
    collection = build_vector_db_local(chunks)

    print("Pipeline ready! You can now query locally using:")
    print("answer = rag_query(collection, 'Your question')")
    
    chat_loop(collection)




