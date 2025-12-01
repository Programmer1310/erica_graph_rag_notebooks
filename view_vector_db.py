import chromadb
from chromadb.config import Settings

DB_DIR = "data/vector_db"

def view_vector_db():
    client = chromadb.PersistentClient(path=DB_DIR)

    # list all collections
    collections = client.list_collections()
    print("\n=== Collections in DB ===")
    for c in collections:
        print(" -", c.name)

    if not collections:
        print("No collections found.")
        return

    # open main collection
    collection = client.get_collection("ai_course")

    # get all items
    print("\n=== Fetching all items ===")
    results = collection.get(include=["documents", "embeddings", "metadatas"])

    ids = results["ids"]
    docs = results["documents"]
    metas = results["metadatas"]

    print(f"\nTotal items: {len(ids)}")
    print("-" * 60)

    for i, id_ in enumerate(ids):
        print(f"ID: {id_}")
        print(f"Document preview: {docs[i][:200]}...")
        print(f"Metadata: {metas[i]}")
        print("-" * 60)


def demo_query():
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection("ai_course")

    query = "what is artificial intelligence?"
    print(f"\n=== Running query: {query} ===")

    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    for i in range(len(results["documents"][0])):
        print(f"\nResult {i+1}:")
        print(results["documents"][0][i][:300], "...")
        print("ID:", results["ids"][0][i])
        print("-" * 60)


#if __name__ == "__main__":
view_vector_db()
demo_query()
