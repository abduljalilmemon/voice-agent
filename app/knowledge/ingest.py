import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "business_knowledge"

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

def ingest_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # split into chunks by double newline
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

    collection = get_collection()

    # clear existing data
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"  Cleared {len(existing['ids'])} existing chunks")

    # add new chunks
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"  Ingested {len(chunks)} chunks from {filepath}")

if __name__ == "__main__":
    print("Ingesting knowledge base...")
    ingest_file("data/knowledge_base/dental_clinic.txt")
    print("Done.")