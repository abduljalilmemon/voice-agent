import os
import chromadb

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "business_knowledge"

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if os.getenv("USE_ONNX_EMBEDDINGS", "false").lower() == "true":
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        embedding_fn = ONNXMiniLM_L6_V2()
    else:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
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

    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    collection = get_collection()

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"  Cleared {len(existing['ids'])} existing chunks")

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"  Ingested {len(chunks)} chunks from {filepath}")

if __name__ == "__main__":
    print("Ingesting knowledge base...")
    ingest_file("data/knowledge_base/dental_clinic.txt")
    print("Done.")