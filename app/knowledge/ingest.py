import logging

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "business_knowledge"

def get_collection():
    # all-MiniLM-L6-v2 via ONNX Runtime -- same model as sentence-transformers, without PyTorch.
    # The embedding function is persisted with the collection, so ingest and query must match.
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ONNXMiniLM_L6_V2()
    )

def ingest_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    collection = get_collection()

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.info("Cleared %d existing chunks", len(existing["ids"]))

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    logger.info("Ingested %d chunks from %s", len(chunks), filepath)

if __name__ == "__main__":
    from app.logging_setup import setup_logging

    setup_logging()
    ingest_file("data/knowledge_base/dental_clinic.txt")
