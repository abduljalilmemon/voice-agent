import logging
from functools import lru_cache

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from app.booking_rules import opening_hours_text
from app.config import BIZ_CONFIG

logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "business_knowledge"

@lru_cache(maxsize=1)
def get_collection():
    """Open the knowledge base once per process -- loading the embedding model takes ~1s."""
    # all-MiniLM-L6-v2 via ONNX Runtime -- same model as sentence-transformers, without PyTorch.
    # The embedding function is persisted with the collection, so ingest and query must match.
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ONNXMiniLM_L6_V2()
    )

def business_profile_chunk() -> str:
    """Core facts live in business_config.json (single source of truth); index them too."""
    c = BIZ_CONFIG
    return (f"CONTACT AND LOCATION: {c['business_name']} is located at {c['address']}. "
            f"Phone: {c['phone']}. Email: {c['email']}. Opening hours: {opening_hours_text()}")

def ingest_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [business_profile_chunk()] + [c.strip() for c in text.split("\n\n") if c.strip()]
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
