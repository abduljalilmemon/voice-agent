import os

from app.knowledge.ingest import get_collection

# Chroma's default L2 distance for all-MiniLM-L6-v2. Measured on this knowledge base, the best
# match for business questions scored <= 0.80 and for off-topic questions >= 0.92.
MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.88"))

def retrieve(query: str, n_results: int = 3) -> str:
    """Return the knowledge-base chunks relevant to the query ("" when nothing is close enough)."""
    results = get_collection().query(query_texts=[query], n_results=n_results)
    docs = [doc for doc, distance in zip(results["documents"][0], results["distances"][0], strict=True)
            if distance <= MAX_DISTANCE]
    return "\n\n".join(docs)
