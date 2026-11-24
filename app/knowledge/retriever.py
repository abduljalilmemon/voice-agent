from app.knowledge.ingest import get_collection


def retrieve(query: str, n_results: int = 3) -> str:
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    docs = results["documents"][0]
    return "\n\n".join(docs) if docs else ""