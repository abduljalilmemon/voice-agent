FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    STORAGE_DIR=/app/storage

WORKDIR /app

# run as an unprivileged user
RUN useradd --create-home --uid 1000 app && chown app:app /app

# dependencies first -- this layer stays cached until requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

USER app

# bake the embedding model into the image so containers don't download it on startup
# (the downloaded archive is only needed when the extracted files are missing, so drop it)
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()(['warm up'])" \
    && rm -f /home/app/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz

COPY --chown=app:app . .
RUN mkdir -p /app/storage

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# rebuild the knowledge base from data/ on every start, then serve
CMD ["sh", "-c", "python -m app.knowledge.ingest && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
