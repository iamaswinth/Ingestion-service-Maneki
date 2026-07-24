FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Bake the fastembed embedding + reranking models into the image at the
# same cache_dir the app uses at runtime (app/config.py: embedding_cache_dir),
# so the first live /query after boot never pays a model download — see
# app/ingestion/embedder.py and app/ingestion/reranker.py.
RUN python -c "from app.ingestion.embedder import warm as warm_embedder; \
    from app.ingestion.reranker import warm as warm_reranker; \
    warm_embedder(); warm_reranker()"

EXPOSE 8000
# WEB_CONCURRENCY controls worker count in prod; safe to raise per-replica —
# every cross-worker/cross-replica race (job claims, sales-script claims) is
# closed with an atomic UPDATE ... WHERE, see app/storage.py/salescript/store.py.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-2}
