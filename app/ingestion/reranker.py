"""Local, free cross-encoder reranking via fastembed (ONNX, CPU, no API key).

Runs after hybrid search + RRF fusion (app/ingestion/store.py::search()) to
re-score candidates by actually reading the query against each candidate's
text, rather than RRF's rank-position blend of the vector/lexical legs.
Same lazy-singleton-plus-warm() shape as app/ingestion/embedder.py, and
shares its cache_dir — fastembed scopes cached model files per model-name
subfolder, so both models coexist there safely.
"""

from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from ..config import settings


@lru_cache(maxsize=1)
def _model() -> TextCrossEncoder:
    return TextCrossEncoder(
        model_name=settings.rerank_model, cache_dir=settings.embedding_cache_dir
    )


def warm() -> None:
    """Force the model to load now rather than on the first rerank call —
    called at startup (app/main.py) and at Docker build time, same reason
    as embedder.py::warm()."""
    _model()


def rerank(query: str, documents: list[str]) -> list[float]:
    """Score each document's relevance to query. Higher = more relevant."""
    if not documents:
        return []
    return list(_model().rerank(query, documents))
