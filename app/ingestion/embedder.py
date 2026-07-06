"""Local, free embeddings via fastembed (ONNX, CPU, no API key).

The model is a lazy module-level singleton — first call pays a ~2s load (plus
a one-time ~130MB download), every call after is fast. Both embedding calls
are synchronous/CPU-bound, so callers run them in a thread via
`asyncio.to_thread` to avoid blocking the event loop.

bge models are trained with an asymmetric convention: passages are embedded
as-is, but queries get a special instruction prefix prepended. Using
`query_embed` for questions and `embed` for stored chunks (matching training)
measurably improves retrieval quality over embedding both the same way.
"""

from functools import lru_cache

from fastembed import TextEmbedding

from ..config import settings


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=settings.embedding_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts for storage. Order matches the input list."""
    if not texts:
        return []
    return [vec.tolist() for vec in _model().embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embed a user question. Uses bge's query instruction prefix."""
    return next(iter(_model().query_embed(text))).tolist()
