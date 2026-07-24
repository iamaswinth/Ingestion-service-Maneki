"""Integration tests for app/ingestion/store.py::search() (hybrid vector +
lexical RRF) against a live pgvector instance. Uses the pg_tenant fixture
(tests/conftest.py) for isolation/cleanup and hand-built Chunk rows with
fake 384-dim embeddings (basis vectors, so "identical" vs "orthogonal" is
unambiguous) rather than the real fastembed model, to keep this fast and
deterministic.
"""

import uuid

from app import config as config_module
from app.ingestion import questions
from app.ingestion import store as ingestion_store
from app.models import Chunk

_DIM = 384


def _fake_vector(seed: int, dim: int = _DIM) -> list[float]:
    """A basis vector with a 1.0 in position `seed % dim`. Two fake vectors
    with the same seed are identical (cosine similarity 1); with different
    seeds they're orthogonal (cosine similarity 0) — fully deterministic
    ranking control without needing the real embedding model."""
    vec = [0.0] * dim
    vec[seed % dim] = 1.0
    return vec


def _chunk(
    *,
    tenant_id: str,
    site_url: str,
    suffix: str,
    text: str,
    embedding_text: str | None = None,
    title: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"{tenant_id}-{suffix}",
        tenant_id=tenant_id,
        job_id="test-job",
        site_url=site_url,
        page_url=site_url,
        anchor_type="page",
        navigation=site_url,
        title=title,
        text=text,
        embedding_text=embedding_text or text,
    )


async def test_search_ranks_closest_vector_first(pg_tenant):
    tenant_id, site_url = pg_tenant
    chunk_a = _chunk(
        tenant_id=tenant_id, site_url=site_url, suffix="a",
        text="We offer a 30 day refund policy for all plans.",
    )
    chunk_b = _chunk(
        tenant_id=tenant_id, site_url=site_url, suffix="b",
        text="Our headquarters are located in San Francisco.",
    )
    await ingestion_store.replace_site_chunks(
        tenant_id, site_url, [chunk_a, chunk_b], [_fake_vector(0), _fake_vector(1)]
    )

    hits = await ingestion_store.search(
        tenant_id=tenant_id, embedding=_fake_vector(0), question="refund policy",
        top_k=5, hybrid=False, rerank=False,
    )

    assert len(hits) == 2
    assert hits[0].text == chunk_a.text
    assert hits[0].score > hits[1].score


async def test_search_dedupes_content_and_question_rows(pg_tenant):
    tenant_id, site_url = pg_tenant
    content = _chunk(
        tenant_id=tenant_id, site_url=site_url, suffix="content-1",
        text="We charge $10 per month for the pro plan.",
    )
    question_chunks = questions.build_question_chunks(
        [content],
        {content.chunk_id: ["How much does it cost?", "What's the monthly price?"]},
    )
    all_chunks = [content] + question_chunks
    # Content row gets a mediocre embedding; the second question row's
    # embedding matches the query exactly, so it should win and drag the
    # content chunk's text along via parent_chunk_id.
    embeddings = [_fake_vector(5), _fake_vector(6), _fake_vector(0)]
    await ingestion_store.replace_site_chunks(tenant_id, site_url, all_chunks, embeddings)

    hits = await ingestion_store.search(
        tenant_id=tenant_id, embedding=_fake_vector(0), question="price",
        top_k=5, hybrid=False, rerank=False,
    )

    assert len(hits) == 1
    assert hits[0].text == content.text
    assert hits[0].matched_question == "What's the monthly price?"


async def test_hybrid_search_can_surface_a_lexical_only_top_match(pg_tenant):
    tenant_id, site_url = pg_tenant
    vector_favored = _chunk(
        tenant_id=tenant_id, site_url=site_url, suffix="vec",
        text="Something about clouds and weather patterns.",
        embedding_text="something about clouds and weather patterns",
    )
    lexical_favored = _chunk(
        tenant_id=tenant_id, site_url=site_url, suffix="lex",
        text="Our enterprise plan includes unlimited seats.",
        embedding_text="Our enterprise plan includes unlimited seats",
    )
    await ingestion_store.replace_site_chunks(
        tenant_id, site_url, [vector_favored, lexical_favored],
        [_fake_vector(0), _fake_vector(50)],
    )
    query = "enterprise plan unlimited seats"

    vector_top1 = await ingestion_store.search(
        tenant_id=tenant_id, embedding=_fake_vector(0), question=query,
        top_k=1, hybrid=False, rerank=False,
    )
    hybrid_top1 = await ingestion_store.search(
        tenant_id=tenant_id, embedding=_fake_vector(0), question=query,
        top_k=1, hybrid=True, debug=True, rerank=False,
    )

    # Pure cosine similarity: the query embedding is an exact match for
    # vector_favored (orthogonal to lexical_favored), so it wins outright.
    assert vector_top1[0].text == vector_favored.text
    # RRF fusion: lexical_favored gets full credit on the lexical leg (its
    # embedding_text shares every query term) plus partial credit on the
    # vector leg (it's still the 2nd-closest of only 2 candidates), enough
    # combined score to outrank vector_favored's vector-leg-only credit.
    assert hybrid_top1[0].text == lexical_favored.text
    assert hybrid_top1[0].lexical_score is not None


async def test_search_isolates_tenants(pg_tenant):
    tenant_a, site_url = pg_tenant
    tenant_b = f"test-{uuid.uuid4().hex[:12]}"
    shared_text = "Shared-looking text used for the isolation test."
    chunk_a = _chunk(tenant_id=tenant_a, site_url=site_url, suffix="shared", text=shared_text)
    chunk_b = _chunk(tenant_id=tenant_b, site_url=site_url, suffix="shared", text=shared_text)
    await ingestion_store.replace_site_chunks(tenant_a, site_url, [chunk_a], [_fake_vector(9)])
    await ingestion_store.replace_site_chunks(tenant_b, site_url, [chunk_b], [_fake_vector(9)])
    try:
        hits = await ingestion_store.search(
            tenant_id=tenant_a, embedding=_fake_vector(9), question="isolation",
            top_k=5, hybrid=False, rerank=False,
        )
        assert len(hits) == 1
    finally:
        await ingestion_store.delete_site_chunks(tenant_b, site_url)


async def test_search_falls_back_to_vector_only_when_hybrid_raises(pg_tenant, monkeypatch):
    tenant_id, site_url = pg_tenant
    chunk = _chunk(
        tenant_id=tenant_id, site_url=site_url, suffix="only",
        text="Some retrievable content for the fallback test.",
    )
    await ingestion_store.replace_site_chunks(tenant_id, site_url, [chunk], [_fake_vector(3)])

    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated hybrid failure")

    monkeypatch.setattr(ingestion_store, "_search_hybrid", _boom)

    hits = await ingestion_store.search(
        tenant_id=tenant_id, embedding=_fake_vector(3), question="anything",
        top_k=5, hybrid=True, rerank=False,
    )

    assert len(hits) == 1
    assert hits[0].text == chunk.text


async def test_zero_lexical_weight_collapses_to_pure_vector_ranking(pg_tenant, monkeypatch):
    tenant_id, site_url = pg_tenant
    vector_favored = _chunk(
        tenant_id=tenant_id, site_url=site_url, suffix="vecfav",
        text="apple banana cherry", embedding_text="apple banana cherry",
    )
    lexical_favored = _chunk(
        tenant_id=tenant_id, site_url=site_url, suffix="lexfav",
        text="durian elderberry fig", embedding_text="durian elderberry fig",
    )
    await ingestion_store.replace_site_chunks(
        tenant_id, site_url, [vector_favored, lexical_favored],
        [_fake_vector(0), _fake_vector(20)],
    )
    monkeypatch.setattr(config_module.settings, "hybrid_lexical_weight", 0.0)

    hits = await ingestion_store.search(
        tenant_id=tenant_id, embedding=_fake_vector(0), question="durian elderberry fig",
        top_k=1, hybrid=True, rerank=False,
    )

    assert hits[0].text == vector_favored.text


async def test_rerank_promotes_the_real_semantic_match(pg_tenant):
    """Fake vectors force an irrelevant chunk to win on pure cosine
    similarity; the cross-encoder reranker should still promote the chunk
    that actually answers the query, proving reranking uses real text
    content, not just the fused vector/lexical scores."""
    tenant_id, site_url = pg_tenant
    irrelevant_but_vector_favored = _chunk(
        tenant_id=tenant_id, site_url=site_url, suffix="irrelevant",
        text="Our headquarters are located in San Francisco, California.",
    )
    relevant_but_vector_disfavored = _chunk(
        tenant_id=tenant_id, site_url=site_url, suffix="relevant",
        text="We offer a full refund within 30 days of purchase for any reason.",
    )
    await ingestion_store.replace_site_chunks(
        tenant_id, site_url,
        [irrelevant_but_vector_favored, relevant_but_vector_disfavored],
        [_fake_vector(0), _fake_vector(1)],
    )
    query = "what is your refund policy"

    without_rerank = await ingestion_store.search(
        tenant_id=tenant_id, embedding=_fake_vector(0), question=query,
        top_k=2, hybrid=False, rerank=False,
    )
    with_rerank = await ingestion_store.search(
        tenant_id=tenant_id, embedding=_fake_vector(0), question=query,
        top_k=2, hybrid=False, rerank=True, debug=True,
    )

    assert without_rerank[0].text == irrelevant_but_vector_favored.text
    assert with_rerank[0].text == relevant_but_vector_disfavored.text
    assert with_rerank[0].rerank_score is not None
    assert with_rerank[0].rerank_score > with_rerank[1].rerank_score
