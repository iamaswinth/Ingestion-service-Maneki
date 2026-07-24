"""Postgres + pgvector storage for chunks, isolated per tenant_id.

Works identically against Neon (prod) or a local `pgvector/pgvector:pg16`
Docker container (dev/test) — just the `DATABASE_URL` changes. Schema is
created lazily on first pool acquisition (idempotent `CREATE ... IF NOT
EXISTS`), so there is no separate migration step to run.

Every read is scoped by `tenant_id` — this is the entire multi-tenant
isolation boundary. There is no other mechanism keeping tenants apart, so
never add a query path that skips this filter.
"""

import asyncio
import logging
from typing import Optional

import asyncpg

from .. import db, schema_guard
from ..config import settings
from ..models import Chunk, QueryHit
from .reranker import rerank as rerank_documents

logger = logging.getLogger(__name__)

_schema_ready = False

_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id          TEXT PRIMARY KEY,
  tenant_id         TEXT NOT NULL,
  job_id            TEXT NOT NULL,
  site_url          TEXT NOT NULL,
  page_url          TEXT NOT NULL,
  section_id        TEXT,
  parent_section_id TEXT,
  anchor_type       TEXT NOT NULL,
  navigation        TEXT NOT NULL,
  title             TEXT,
  content_type      TEXT,
  chunk_index       INT NOT NULL,
  text              TEXT NOT NULL,
  embedding         vector(384) NOT NULL,
  created_at        timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks (tenant_id);
CREATE INDEX IF NOT EXISTS chunks_tenant_page_idx ON chunks (tenant_id, page_url);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
  ON chunks USING hnsw (embedding vector_cosine_ops);

-- Doc2query (idempotent migrations for pre-existing tables):
-- 'question' rows are synthetic visitor questions embedded as extra vectors
-- pointing back at their content chunk via parent_chunk_id.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'content';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS parent_chunk_id TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS question TEXT;

-- Hybrid retrieval (idempotent migration for pre-existing tables):
-- embedding_text was previously computed at ingest time but never persisted;
-- storing it lets full-text search run over the same text the dense
-- embedding was built from (title-prefixed content, or the raw synthetic
-- question for kind='question' rows).
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_text TEXT;

-- Two-arg to_tsvector with a literal config name is the documented pattern
-- for a generated column (the config is resolved at DDL time). COALESCE to
-- `text` so lexical search still works for any row before backfill runs.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector tsvector
  GENERATED ALWAYS AS (to_tsvector('english', coalesce(embedding_text, text))) STORED;

CREATE INDEX IF NOT EXISTS chunks_search_vector_idx
  ON chunks USING GIN (search_vector);

-- Backfill rows ingested before embedding_text existed. Exact for
-- kind='question' (question text is already stored verbatim); an
-- approximation for kind='content' since the page-level title isn't
-- persisted per-chunk (only the section title is). WHERE-guarded so this
-- is a no-op scan once every row has been backfilled.
UPDATE chunks
SET embedding_text = CASE
    WHEN kind = 'question' THEN question
    ELSE coalesce(title || ': ' || text, text)
END
WHERE embedding_text IS NULL;
"""


async def get_pool() -> asyncpg.Pool:
    global _schema_ready
    pool = await db.get_pool()
    if not _schema_ready:
        async with pool.acquire() as conn:
            # Created in dev/test, only verified in production — see
            # app/schema_guard.py.
            if schema_guard.auto_create_enabled():
                await conn.execute(_SCHEMA_SQL)
            else:
                await schema_guard.verify_tables(conn, ("chunks",))
        _schema_ready = True
    return pool


def _chunk_rows(chunks: list[Chunk], embeddings: list[list[float]]) -> list[tuple]:
    return [
        (
            c.chunk_id,
            c.tenant_id,
            c.job_id,
            c.site_url,
            c.page_url,
            c.section_id,
            c.parent_section_id,
            c.anchor_type,
            c.navigation,
            c.title,
            c.content_type,
            c.chunk_index,
            c.text,
            emb,
            c.kind,
            c.parent_chunk_id,
            c.question,
            c.embedding_text,
        )
        for c, emb in zip(chunks, embeddings)
    ]


async def _insert_chunk_rows(conn: asyncpg.Connection, rows: list[tuple]) -> None:
    if rows:
        await conn.executemany(
            """
            INSERT INTO chunks (
                chunk_id, tenant_id, job_id, site_url, page_url,
                section_id, parent_section_id, anchor_type, navigation,
                title, content_type, chunk_index, text, embedding,
                kind, parent_chunk_id, question, embedding_text
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
            """,
            rows,
        )


async def delete_site_chunks(tenant_id: str, site_url: str) -> None:
    """Wipes a failed/emptied crawl's chunks. Excludes kind='sales_script' so a
    reviewed, approved sales script survives a scrape that later 404s or comes
    back with zero pages."""
    pool = await get_pool()
    await pool.execute(
        "DELETE FROM chunks WHERE tenant_id = $1 AND site_url = $2 AND kind != 'sales_script'",
        tenant_id,
        site_url,
    )


async def replace_site_chunks(
    tenant_id: str, site_url: str, chunks: list[Chunk], embeddings: list[list[float]]
) -> int:
    """Atomically swap all of a site's content/question chunks for a fresh set
    from one crawl.

    Deletes by (tenant_id, site_url) rather than job_id so that pages/sections
    removed from the live site don't leave stale, unreachable rows behind from
    a previous crawl of the same site. Delete + insert share one transaction
    so a concurrent /query never sees the site's chunks disappear.

    Excludes kind='sales_script': those rows are replaced only via
    replace_site_script_chunks (on approval), so an ordinary re-ingest never
    wipes a reviewed, approved sales script out from under the voice agent.
    """
    pool = await get_pool()
    rows = _chunk_rows(chunks, embeddings)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM chunks WHERE tenant_id = $1 AND site_url = $2 AND kind != 'sales_script'",
                tenant_id,
                site_url,
            )
            await _insert_chunk_rows(conn, rows)
    return len(rows)


async def replace_site_script_chunks(
    tenant_id: str, site_url: str, chunks: list[Chunk], embeddings: list[list[float]]
) -> int:
    """Swap only a site's kind='sales_script' chunks, leaving content/question
    rows from the normal ingest path untouched. Called on sales-script
    approval, mirroring replace_site_chunks's delete+insert-in-one-transaction
    shape.
    """
    pool = await get_pool()
    rows = _chunk_rows(chunks, embeddings)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM chunks WHERE tenant_id = $1 AND site_url = $2 AND kind = 'sales_script'",
                tenant_id,
                site_url,
            )
            await _insert_chunk_rows(conn, rows)
    return len(rows)


async def count_job_chunks(tenant_id: str, job_id: str) -> int:
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT COUNT(*) AS c FROM chunks WHERE tenant_id = $1 AND job_id = $2",
        tenant_id,
        job_id,
    )
    return int(row["c"]) if row else 0


def _build_conditions(
    tenant_id: str, site_url: Optional[str], page_url: Optional[str]
) -> tuple[list[str], list]:
    conditions = ["tenant_id = $1"]
    params: list = [tenant_id]
    if site_url:
        params.append(site_url)
        conditions.append(f"site_url = ${len(params)}")
    if page_url:
        params.append(page_url)
        conditions.append(f"page_url = ${len(params)}")
    return conditions, params


async def _search_vector_only(
    tenant_id: str,
    embedding: list[float],
    top_k: int,
    site_url: Optional[str],
    page_url: Optional[str],
) -> list[QueryHit]:
    pool = await get_pool()
    conditions, params = _build_conditions(tenant_id, site_url, page_url)

    params.append(embedding)
    embedding_idx = len(params)
    params.append(top_k)
    top_k_idx = len(params)
    # Doc2query gives each content chunk ~1 + questions_per_chunk vectors, so
    # over-fetch by that factor before deduping down to top_k distinct parents.
    params.append(max(top_k * 8, 50))
    candidate_idx = len(params)

    # A content chunk can match via its own text vector AND via any of its
    # synthetic question vectors (doc2query). Keep only the best-scoring hit
    # per underlying content chunk: COALESCE(parent_chunk_id, chunk_id) is
    # the content chunk's id for both row kinds.
    #
    # The candidate CTE's ORDER BY + LIMIT lets the planner use the HNSW index
    # for a nearest-neighbor top-N scan; the outer dedupe then runs over just
    # that small candidate set instead of every tenant row.
    query = f"""
        WITH candidates AS (
            SELECT *, 1 - (embedding <=> ${embedding_idx}) AS score
            FROM chunks
            WHERE {' AND '.join(conditions)}
            ORDER BY embedding <=> ${embedding_idx}
            LIMIT ${candidate_idx}
        )
        SELECT text, page_url, section_id, title, content_type, anchor_type,
               navigation, kind, question, score
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY COALESCE(parent_chunk_id, chunk_id)
                       ORDER BY score DESC
                   ) AS rn
            FROM candidates
        ) ranked
        WHERE rn = 1
        ORDER BY score DESC
        LIMIT ${top_k_idx}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [
        QueryHit(
            text=row["text"],
            score=float(row["score"]),
            page_url=row["page_url"],
            section_id=row["section_id"],
            title=row["title"],
            content_type=row["content_type"] or "generic",
            anchor_type=row["anchor_type"],
            navigation=row["navigation"],
            matched_question=row["question"] if row["kind"] == "question" else None,
        )
        for row in rows
    ]


def _normalize_rrf(raw_score: float) -> float:
    """Map a raw RRF fusion score onto 0..1 so it is comparable with the
    cosine similarity `_search_vector_only` returns.

    Raw RRF lives on a scale set entirely by `hybrid_rrf_k`: the best a chunk
    can score is `(w_vec + w_lex) / (k + 1)` — about **0.033** at the default
    k=60 — while the vector-only path returns cosine similarity in 0..1.

    One `score` field carrying two incompatible scales is not a cosmetic
    inconsistency: `hybrid_search_enabled` defaults to true, and
    voice_runtime's `retrieval_agentic_score_threshold` (0.35, calibrated for
    cosine) is then mathematically unreachable. Every hit is judged "thin",
    the agent discards a perfectly good knowledge base, and every grounded
    question falls through to "I don't have any information about that".

    After normalising: rank 1 in both legs = 1.0, rank 1 in a single leg =
    0.5, and something scraping in at rank ~50 of one leg lands near 0.28 —
    so the existing 0.35 threshold recovers its intended meaning of "actually
    ranked well by at least one retriever".
    """
    best_possible = (
        settings.hybrid_vector_weight + settings.hybrid_lexical_weight
    ) / (settings.hybrid_rrf_k + 1)
    if best_possible <= 0:
        return 0.0
    return min(raw_score / best_possible, 1.0)


async def _search_hybrid(
    tenant_id: str,
    embedding: list[float],
    question: str,
    top_k: int,
    site_url: Optional[str],
    page_url: Optional[str],
    debug: bool,
) -> list[QueryHit]:
    """Vector + Postgres full-text, fused via Reciprocal Rank Fusion (RRF).

    Both legs run as CTEs inside a single round trip (candidates carry only
    ids/ranks; the wide display columns are joined back in once, only for the
    final top_k winners). The dedupe below is the same
    COALESCE(parent_chunk_id, chunk_id) invariant _search_vector_only uses,
    just ordered by the fused rrf_score instead of raw cosine score, so a
    chunk matched via its own text (either leg) or via a sibling doc2query
    question (either leg) still collapses to one row.
    """
    pool = await get_pool()
    conditions, params = _build_conditions(tenant_id, site_url, page_url)
    where_clause = " AND ".join(conditions)

    params.append(question)
    question_idx = len(params)
    params.append(embedding)
    embedding_idx = len(params)
    candidate_n = max(
        top_k * settings.hybrid_candidate_multiplier, settings.hybrid_candidate_floor
    )
    params.append(candidate_n)
    vector_candidate_idx = len(params)
    params.append(candidate_n)
    lexical_candidate_idx = len(params)
    params.append(settings.hybrid_rrf_k)
    rrf_k_idx = len(params)
    params.append(settings.hybrid_vector_weight)
    vec_weight_idx = len(params)
    params.append(settings.hybrid_lexical_weight)
    lex_weight_idx = len(params)
    params.append(top_k)
    top_k_idx = len(params)

    query = f"""
        WITH q AS (
            -- websearch_to_tsquery ANDs every bare term together, so a
            -- multi-word conversational query only matches a chunk that
            -- happens to contain *all* of its words. Rewriting the top-level
            -- ANDs to ORs (leaving quoted-phrase proximity operators intact)
            -- turns this into BM25-style "more/rarer matching terms rank
            -- higher" scoring via ts_rank_cd, instead of all-or-nothing.
            SELECT to_tsquery(
                'english',
                replace(websearch_to_tsquery('english', ${question_idx})::text, ' & ', ' | ')
            ) AS tsq
        ),
        vector_candidates AS (
            SELECT chunk_id, parent_chunk_id,
                   1 - (embedding <=> ${embedding_idx}) AS vscore,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> ${embedding_idx}) AS vrank
            FROM chunks
            WHERE {where_clause}
            ORDER BY embedding <=> ${embedding_idx}
            LIMIT ${vector_candidate_idx}
        ),
        lexical_scored AS (
            SELECT c.chunk_id, c.parent_chunk_id,
                   ts_rank_cd(c.search_vector, q.tsq) AS lscore
            FROM chunks c, q
            WHERE {where_clause} AND c.search_vector @@ q.tsq
        ),
        lexical_candidates AS (
            SELECT chunk_id, parent_chunk_id, lscore,
                   ROW_NUMBER() OVER (ORDER BY lscore DESC) AS lrank
            FROM lexical_scored
            ORDER BY lscore DESC
            LIMIT ${lexical_candidate_idx}
        ),
        fused AS (
            SELECT
                COALESCE(v.chunk_id, l.chunk_id) AS chunk_id,
                COALESCE(v.parent_chunk_id, l.parent_chunk_id) AS parent_chunk_id,
                -- Explicit ::float8 casts matter: with unspecified parameter
                -- types, Postgres infers each $n's type from its syntactic
                -- context, and `+ v.vrank`/`+ l.lrank` (bigint) would pull
                -- the weight/k params into bigint too, silently truncating
                -- the whole division to 0 via integer division.
                COALESCE(${vec_weight_idx}::float8 / (${rrf_k_idx}::float8 + v.vrank), 0)
              + COALESCE(${lex_weight_idx}::float8 / (${rrf_k_idx}::float8 + l.lrank), 0) AS rrf_score,
                v.vscore, l.lscore
            FROM vector_candidates v
            FULL OUTER JOIN lexical_candidates l ON v.chunk_id = l.chunk_id
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY COALESCE(parent_chunk_id, chunk_id)
                       ORDER BY rrf_score DESC
                   ) AS rn
            FROM fused
        )
        SELECT c.text, c.page_url, c.section_id, c.title, c.content_type,
               c.anchor_type, c.navigation, c.kind, c.question,
               r.rrf_score, r.vscore, r.lscore
        FROM ranked r
        JOIN chunks c ON c.chunk_id = r.chunk_id
        WHERE r.rn = 1
        ORDER BY r.rrf_score DESC
        LIMIT ${top_k_idx}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [
        QueryHit(
            text=row["text"],
            score=_normalize_rrf(float(row["rrf_score"])),
            page_url=row["page_url"],
            section_id=row["section_id"],
            title=row["title"],
            content_type=row["content_type"] or "generic",
            anchor_type=row["anchor_type"],
            navigation=row["navigation"],
            matched_question=row["question"] if row["kind"] == "question" else None,
            vector_score=(
                float(row["vscore"]) if debug and row["vscore"] is not None else None
            ),
            lexical_score=(
                float(row["lscore"]) if debug and row["lscore"] is not None else None
            ),
        )
        for row in rows
    ]


async def _rerank_hits(question: str, hits: list[QueryHit], debug: bool) -> list[QueryHit]:
    scores = await asyncio.to_thread(rerank_documents, question, [h.text for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
    return [h.model_copy(update={"rerank_score": s}) if debug else h for h, s in ranked]


async def search(
    tenant_id: str,
    embedding: list[float],
    question: str,
    top_k: int,
    site_url: Optional[str] = None,
    page_url: Optional[str] = None,
    hybrid: Optional[bool] = None,
    debug: bool = False,
    rerank: Optional[bool] = None,
) -> list[QueryHit]:
    use_hybrid = settings.hybrid_search_enabled if hybrid is None else hybrid
    use_rerank = settings.rerank_enabled if rerank is None else rerank

    # Fetch more than top_k when reranking so the cross-encoder has real
    # room to promote a good match that RRF/cosine ranked lower — the final
    # top_k slice happens only after reranking, below.
    fetch_k = top_k
    if use_rerank:
        fetch_k = max(
            top_k,
            min(top_k * settings.rerank_candidate_multiplier, settings.rerank_candidate_ceiling),
        )

    if not use_hybrid:
        hits = await _search_vector_only(tenant_id, embedding, fetch_k, site_url, page_url)
    else:
        try:
            hits = await _search_hybrid(
                tenant_id, embedding, question, fetch_k, site_url, page_url, debug
            )
        except Exception:
            logger.exception("Hybrid search failed; falling back to vector-only")
            hits = await _search_vector_only(tenant_id, embedding, fetch_k, site_url, page_url)

    if use_rerank and hits:
        hits = await _rerank_hits(question, hits, debug)

    return hits[:top_k]
