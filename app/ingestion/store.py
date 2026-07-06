"""Postgres + pgvector storage for chunks, isolated per tenant_id.

Works identically against Neon (prod) or a local `pgvector/pgvector:pg16`
Docker container (dev/test) — just the `DATABASE_URL` changes. Schema is
created lazily on first pool acquisition (idempotent `CREATE ... IF NOT
EXISTS`), so there is no separate migration step to run.

Every read is scoped by `tenant_id` — this is the entire multi-tenant
isolation boundary. There is no other mechanism keeping tenants apart, so
never add a query path that skips this filter.
"""

import asyncpg

from .. import db
from ..models import Chunk, QueryHit

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
"""


async def get_pool() -> asyncpg.Pool:
    global _schema_ready
    pool = await db.get_pool()
    if not _schema_ready:
        async with pool.acquire() as conn:
            await conn.execute(_SCHEMA_SQL)
        _schema_ready = True
    return pool


async def delete_site_chunks(tenant_id: str, site_url: str) -> None:
    pool = await get_pool()
    await pool.execute(
        "DELETE FROM chunks WHERE tenant_id = $1 AND site_url = $2", tenant_id, site_url
    )


async def replace_site_chunks(
    tenant_id: str, site_url: str, chunks: list[Chunk], embeddings: list[list[float]]
) -> int:
    """Atomically swap all of a site's chunks for a fresh set from one crawl.

    Deletes by (tenant_id, site_url) rather than job_id so that pages/sections
    removed from the live site don't leave stale, unreachable rows behind from
    a previous crawl of the same site. Delete + insert share one transaction
    so a concurrent /query never sees the site's chunks disappear.
    """
    pool = await get_pool()
    rows = [
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
        )
        for c, emb in zip(chunks, embeddings)
    ]
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM chunks WHERE tenant_id = $1 AND site_url = $2",
                tenant_id,
                site_url,
            )
            if rows:
                await conn.executemany(
                    """
                    INSERT INTO chunks (
                        chunk_id, tenant_id, job_id, site_url, page_url,
                        section_id, parent_section_id, anchor_type, navigation,
                        title, content_type, chunk_index, text, embedding,
                        kind, parent_chunk_id, question
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
                    """,
                    rows,
                )
    return len(rows)


async def count_job_chunks(tenant_id: str, job_id: str) -> int:
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT COUNT(*) AS c FROM chunks WHERE tenant_id = $1 AND job_id = $2",
        tenant_id,
        job_id,
    )
    return int(row["c"]) if row else 0


async def search(
    tenant_id: str,
    embedding: list[float],
    top_k: int,
    site_url: Optional[str] = None,
    page_url: Optional[str] = None,
) -> list[QueryHit]:
    pool = await get_pool()

    conditions = ["tenant_id = $1"]
    params: list = [tenant_id]
    if site_url:
        params.append(site_url)
        conditions.append(f"site_url = ${len(params)}")
    if page_url:
        params.append(page_url)
        conditions.append(f"page_url = ${len(params)}")

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
