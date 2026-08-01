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
-- Every ingest's delete+replace (replace_site_chunks) and delete_site_chunks
-- filter by exactly this pair; without it they ran against the tenant-only
-- index above and rechecked every one of that tenant's rows against site_url.
CREATE INDEX IF NOT EXISTS chunks_tenant_site_idx ON chunks (tenant_id, site_url);
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
            -- Belt-and-braces: chunk_id is a content hash, so a residual
            -- collision (two distinct logical chunks hashing the same id)
            -- would otherwise violate the primary key and fail the whole
            -- batch insert for this transaction, taking every other chunk in
            -- it down too. Degrade to last-writer-wins instead.
            ON CONFLICT (chunk_id) DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id,
                job_id = EXCLUDED.job_id,
                site_url = EXCLUDED.site_url,
                page_url = EXCLUDED.page_url,
                section_id = EXCLUDED.section_id,
                parent_section_id = EXCLUDED.parent_section_id,
                anchor_type = EXCLUDED.anchor_type,
                navigation = EXCLUDED.navigation,
                title = EXCLUDED.title,
                content_type = EXCLUDED.content_type,
                chunk_index = EXCLUDED.chunk_index,
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                kind = EXCLUDED.kind,
                parent_chunk_id = EXCLUDED.parent_chunk_id,
                question = EXCLUDED.question,
                embedding_text = EXCLUDED.embedding_text
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
    tenant_id: str,
    site_url: Optional[str],
    page_url: Optional[str],
    include_questions: bool = True,
    content_type: Optional[str] = None,
    kind: Optional[str] = None,
) -> tuple[list[str], list]:
    conditions = ["tenant_id = $1"]
    params: list = [tenant_id]
    if site_url:
        params.append(site_url)
        conditions.append(f"site_url = ${len(params)}")
    if page_url:
        params.append(page_url)
        conditions.append(f"page_url = ${len(params)}")
    if content_type:
        params.append(content_type)
        conditions.append(f"content_type = ${len(params)}")
    if kind:
        # An explicit kind ask (e.g. "only sales_script") is a stronger,
        # more specific constraint than the include_questions default below
        # — applying both could be self-contradictory (kind='question' AND
        # kind != 'question' is never satisfiable), so an explicit kind wins
        # outright rather than being combined with it.
        params.append(kind)
        conditions.append(f"kind = ${len(params)}")
    elif not include_questions:
        # No placeholder needed: 'question' is a fixed literal, not caller
        # input. Lets scripts/eval_retrieval.py (and any future caller) A/B
        # doc2query's contribution against a live tenant without a re-ingest.
        conditions.append("kind != 'question'")
    return conditions, params


async def _search_vector_only(
    tenant_id: str,
    embedding: list[float],
    top_k: int,
    site_url: Optional[str],
    page_url: Optional[str],
    include_questions: bool = True,
    content_type: Optional[str] = None,
    kind: Optional[str] = None,
) -> list[QueryHit]:
    pool = await get_pool()
    conditions, params = _build_conditions(
        tenant_id, site_url, page_url, include_questions, content_type, kind
    )

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
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY COALESCE(parent_chunk_id, chunk_id)
                       ORDER BY score DESC
                   ) AS rn
            FROM candidates
        )
        -- Question rows carry no text of their own (see
        -- questions.py::build_question_chunks) — when the winning row is a
        -- question, its parent's real content is fetched via this join
        -- instead. A no-op (COALESCE falls through to ranked.text) for
        -- kind='content' rows, where parent_chunk_id is null.
        SELECT COALESCE(parent.text, ranked.text) AS text, ranked.page_url,
               ranked.section_id, ranked.title, ranked.content_type,
               ranked.anchor_type, ranked.navigation, ranked.kind,
               ranked.question, ranked.score
        FROM ranked
        LEFT JOIN chunks parent ON parent.chunk_id = ranked.parent_chunk_id
        WHERE ranked.rn = 1
        ORDER BY ranked.score DESC
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
    include_questions: bool = True,
    content_type: Optional[str] = None,
    kind: Optional[str] = None,
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
    conditions, params = _build_conditions(
        tenant_id, site_url, page_url, include_questions, content_type, kind
    )
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
        -- Question rows carry no text of their own (see
        -- questions.py::build_question_chunks); when the winning row is a
        -- question, its parent's real content is fetched via this second
        -- join. A no-op (COALESCE falls through to c.text) for kind='content'
        -- rows, where parent_chunk_id is null.
        SELECT COALESCE(parent.text, c.text) AS text, c.page_url, c.section_id,
               c.title, c.content_type, c.anchor_type, c.navigation, c.kind,
               c.question, r.rrf_score, r.vscore, r.lscore
        FROM ranked r
        JOIN chunks c ON c.chunk_id = r.chunk_id
        LEFT JOIN chunks parent ON parent.chunk_id = c.parent_chunk_id
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


def _normalize_rerank_scores(scores: list[float]) -> list[float]:
    """Min-max the cross-encoder's raw logits onto 0..1 within this batch.

    A plain sigmoid was tried first and measured against the real model this
    app ships (Xenova/ms-marco-MiniLM-L-6-v2, see reranker.py): its logits run
    far more negative than a sigmoid assumes — a clearly relevant one-sentence
    passage scored -5.8, an irrelevant one -11.3. `sigmoid(-5.8) ≈ 0.003`, so a
    fixed sigmoid would leave *every* reranked score under voice_runtime's 0.35
    `_hits_insufficient` threshold regardless of relevance — worse than the bug
    being fixed, since a great match would now always read as "thin". This
    model's logits carry no absolute calibration this codebase can rely on
    without ingesting real content and empirically tuning against it (exactly
    the trap the sigmoid version fell into from a single hand-picked example).

    Min-max over the batch actually being reranked guarantees what the
    downstream re-sort needs — the top-ranked hit always scores highest — using
    only relative information, no hardcoded constant. It has one known
    limitation: if every candidate in the batch is equally irrelevant, the
    "best of a bad set" still lands at 1.0, same as a genuinely great match
    would. voice_runtime's 0.35 threshold was tuned against cosine/RRF scores,
    not cross-encoder logits, so its accuracy against *this* scale is not
    validated here — `debug=True`'s `rerank_score` field exists precisely so
    that can be measured against real traffic rather than assumed.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    spread = hi - lo
    if spread <= 0:
        # No signal to discriminate on (a single hit, or a genuine tie) —
        # 0.5 reads as "no information", not the false confidence of 1.0 or
        # the false alarm of 0.0.
        return [0.5] * len(scores)
    return [(s - lo) / spread for s in scores]


async def _rerank_hits(question: str, hits: list[QueryHit], debug: bool) -> list[QueryHit]:
    """Cross-encoder rerank. `score` is overwritten with the (batch-normalized)
    rerank outcome — not left at its pre-rerank RRF/cosine value.

    Previously `score` was never touched here at all (only `rerank_score` was,
    and only under `debug`), so every consumer that sorts or thresholds on
    `score` — voice_runtime's `_merge_hits` re-sorts on it, `_hits_insufficient`
    thresholds it against 0.35 — silently saw the pre-rerank order. The
    cross-encoder forward pass ran and changed nothing. See
    `_normalize_rerank_scores` for why this is a min-max, not a sigmoid. The
    raw logit survives in `rerank_score`, under `debug`, for tuning.
    """
    scores = await asyncio.to_thread(rerank_documents, question, [h.text for h in hits])
    normalized = _normalize_rerank_scores(scores)
    ranked = sorted(zip(hits, scores, normalized), key=lambda triple: triple[1], reverse=True)
    return [
        h.model_copy(update={"score": norm, **({"rerank_score": s} if debug else {})})
        for h, s, norm in ranked
    ]


def _apply_section_diversity_cap(hits: list[QueryHit], max_per_section: int) -> list[QueryHit]:
    """Cap hits from any one (page_url, section_id) at `max_per_section`.

    Ranking alone can hand back top_k slices of a single section — five
    different sentences of the pricing block for "what do you offer?" — which
    both starves the agent of breadth and, since voice_runtime's
    `_pick_navigation` always follows `hits[0]`, lets ranking alone decide
    where the visitor's browser navigates. Over-full sections are pushed to
    the back rather than dropped, so if the tenant's content genuinely has
    fewer than top_k distinct sections, the slice below still fills out to
    top_k instead of coming back short.
    """
    if max_per_section <= 0:
        return hits
    counts: dict[tuple[Optional[str], Optional[str]], int] = {}
    kept: list[QueryHit] = []
    overflow: list[QueryHit] = []
    for hit in hits:
        key = (hit.page_url, hit.section_id)
        if counts.get(key, 0) < max_per_section:
            counts[key] = counts.get(key, 0) + 1
            kept.append(hit)
        else:
            overflow.append(hit)
    return kept + overflow


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
    include_questions: bool = True,
    content_type: Optional[str] = None,
    kind: Optional[str] = None,
) -> list[QueryHit]:
    use_hybrid = settings.hybrid_search_enabled if hybrid is None else hybrid
    use_rerank = settings.rerank_enabled if rerank is None else rerank
    apply_diversity_cap = settings.max_hits_per_section > 0

    # Fetch more than top_k whenever something downstream needs headroom to
    # reorder or filter within the candidate set: reranking needs room for the
    # cross-encoder to promote a match RRF/cosine ranked lower, and the
    # per-section diversity cap below needs overflow candidates to draw from
    # instead of already being handed exactly top_k rows (which, pre-cap,
    # could all be one section). Both share the same oversample knobs — no
    # need for a second multiplier/ceiling pair for what is the same "give the
    # next step real candidates to work with" need.
    fetch_k = top_k
    if use_rerank or apply_diversity_cap:
        fetch_k = max(
            top_k,
            min(top_k * settings.rerank_candidate_multiplier, settings.rerank_candidate_ceiling),
        )

    if not use_hybrid:
        hits = await _search_vector_only(
            tenant_id, embedding, fetch_k, site_url, page_url, include_questions,
            content_type, kind,
        )
    else:
        try:
            hits = await _search_hybrid(
                tenant_id, embedding, question, fetch_k, site_url, page_url, debug,
                include_questions, content_type, kind,
            )
        except Exception:
            logger.exception("Hybrid search failed; falling back to vector-only")
            hits = await _search_vector_only(
                tenant_id, embedding, fetch_k, site_url, page_url, include_questions,
                content_type, kind,
            )

    if use_rerank and hits:
        hits = await _rerank_hits(question, hits, debug)

    if apply_diversity_cap:
        hits = _apply_section_diversity_cap(hits, settings.max_hits_per_section)

    return hits[:top_k]
