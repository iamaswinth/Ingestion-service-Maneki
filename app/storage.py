"""Postgres-backed persistence for jobs and scraped pages.

Two tables:

    jobs   one row per crawl — JobState, keyed by job_id
    pages  one row per scraped page, keyed by (job_id, page_index)

Living in Postgres (rather than on local disk) means multiple uvicorn
workers or container replicas share state with no shared volume, and the
scraping->persisted transition can be claimed atomically across all of them
(see `mark_persisting`) instead of relying on an in-process lock.
"""

import asyncio

import asyncpg

from . import db, schema_guard
from .ingestion.embedder import embed_documents
from .links import page_key
from .models import (
    JobPages,
    JobState,
    Page,
    PageAction,
    PageActionsRecord,
    PageLink,
    PageLinksRecord,
    PageSummary,
    Product,
    Section,
)
from .products import make_product_key

_schema_ready = False

_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS jobs (
  job_id             TEXT PRIMARY KEY,
  url                TEXT NOT NULL,
  tenant_id          TEXT NOT NULL,
  status             TEXT NOT NULL,
  completed_pages    INT NOT NULL DEFAULT 0,
  total_pages        INT NOT NULL DEFAULT 0,
  error              TEXT,
  persisted          BOOLEAN NOT NULL DEFAULT false,
  ingest_status      TEXT NOT NULL DEFAULT 'not_started',
  ingested_chunks    INT NOT NULL DEFAULT 0,
  ingested_questions INT NOT NULL DEFAULT 0,
  ingest_error       TEXT,
  created_at         timestamptz NOT NULL DEFAULT now(),
  updated_at         timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS jobs_tenant_idx ON jobs (tenant_id);
CREATE INDEX IF NOT EXISTS jobs_ingest_status_idx ON jobs (ingest_status);
CREATE INDEX IF NOT EXISTS jobs_open_idx ON jobs (status) WHERE persisted = false;

CREATE TABLE IF NOT EXISTS pages (
  job_id         TEXT NOT NULL REFERENCES jobs (job_id) ON DELETE CASCADE,
  page_index     INT NOT NULL,
  url            TEXT NOT NULL,
  title          TEXT,
  description    TEXT,
  markdown       TEXT NOT NULL,
  chars          INT NOT NULL,
  section_count  INT NOT NULL DEFAULT 0,
  sections       JSONB,
  links          JSONB,
  url_key        TEXT,
  page_actions   JSONB,
  created_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (job_id, page_index)
);

CREATE INDEX IF NOT EXISTS pages_job_idx ON pages (job_id);
CREATE INDEX IF NOT EXISTS pages_url_key_idx ON pages (url_key);

CREATE TABLE IF NOT EXISTS products (
  product_key    TEXT PRIMARY KEY,
  tenant_id      TEXT NOT NULL,
  job_id         TEXT NOT NULL,
  site_url       TEXT NOT NULL,
  page_url       TEXT NOT NULL,
  name           TEXT NOT NULL,
  sku            TEXT,
  price_amount   NUMERIC,
  price_currency TEXT,
  availability   TEXT,
  image_url      TEXT,
  description    TEXT,
  attributes     JSONB,
  source         TEXT NOT NULL,
  embedding      vector(384),
  created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS products_tenant_idx      ON products (tenant_id);
CREATE INDEX IF NOT EXISTS products_tenant_site_idx ON products (tenant_id, site_url);
CREATE INDEX IF NOT EXISTS products_tenant_page_idx ON products (tenant_id, page_url);
CREATE INDEX IF NOT EXISTS products_embedding_idx
  ON products USING hnsw (embedding vector_cosine_ops);
"""
# `links`/`url_key`/`page_actions` and the `products` table land in a fresh
# dev DB via the CREATE TABLE(s) above. An existing database (one that
# already had `pages` before this change) only gets them via
# migrations/0004_page_links.sql and migrations/0005_products_and_actions.sql
# — CREATE TABLE IF NOT EXISTS is a silent no-op against a table that
# already exists, and schema_guard.verify_tables only checks the table
# itself, never columns. Per CLAUDE.md, new schema changes go in a new
# migration file, not as an ALTER appended here — run `python -m app.migrate`
# on an existing local DB.


async def _pool() -> asyncpg.Pool:
    global _schema_ready
    pool = await db.get_pool()
    if not _schema_ready:
        async with pool.acquire() as conn:
            # Created in dev/test, only verified in production — see
            # app/schema_guard.py.
            if schema_guard.auto_create_enabled():
                await conn.execute(_SCHEMA_SQL)
            else:
                await schema_guard.verify_tables(conn, ("jobs", "pages", "products"))
        _schema_ready = True
    return pool


def _row_to_job(row: asyncpg.Record) -> JobState:
    return JobState(
        job_id=row["job_id"],
        url=row["url"],
        tenant_id=row["tenant_id"],
        status=row["status"],
        completed_pages=row["completed_pages"],
        total_pages=row["total_pages"],
        error=row["error"],
        persisted=row["persisted"],
        ingest_status=row["ingest_status"],
        ingested_chunks=row["ingested_chunks"],
        ingested_questions=row["ingested_questions"],
        ingest_error=row["ingest_error"],
        created_at=row["created_at"].isoformat() if row["created_at"] else None,
        updated_at=row["updated_at"].isoformat() if row["updated_at"] else None,
    )


async def create_job(job_id: str, url: str, tenant_id: str) -> JobState:
    pool = await _pool()
    row = await pool.fetchrow(
        """
        INSERT INTO jobs (job_id, url, tenant_id, status)
        VALUES ($1, $2, $3, 'scraping')
        RETURNING *
        """,
        job_id,
        url,
        tenant_id,
    )
    return _row_to_job(row)


async def save_job(state: JobState) -> None:
    pool = await _pool()
    row = await pool.fetchrow(
        """
        UPDATE jobs SET
          status = $2, completed_pages = $3, total_pages = $4, error = $5,
          persisted = $6, ingest_status = $7, ingested_chunks = $8,
          ingested_questions = $9, ingest_error = $10, updated_at = now()
        WHERE job_id = $1
        RETURNING updated_at
        """,
        state.job_id,
        state.status,
        state.completed_pages,
        state.total_pages,
        state.error,
        state.persisted,
        state.ingest_status,
        state.ingested_chunks,
        state.ingested_questions,
        state.ingest_error,
    )
    if row:
        state.updated_at = row["updated_at"].isoformat()


async def load_job(job_id: str) -> JobState | None:
    pool = await _pool()
    row = await pool.fetchrow("SELECT * FROM jobs WHERE job_id = $1", job_id)
    return _row_to_job(row) if row else None


async def reconcile_interrupted_jobs() -> int:
    """Startup-only. A process restart mid-ingest leaves rows stuck at
    ingest_status='ingesting' with no background task left to finish them.
    Flags those so callers see a real status instead of a silent hang;
    POST /ingest/{job_id} retries. Returns the number of rows affected.
    """
    pool = await _pool()
    result = await pool.execute(
        """
        UPDATE jobs SET ingest_status = 'ingest_failed',
                        ingest_error = 'interrupted by server restart',
                        updated_at = now()
        WHERE ingest_status = 'ingesting'
        """
    )
    return int(result.split()[-1])


async def list_open_jobs() -> list[JobState]:
    """Jobs the background poller should still check on Firecrawl: not yet
    persisted and not already terminal locally (Firecrawl won't change a
    failed/cancelled crawl, so repolling those is wasted work)."""
    pool = await _pool()
    rows = await pool.fetch(
        "SELECT * FROM jobs WHERE persisted = false AND status NOT IN ('failed', 'cancelled')"
    )
    return [_row_to_job(row) for row in rows]


async def mark_persisting(job_id: str, total_pages: int) -> JobState | None:
    """Atomically claim the scraping->persisted transition for a job.

    Returns the claimed row if this call won the race, or None if another
    caller — in this process or a different replica — already claimed it.
    This is the cross-replica-safe replacement for an in-process lock: the
    UPDATE's WHERE clause is only satisfied for the one caller that runs it
    first, no matter how many processes are polling the same job_id.
    """
    pool = await _pool()
    row = await pool.fetchrow(
        """
        UPDATE jobs SET persisted = true, status = 'completed',
                        total_pages = $2, updated_at = now()
        WHERE job_id = $1 AND persisted = false
        RETURNING *
        """,
        job_id,
        total_pages,
    )
    return _row_to_job(row) if row else None


async def revert_persisting(job_id: str, error: str) -> None:
    """Roll back a claim if persisting pages failed after mark_persisting
    succeeded, so the next poll retries instead of getting stuck at
    persisted=true with no pages ever written.
    """
    pool = await _pool()
    await pool.execute(
        "UPDATE jobs SET persisted = false, error = $2, updated_at = now() WHERE job_id = $1",
        job_id,
        error,
    )


def _resolve_products_and_actions(
    tenant_id: str, site_url: str, page: Page
) -> tuple[list[Product], list[PageAction]]:
    """Assigns each of this page's products its persisted product_key (see
    app/products.py::make_product_key's docstring for why this can't happen
    at extraction time), and — only when the page has *exactly one*
    product — associates every one of its page_actions with that product.
    A page with zero or multiple products leaves page_actions.product_key
    unset; see app/actions.py's module docstring for the documented
    multi-product-page limitation this reflects.
    """
    products = [
        product.model_copy(
            update={
                "product_key": make_product_key(
                    tenant_id, site_url, page.url, product.sku or product.name
                )
            }
        )
        for product in page.products
    ]
    if len(products) == 1:
        page_actions = [
            action.model_copy(update={"product_key": products[0].product_key})
            for action in page.page_actions
        ]
    else:
        page_actions = list(page.page_actions)
    return products, page_actions


async def persist_pages(
    job_id: str, tenant_id: str, site_url: str, pages: list[Page]
) -> list[PageSummary]:
    """Replace all pages for a job (DELETE + INSERT in one transaction, so
    re-running with a different page set never leaves stale rows from a
    previous crawl of the same job behind), plus this site's products (a
    separate table, replaced via replace_site_products below — same
    tenant+site replace scope app/ingestion/store.py::replace_site_chunks
    already uses for chunks, since a re-crawl's product set entirely
    supersedes the last one).

    `tenant_id`/`site_url` are needed here (not just job_id) because
    products/page_actions need them to resolve product_key — see
    _resolve_products_and_actions.
    """
    pool = await _pool()
    rows = []
    summaries: list[PageSummary] = []
    all_products: list[Product] = []
    for i, page in enumerate(pages, start=1):
        chars = len(page.markdown)
        sections_payload = (
            [s.model_dump() for s in page.sections] if page.sections else None
        )
        links_payload = [l.model_dump() for l in page.links] if page.links else None
        products, page_actions = _resolve_products_and_actions(tenant_id, site_url, page)
        all_products.extend(products)
        page_actions_payload = (
            [a.model_dump() for a in page_actions] if page_actions else None
        )
        rows.append(
            (
                job_id,
                i,
                page.url,
                page.title,
                page.description,
                page.markdown,
                chars,
                len(page.sections),
                sections_payload,
                links_payload,
                page_key(page.url) or page.url,
                page_actions_payload,
            )
        )
        summaries.append(
            PageSummary(
                url=page.url,
                title=page.title,
                description=page.description,
                chars=chars,
                section_count=len(page.sections),
            )
        )

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM pages WHERE job_id = $1", job_id)
            if rows:
                await conn.executemany(
                    """
                    INSERT INTO pages (job_id, page_index, url, title, description,
                                       markdown, chars, section_count, sections,
                                       links, url_key, page_actions)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                    """,
                    rows,
                )

    await replace_site_products(tenant_id, site_url, job_id, all_products)
    return summaries


async def replace_site_products(
    tenant_id: str, site_url: str, job_id: str, products: list[Product]
) -> int:
    """Replace all deterministically-extracted (JSON-LD/OG,
    app/products.py) products for (tenant_id, site_url) — a re-crawl's
    product set entirely supersedes the last one, same
    delete+insert-in-one-transaction shape persist_pages already uses for
    pages and replace_site_chunks uses for chunks.

    Embeddings are computed here, not by the caller: unlike chunk embedding
    (deferred to the async ingest_job step alongside doc2query), this is a
    local, non-network fastembed call — no different in kind from any other
    pure computation already living in this module (e.g. app/links.py's
    page_key, imported above). asyncio.to_thread mirrors how /query already
    wraps the same blocking call.

    The LLM-fallback enrichment path (app/ingestion/product_enrichment.py,
    for a page with neither JSON-LD nor OG signal) adds rows separately via
    upsert_products, so it never wipes what this function just wrote.
    """
    pool = await _pool()
    rows = []
    if products:
        texts = [
            f"{p.name} — {p.description}" if p.description else p.name for p in products
        ]
        embeddings = await asyncio.to_thread(embed_documents, texts)
        for product, embedding in zip(products, embeddings):
            rows.append(
                (
                    product.product_key,
                    tenant_id,
                    job_id,
                    site_url,
                    product.page_url,
                    product.name,
                    product.sku,
                    product.price_amount,
                    product.price_currency,
                    product.availability,
                    product.image_url,
                    product.description,
                    product.attributes or None,
                    product.source,
                    embedding,
                )
            )

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM products WHERE tenant_id = $1 AND site_url = $2",
                tenant_id,
                site_url,
            )
            if rows:
                await conn.executemany(
                    """
                    INSERT INTO products (product_key, tenant_id, job_id, site_url, page_url,
                                           name, sku, price_amount, price_currency, availability,
                                           image_url, description, attributes, source, embedding)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                    """,
                    rows,
                )
    return len(rows)


async def upsert_products(
    tenant_id: str, site_url: str, job_id: str, products: list[Product]
) -> int:
    """Adds (or refreshes) rows without touching anything else for the
    tenant+site — used by app/ingestion/product_enrichment.py's LLM fallback
    to add products for a page that had neither JSON-LD nor OG signal,
    without wiping the deterministic set replace_site_products just wrote.
    Callers are responsible for having already resolved product_key
    (app/products.py::make_product_key) on each Product, same as
    replace_site_products expects.

    ON CONFLICT (product_key) DO UPDATE rather than a plain INSERT: a later
    re-ingest of the same page produces the same product_key (a stable hash
    of tenant_id|site_url|page_url|sku-or-name), so this stays idempotent
    across repeated runs instead of failing on the primary key.
    """
    pool = await _pool()
    if not products:
        return 0
    texts = [f"{p.name} — {p.description}" if p.description else p.name for p in products]
    embeddings = await asyncio.to_thread(embed_documents, texts)
    rows = [
        (
            product.product_key,
            tenant_id,
            job_id,
            site_url,
            product.page_url,
            product.name,
            product.sku,
            product.price_amount,
            product.price_currency,
            product.availability,
            product.image_url,
            product.description,
            product.attributes or None,
            product.source,
            embedding,
        )
        for product, embedding in zip(products, embeddings)
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO products (product_key, tenant_id, job_id, site_url, page_url,
                                   name, sku, price_amount, price_currency, availability,
                                   image_url, description, attributes, source, embedding)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
            ON CONFLICT (product_key) DO UPDATE SET
                job_id = EXCLUDED.job_id,
                name = EXCLUDED.name,
                sku = EXCLUDED.sku,
                price_amount = EXCLUDED.price_amount,
                price_currency = EXCLUDED.price_currency,
                availability = EXCLUDED.availability,
                image_url = EXCLUDED.image_url,
                description = EXCLUDED.description,
                attributes = EXCLUDED.attributes,
                source = EXCLUDED.source,
                embedding = EXCLUDED.embedding
            """,
            rows,
        )
    return len(rows)


async def load_page_links(
    tenant_id: str, page_url_key: str, site_url: str | None = None
) -> PageLinksRecord | None:
    """Links captured from one page of the tenant's most recent completed
    crawl. `page_url_key` is a normalized key (app/links.py::page_key), not
    a raw URL — comparison semantics are defined in exactly one place there
    and mirrored in voice_runtime/urls.py and widget/src/navigation.ts.

    site_url omitted => the tenant's most recently started crawl, the same
    one-site-per-tenant fallback app/salescript/store.py::load_current uses
    (and the only mode voice_runtime can call — it never learns a site_url
    from LiveKit dispatch metadata).

    One JOIN rather than a two-step "resolve job_id, then read page": a
    two-step read would race a re-crawl's persist_pages DELETE+INSERT
    transaction landing in between the two queries.
    """
    pool = await _pool()
    row = await pool.fetchrow(
        """
        SELECT j.job_id AS job_id, j.url AS site_url, p.url AS page_url, p.links
        FROM pages p
        JOIN jobs j ON j.job_id = p.job_id
        WHERE j.tenant_id = $1
          AND j.persisted = true
          AND p.url_key = $2
          AND ($3::text IS NULL OR j.url = $3)
        ORDER BY j.created_at DESC
        LIMIT 1
        """,
        tenant_id,
        page_url_key,
        site_url,
    )
    if row is None:
        return None
    return PageLinksRecord(
        job_id=row["job_id"],
        site_url=row["site_url"],
        page_url=row["page_url"],
        links=[PageLink(**item) for item in (row["links"] or [])],
    )


async def load_page_actions(
    tenant_id: str, page_url_key: str, site_url: str | None = None
) -> PageActionsRecord | None:
    """Clickable controls captured from one page of the tenant's most recent
    completed crawl — mirrors load_page_links exactly (same JOIN, same
    most-recent-crawl fallback, same race-avoidance reasoning), reading
    page_actions instead of links."""
    pool = await _pool()
    row = await pool.fetchrow(
        """
        SELECT j.job_id AS job_id, j.url AS site_url, p.url AS page_url, p.page_actions
        FROM pages p
        JOIN jobs j ON j.job_id = p.job_id
        WHERE j.tenant_id = $1
          AND j.persisted = true
          AND p.url_key = $2
          AND ($3::text IS NULL OR j.url = $3)
        ORDER BY j.created_at DESC
        LIMIT 1
        """,
        tenant_id,
        page_url_key,
        site_url,
    )
    if row is None:
        return None
    return PageActionsRecord(
        job_id=row["job_id"],
        site_url=row["site_url"],
        page_url=row["page_url"],
        page_actions=[PageAction(**item) for item in (row["page_actions"] or [])],
    )


def _row_to_product(row: asyncpg.Record) -> Product:
    return Product(
        product_key=row["product_key"],
        page_url=row["page_url"],
        name=row["name"],
        sku=row["sku"],
        price_amount=row["price_amount"],
        price_currency=row["price_currency"],
        availability=row["availability"],
        image_url=row["image_url"],
        description=row["description"],
        attributes=row["attributes"] or {},
        source=row["source"],
    )


async def search_products(
    tenant_id: str,
    query_embedding: list[float] | None = None,
    price_min=None,
    price_max=None,
    availability: str | None = None,
    limit: int = 20,
) -> list[Product]:
    """Catalog search backing GET /products/{tenant_id} — "suggest
    alternatives" and grounding "what's in stock" in real data. Omitting
    query_embedding returns a plain listing (most recently crawled first),
    filtered by the same price/availability predicates. The simpler cousin
    of app/ingestion/store.py::search: no hybrid/lexical fusion needed for a
    catalog this size, just vector similarity plus SQL filters.
    """
    pool = await _pool()
    conditions = ["tenant_id = $1"]
    params: list = [tenant_id]
    if price_min is not None:
        params.append(price_min)
        conditions.append(f"price_amount >= ${len(params)}")
    if price_max is not None:
        params.append(price_max)
        conditions.append(f"price_amount <= ${len(params)}")
    if availability is not None:
        params.append(availability)
        conditions.append(f"availability = ${len(params)}")
    where_clause = " AND ".join(conditions)

    cols = (
        "product_key, page_url, name, sku, price_amount, price_currency, "
        "availability, image_url, description, attributes, source"
    )
    if query_embedding is not None:
        params.append(query_embedding)
        embedding_idx = len(params)
        params.append(limit)
        limit_idx = len(params)
        rows = await pool.fetch(
            f"""
            SELECT {cols}
            FROM products
            WHERE {where_clause} AND embedding IS NOT NULL
            ORDER BY embedding <=> ${embedding_idx}
            LIMIT ${limit_idx}
            """,
            *params,
        )
    else:
        params.append(limit)
        limit_idx = len(params)
        rows = await pool.fetch(
            f"""
            SELECT {cols}
            FROM products
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${limit_idx}
            """,
            *params,
        )
    return [_row_to_product(row) for row in rows]


async def load_pages(job_id: str, include_content: bool = False) -> JobPages | None:
    pool = await _pool()
    job_row = await pool.fetchrow("SELECT url FROM jobs WHERE job_id = $1", job_id)
    if job_row is None:
        return None

    # description is small (like title) so it's always selected, not gated
    # behind include_content like markdown/sections — the ingestion path
    # (include_content=True) needs it on Page to fold into the first chunk's
    # embedding_text (see ingestion/service.py::_to_page).
    cols = "url, title, description, chars, section_count"
    if include_content:
        cols += ", markdown, sections"
    rows = await pool.fetch(
        f"SELECT {cols} FROM pages WHERE job_id = $1 ORDER BY page_index", job_id
    )

    summaries = []
    for row in rows:
        summary = PageSummary(
            url=row["url"],
            title=row["title"],
            description=row["description"],
            chars=row["chars"],
            section_count=row["section_count"],
        )
        if include_content:
            summary.markdown = row["markdown"]
            if row["sections"]:
                summary.sections = [Section(**item) for item in row["sections"]]
        summaries.append(summary)

    return JobPages(
        job_id=job_id, url=job_row["url"], page_count=len(summaries), pages=summaries
    )
