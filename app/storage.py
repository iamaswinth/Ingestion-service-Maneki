"""Postgres-backed persistence for jobs and scraped pages.

Two tables:

    jobs   one row per crawl — JobState, keyed by job_id
    pages  one row per scraped page, keyed by (job_id, page_index)

Living in Postgres (rather than on local disk) means multiple uvicorn
workers or container replicas share state with no shared volume, and the
scraping->persisted transition can be claimed atomically across all of them
(see `mark_persisting`) instead of relying on an in-process lock.
"""

import asyncpg

from . import db, schema_guard
from .models import JobPages, JobState, Page, PageSummary, Section

_schema_ready = False

_SCHEMA_SQL = """
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
  created_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (job_id, page_index)
);

CREATE INDEX IF NOT EXISTS pages_job_idx ON pages (job_id);
"""


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
                await schema_guard.verify_tables(conn, ("jobs", "pages"))
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


async def persist_pages(job_id: str, pages: list[Page]) -> list[PageSummary]:
    """Replace all pages for a job. Idempotent — DELETE + INSERT in one
    transaction, so re-running with a different page set never leaves stale
    rows from a previous crawl of the same job behind.
    """
    pool = await _pool()
    rows = []
    summaries: list[PageSummary] = []
    for i, page in enumerate(pages, start=1):
        chars = len(page.markdown)
        sections_payload = (
            [s.model_dump() for s in page.sections] if page.sections else None
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
                                       markdown, chars, section_count, sections)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    """,
                    rows,
                )
    return summaries


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
