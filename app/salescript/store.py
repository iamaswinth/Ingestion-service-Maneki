"""Postgres persistence for generated sales scripts.

One row per (tenant_id, site_url) — the *current* script for that site, not a
history. Mirrors app/storage.py's lazy-schema + atomic-claim idioms so
concurrent `POST /sales-script/{job_id}` calls for the same site are safe
without an in-process lock (see claim_generation).
"""

from typing import Optional

import asyncpg

from .. import db
from ..models import SalesScriptCritique, SalesScriptRecord

_schema_ready = False

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sales_scripts (
  id             BIGSERIAL PRIMARY KEY,
  tenant_id      TEXT NOT NULL,
  site_url       TEXT NOT NULL,
  job_id         TEXT NOT NULL,
  status         TEXT NOT NULL DEFAULT 'not_started',
  script         JSONB,
  critique       JSONB,
  revision_count INT NOT NULL DEFAULT 0,
  error          TEXT,
  created_at     timestamptz NOT NULL DEFAULT now(),
  updated_at     timestamptz NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, site_url)
);

CREATE INDEX IF NOT EXISTS sales_scripts_tenant_idx ON sales_scripts (tenant_id);
CREATE INDEX IF NOT EXISTS sales_scripts_status_idx ON sales_scripts (status);
"""


async def _pool() -> asyncpg.Pool:
    global _schema_ready
    pool = await db.get_pool()
    if not _schema_ready:
        async with pool.acquire() as conn:
            await conn.execute(_SCHEMA_SQL)
        _schema_ready = True
    return pool


def _row_to_record(row: asyncpg.Record) -> SalesScriptRecord:
    return SalesScriptRecord(
        tenant_id=row["tenant_id"],
        site_url=row["site_url"],
        job_id=row["job_id"],
        status=row["status"],
        script=row["script"],
        critique=SalesScriptCritique(**row["critique"]) if row["critique"] else None,
        revision_count=row["revision_count"],
        error=row["error"],
        created_at=row["created_at"].isoformat() if row["created_at"] else None,
        updated_at=row["updated_at"].isoformat() if row["updated_at"] else None,
    )


async def claim_generation(
    tenant_id: str, site_url: str, job_id: str
) -> Optional[SalesScriptRecord]:
    """Atomically claim the right to run generation for this site.

    Returns the claimed row, or None if another caller already holds
    status='generating' — the race-safe idiom mirrors
    app/storage.py::mark_persisting's `UPDATE ... WHERE ... RETURNING *`.
    """
    pool = await _pool()
    row = await pool.fetchrow(
        """
        INSERT INTO sales_scripts (tenant_id, site_url, job_id, status)
        VALUES ($1, $2, $3, 'generating')
        ON CONFLICT (tenant_id, site_url) DO UPDATE SET
            job_id = EXCLUDED.job_id, status = 'generating',
            error = NULL, updated_at = now()
        WHERE sales_scripts.status != 'generating'
        RETURNING *
        """,
        tenant_id,
        site_url,
        job_id,
    )
    return _row_to_record(row) if row else None


async def save_result(
    tenant_id: str,
    site_url: str,
    script: dict,
    critique: Optional[dict],
    revision_count: int,
) -> None:
    pool = await _pool()
    await pool.execute(
        """
        UPDATE sales_scripts SET
            status = 'pending_review', script = $3, critique = $4,
            revision_count = $5, error = NULL, updated_at = now()
        WHERE tenant_id = $1 AND site_url = $2
        """,
        tenant_id,
        site_url,
        script,
        critique,
        revision_count,
    )


async def save_failure(tenant_id: str, site_url: str, error: str) -> None:
    pool = await _pool()
    await pool.execute(
        """
        UPDATE sales_scripts SET status = 'failed', error = $3, updated_at = now()
        WHERE tenant_id = $1 AND site_url = $2
        """,
        tenant_id,
        site_url,
        error,
    )


async def load_current(
    tenant_id: str, site_url: Optional[str] = None
) -> Optional[SalesScriptRecord]:
    """The current row for a site, or (if site_url is omitted) the
    most-recently-updated row for the tenant — the one-site-per-tenant
    assumption this API is built around."""
    pool = await _pool()
    if site_url:
        row = await pool.fetchrow(
            "SELECT * FROM sales_scripts WHERE tenant_id = $1 AND site_url = $2",
            tenant_id,
            site_url,
        )
    else:
        row = await pool.fetchrow(
            """
            SELECT * FROM sales_scripts WHERE tenant_id = $1
            ORDER BY updated_at DESC LIMIT 1
            """,
            tenant_id,
        )
    return _row_to_record(row) if row else None


async def approve(tenant_id: str, site_url: str) -> Optional[SalesScriptRecord]:
    """Flip pending_review -> ready. Returns None if there's no such row or
    it isn't in pending_review — the caller disambiguates via load_current."""
    pool = await _pool()
    row = await pool.fetchrow(
        """
        UPDATE sales_scripts SET status = 'ready', updated_at = now()
        WHERE tenant_id = $1 AND site_url = $2 AND status = 'pending_review'
        RETURNING *
        """,
        tenant_id,
        site_url,
    )
    return _row_to_record(row) if row else None
