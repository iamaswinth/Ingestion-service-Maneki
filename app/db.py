"""Shared asyncpg connection pool.

Both `app/ingestion/store.py` (the `chunks` table) and `app/storage.py` (the
`jobs`/`pages` tables) point at the same Postgres database, so they share one
pool from here rather than each opening their own. Each of those modules
still owns and applies its own schema SQL lazily on first use.
"""

import json
import logging
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

from .config import settings

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)
    # Transparent jsonb <-> Python list/dict roundtrip (used by pages.sections)
    # so callers pass/receive plain Python objects, not raw JSON strings.
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )

    # HNSW query-time recall knobs, set per session (not index-wide) so they
    # apply to every query on this connection without touching the index
    # itself. Guarded: an older pgvector (<0.8, no iterative_scan) or a
    # locked-down managed Postgres that rejects SET on these GUCs should
    # degrade to pgvector's stock defaults, not fail the connection.
    try:
        await conn.execute(f"SET hnsw.ef_search = {int(settings.hnsw_ef_search)}")
    except Exception:
        logger.warning("could not set hnsw.ef_search; using pgvector's default", exc_info=True)
    try:
        # 'relaxed_order' keeps the HNSW scan walking the graph until enough
        # rows survive the query's WHERE filter (here, tenant_id) instead of
        # stopping at ef_search candidates scanned globally and then applying
        # the filter — without it, a small tenant's rows can simply not be
        # among the top-N nearest neighbors *before* filtering and a query
        # returns too few or zero hits despite the tenant's data being
        # present. Safe to use here: every caller of this connection re-sorts
        # its candidate set downstream (RRF fusion, cross-encoder rerank, or
        # a plain ORDER BY), so relaxed (non-exact) ordering from the index
        # scan itself is never depended on directly.
        await conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")
    except Exception:
        logger.warning(
            "could not set hnsw.iterative_scan (pgvector <0.8?); filtered queries "
            "on a multi-tenant table may under-return",
            exc_info=True,
        )


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        # register_vector() in _init_connection needs the `vector` type to
        # already exist in pg_catalog, and _init_connection runs during
        # create_pool() below (for the pool's initial min_size connections) —
        # before any schema SQL elsewhere would get a chance to create the
        # extension. Bootstrap it once here, via a throwaway connection,
        # rather than repeating the (idempotent but unnecessary) CREATE
        # EXTENSION on every pooled connection.
        bootstrap_conn = await asyncpg.connect(settings.database_url)
        try:
            await bootstrap_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await bootstrap_conn.close()
        _pool = await asyncpg.create_pool(
            settings.database_url, init=_init_connection, min_size=1, max_size=5
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
