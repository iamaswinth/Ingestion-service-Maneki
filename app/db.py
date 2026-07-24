"""Shared asyncpg connection pool.

Both `app/ingestion/store.py` (the `chunks` table) and `app/storage.py` (the
`jobs`/`pages` tables) point at the same Postgres database, so they share one
pool from here rather than each opening their own. Each of those modules
still owns and applies its own schema SQL lazily on first use.
"""

import json
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

from .config import settings

_pool: Optional[asyncpg.Pool] = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)
    # Transparent jsonb <-> Python list/dict roundtrip (used by pages.sections)
    # so callers pass/receive plain Python objects, not raw JSON strings.
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
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
