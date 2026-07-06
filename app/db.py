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
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await register_vector(conn)
    # Transparent jsonb <-> Python list/dict roundtrip (used by pages.sections)
    # so callers pass/receive plain Python objects, not raw JSON strings.
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.database_url, init=_init_connection, min_size=1, max_size=5
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
