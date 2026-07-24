import uuid

import pytest
import pytest_asyncio

from app import db
from app.ingestion import store as ingestion_store


@pytest_asyncio.fixture
async def pg_tenant():
    """A unique (tenant_id, site_url) pair for one test, with automatic
    chunk cleanup. Skips the test if Postgres isn't reachable — CI provides
    one via a service container (.github/workflows/ci.yml); local runs
    without the pgvector container up degrade to a skip, not a failure.

    app/db.py's pool is a single module-level global bound to the event
    loop it was created on. asyncpg connections/pools are not safe to reuse
    across event loops, so this suite pins one shared loop for the whole
    session (asyncio_default_fixture_loop_scope/test_loop_scope = session
    in pytest.ini) rather than churning a fresh pool per test — recreating
    the pool on every function-scoped loop hit a real Windows ProactorEventLoop
    + asyncpg incompatibility (stale callbacks firing against an already-closed
    loop). The pool itself is closed once at session end, below.
    """
    tenant_id = f"test-{uuid.uuid4().hex[:12]}"
    site_url = "https://test.example.com"
    try:
        await ingestion_store.get_pool()
    except Exception as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    yield tenant_id, site_url
    await ingestion_store.delete_site_chunks(tenant_id, site_url)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _close_pool_at_session_end():
    yield
    await db.close_pool()
