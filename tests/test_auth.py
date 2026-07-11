"""Tests app/auth.py's internal-service-token gate — the boundary that
closes the direct-to-ingestion vulnerability now that the API gateway and
voice runtime are the only sanctioned callers.

Scoped to the auth layer itself: "a valid token passes the gate" is
asserted as "not 401", not "the full handler succeeds" — a real /query or
/scrape call still needs live Postgres/Firecrawl/embedding infra that this
test environment doesn't have, and that's out of scope for a test about
authentication, not business logic.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app import config as config_module
from app.main import app

TOKEN = "test-internal-service-token"


@pytest.fixture(autouse=True)
def patch_internal_token(monkeypatch):
    monkeypatch.setattr(config_module.settings, "internal_service_token", TOKEN)


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health_stays_open_without_a_token():
    async with await _client() as client:
        resp = await client.get("/health")
    assert resp.status_code == 200


async def test_query_rejects_missing_token():
    async with await _client() as client:
        resp = await client.post("/query", json={"tenant_id": "acme", "question": "hi"})
    assert resp.status_code == 401


async def test_query_rejects_wrong_token():
    async with await _client() as client:
        resp = await client.post(
            "/query",
            json={"tenant_id": "acme", "question": "hi"},
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert resp.status_code == 401


async def test_scrape_rejects_missing_token():
    async with await _client() as client:
        resp = await client.post(
            "/scrape", json={"url": "https://example.com", "tenant_id": "acme"}
        )
    assert resp.status_code == 401


async def test_approve_sales_script_rejects_missing_token():
    async with await _client() as client:
        resp = await client.post("/sales-script/acme/approve")
    assert resp.status_code == 401


async def test_approve_sales_script_rejects_wrong_token():
    async with await _client() as client:
        resp = await client.post(
            "/sales-script/acme/approve", headers={"Authorization": "Bearer nope"}
        )
    assert resp.status_code == 401


async def test_valid_token_passes_the_auth_gate():
    # /map's handler wraps downstream (Firecrawl) failures in a 502 without
    # touching Postgres or the embedding model — the lightest protected
    # endpoint to prove a valid token clears app/auth.py specifically.
    async with await _client() as client:
        resp = await client.get(
            "/map",
            params={"url": "https://example.com"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
    assert resp.status_code != 401


async def test_unset_internal_token_fails_closed(monkeypatch):
    monkeypatch.setattr(config_module.settings, "internal_service_token", "")
    async with await _client() as client:
        resp = await client.post(
            "/query",
            json={"tenant_id": "acme", "question": "hi"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
    assert resp.status_code == 401
