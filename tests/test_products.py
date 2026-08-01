"""Tests for app/products.py (extract_products, make_product_key), the
storage-layer functions that back GET /products/{tenant_id}
(app/storage.py::replace_site_products / search_products / upsert_products),
and app/ingestion/product_enrichment.py's LLM fallback.

Pure-function tests need no I/O; DB tests use the pg_tenant fixture
(tests/conftest.py), which skips cleanly when Postgres isn't reachable.
Mirrors tests/test_page_links.py's structure.
"""

import asyncio
import json
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from app import storage
from app.ingestion.product_enrichment import (
    _extract_one,
    _looks_like_a_product_page,
    enrich_products,
)
from app.models import JobPages, PageSummary, Product
from app.products import extract_products, make_product_key

# ---- extract_products: JSON-LD ----------------------------------------------


def test_extracts_single_jsonld_product():
    html = """
    <script type="application/ld+json">
    {"@context":"https://schema.org","@type":"Product","name":"Blue Widget","sku":"BW-1",
     "offers":{"@type":"Offer","price":"49.99","priceCurrency":"USD","availability":"https://schema.org/InStock"},
     "image":"https://example.com/blue.jpg","description":"A very blue widget."}
    </script>
    """
    products = extract_products(html, "https://example.com/blue-widget")
    assert len(products) == 1
    p = products[0]
    assert p.name == "Blue Widget"
    assert p.sku == "BW-1"
    assert p.price_amount == Decimal("49.99")
    assert p.price_currency == "USD"
    assert p.availability == "https://schema.org/InStock"
    assert p.image_url == "https://example.com/blue.jpg"
    assert p.description == "A very blue widget."
    assert p.source == "jsonld"
    assert p.product_key is None  # resolved later, at persistence time


def test_extracts_product_from_a_list_of_jsonld_objects():
    html = """
    <script type="application/ld+json">
    [{"@type":"BreadcrumbList"}, {"@type":"Product","name":"Red Widget"}]
    </script>
    """
    products = extract_products(html, "https://example.com/red")
    assert [p.name for p in products] == ["Red Widget"]


def test_extracts_product_from_an_at_graph_wrapper():
    html = """
    <script type="application/ld+json">
    {"@graph":[{"@type":"WebPage"},{"@type":"Product","name":"Green Widget","offers":{"price":19.5,"priceCurrency":"USD"}}]}
    </script>
    """
    products = extract_products(html, "https://example.com/green")
    assert len(products) == 1
    assert products[0].name == "Green Widget"
    assert products[0].price_amount == Decimal("19.5")


def test_multiple_products_on_one_page_are_all_captured():
    html = """
    <script type="application/ld+json">
    [{"@type":"Product","name":"A"},{"@type":"Product","name":"B"}]
    </script>
    """
    products = extract_products(html, "https://example.com/category")
    assert {p.name for p in products} == {"A", "B"}


def test_product_without_a_name_is_dropped():
    html = '<script type="application/ld+json">{"@type":"Product","sku":"X"}</script>'
    assert extract_products(html, "https://example.com/x") == []


def test_offer_as_a_list_uses_the_first_offer():
    html = """
    <script type="application/ld+json">
    {"@type":"Product","name":"Multi-offer","offers":[{"price":"10.00","priceCurrency":"USD"},
                                                        {"price":"12.00","priceCurrency":"EUR"}]}
    </script>
    """
    products = extract_products(html, "https://example.com/multi")
    assert products[0].price_amount == Decimal("10.00")
    assert products[0].price_currency == "USD"


def test_missing_price_degrades_to_none_not_an_exception():
    html = '<script type="application/ld+json">{"@type":"Product","name":"No price"}</script>'
    products = extract_products(html, "https://example.com/x")
    assert products[0].price_amount is None


def test_non_numeric_price_degrades_to_none():
    html = """
    <script type="application/ld+json">
    {"@type":"Product","name":"Contact us","offers":{"price":"Contact us for pricing"}}
    </script>
    """
    products = extract_products(html, "https://example.com/x")
    assert products[0].price_amount is None


def test_malformed_jsonld_does_not_raise():
    html = '<script type="application/ld+json">{not valid json}</script>'
    assert extract_products(html, "https://example.com/x") == []


def test_non_product_type_is_ignored():
    html = '<script type="application/ld+json">{"@type":"Organization","name":"Acme Inc"}</script>'
    assert extract_products(html, "https://example.com/x") == []


def test_image_as_object_extracts_the_url():
    html = """
    <script type="application/ld+json">
    {"@type":"Product","name":"Imaged","image":{"@type":"ImageObject","url":"https://x.com/i.jpg"}}
    </script>
    """
    products = extract_products(html, "https://example.com/x")
    assert products[0].image_url == "https://x.com/i.jpg"


# ---- extract_products: Open Graph fallback ----------------------------------


def test_og_product_tags_used_when_no_jsonld():
    html = """
    <meta property="og:type" content="product">
    <meta property="og:title" content="Green Widget">
    <meta property="product:price:amount" content="9.99">
    <meta property="product:price:currency" content="USD">
    <meta property="og:image" content="https://x.com/g.jpg">
    """
    products = extract_products(html, "https://example.com/green")
    assert len(products) == 1
    p = products[0]
    assert p.name == "Green Widget"
    assert p.price_amount == Decimal("9.99")
    assert p.price_currency == "USD"
    assert p.image_url == "https://x.com/g.jpg"
    assert p.source == "og"


def test_og_tags_ignored_when_jsonld_already_found_something():
    html = """
    <script type="application/ld+json">{"@type":"Product","name":"From JSON-LD"}</script>
    <meta property="og:type" content="product">
    <meta property="og:title" content="Should not appear">
    """
    products = extract_products(html, "https://example.com/x")
    assert [p.name for p in products] == ["From JSON-LD"]


def test_og_type_not_product_yields_nothing():
    html = '<meta property="og:type" content="website"><meta property="og:title" content="Home">'
    assert extract_products(html, "https://example.com/x") == []


def test_og_product_without_title_yields_nothing():
    html = '<meta property="og:type" content="product">'
    assert extract_products(html, "https://example.com/x") == []


# ---- extract_products: malformed/empty input --------------------------------


def test_extract_products_empty_or_none_html_returns_empty():
    assert extract_products("", "https://x.com/x") == []
    assert extract_products(None, "https://x.com/x") == []


def test_extract_products_caps_at_max_products_per_page():
    from app.products import MAX_PRODUCTS_PER_PAGE

    objs = [{"@type": "Product", "name": f"P{i}"} for i in range(80)]
    html = f'<script type="application/ld+json">{json.dumps(objs)}</script>'
    products = extract_products(html, "https://example.com/x")
    assert len(products) == MAX_PRODUCTS_PER_PAGE


# ---- make_product_key -------------------------------------------------------


def test_make_product_key_is_deterministic():
    key1 = make_product_key("tenant-a", "https://x.com", "https://x.com/p", "SKU-1")
    key2 = make_product_key("tenant-a", "https://x.com", "https://x.com/p", "SKU-1")
    assert key1 == key2


def test_make_product_key_differs_by_site_url():
    # Same tenant/page/sku, different site_url (two seeds of the same
    # tenant) must not collide.
    key1 = make_product_key("tenant-a", "https://x.com", "https://x.com/p", "SKU-1")
    key2 = make_product_key("tenant-a", "https://x.com/docs", "https://x.com/p", "SKU-1")
    assert key1 != key2


# ---- product_enrichment: the price-signal gate ------------------------------


def test_looks_like_a_product_page_true_for_price_like_text():
    assert _looks_like_a_product_page("Buy now for $49.99") is True


def test_looks_like_a_product_page_false_for_generic_text():
    assert _looks_like_a_product_page("About our team and mission") is False


# ---- product_enrichment: _extract_one ---------------------------------------


class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, data):
        self.content = [_FakeTextBlock(json.dumps(data) if not isinstance(data, str) else data)]


async def test_extract_one_returns_a_product_on_success():
    with patch("app.ingestion.product_enrichment._get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _FakeResponse(
            {
                "is_product_page": True,
                "name": "Green Widget",
                "price_amount": 29.99,
                "price_currency": "USD",
                "description": "A fine widget.",
            }
        )
        product = await _extract_one(
            "https://x.com/green", "Buy the green widget for $29.99", asyncio.Semaphore(1)
        )
        assert product is not None
        assert product.name == "Green Widget"
        assert product.source == "llm"


async def test_extract_one_returns_none_for_a_non_product_page():
    with patch("app.ingestion.product_enrichment._get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _FakeResponse(
            {
                "is_product_page": False,
                "name": "",
                "price_amount": 0,
                "price_currency": "",
                "description": "",
            }
        )
        result = await _extract_one("https://x.com/about", "About us", asyncio.Semaphore(1))
        assert result is None


async def test_extract_one_returns_none_on_a_non_dict_json_payload():
    with patch("app.ingestion.product_enrichment._get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _FakeResponse("just a bare string")
        result = await _extract_one("https://x.com/weird", "weird", asyncio.Semaphore(1))
        assert result is None


async def test_extract_one_fails_open_on_an_llm_error():
    with patch("app.ingestion.product_enrichment._get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("boom")
        result = await _extract_one("https://x.com/err", "err", asyncio.Semaphore(1))
        assert result is None


async def test_enrich_products_skips_pages_that_already_have_a_deterministic_product(pg_tenant):
    tenant_id, site_url = pg_tenant
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    existing = Product(page_url=f"{site_url}/has-product", name="Existing", source="jsonld")
    existing = existing.model_copy(
        update={"product_key": make_product_key(tenant_id, site_url, existing.page_url, "Existing")}
    )
    try:
        await storage.upsert_products(tenant_id, site_url, job_id, [existing])

        job_pages = JobPages(
            job_id=job_id,
            url=site_url,
            page_count=1,
            pages=[
                PageSummary(
                    url=f"{site_url}/has-product",
                    chars=10,
                    markdown="Buy now for $19.99",
                )
            ],
        )
        with patch("app.ingestion.product_enrichment._get_client") as mock_get_client:
            count = await enrich_products(tenant_id, site_url, job_id, job_pages)
            # The only candidate page already has a product -> no LLM call made.
            mock_get_client.assert_not_called()
            assert count == 0
    finally:
        pool = await storage._pool()
        await pool.execute(
            "DELETE FROM products WHERE tenant_id = $1 AND site_url = $2", tenant_id, site_url
        )


async def test_enrich_products_skips_pages_without_a_price_like_signal(pg_tenant):
    tenant_id, site_url = pg_tenant
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    job_pages = JobPages(
        job_id=job_id,
        url=site_url,
        page_count=1,
        pages=[PageSummary(url=f"{site_url}/about", chars=10, markdown="About our mission")],
    )
    with patch("app.ingestion.product_enrichment._get_client") as mock_get_client:
        count = await enrich_products(tenant_id, site_url, job_id, job_pages)
        mock_get_client.assert_not_called()
        assert count == 0


# ---- storage: replace_site_products / search_products / upsert_products ----


async def test_replace_site_products_persists_and_embeds(pg_tenant):
    tenant_id, site_url = pg_tenant
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    product = Product(
        page_url=f"{site_url}/widget", name="Blue Widget", price_amount=Decimal("49.99"),
        price_currency="USD", source="jsonld",
    )
    product = product.model_copy(
        update={"product_key": make_product_key(tenant_id, site_url, product.page_url, "Blue Widget")}
    )
    try:
        count = await storage.replace_site_products(tenant_id, site_url, job_id, [product])
        assert count == 1

        results = await storage.search_products(tenant_id)
        assert len(results) == 1
        assert results[0].name == "Blue Widget"
        assert results[0].product_key == product.product_key
    finally:
        pool = await storage._pool()
        await pool.execute(
            "DELETE FROM products WHERE tenant_id = $1 AND site_url = $2", tenant_id, site_url
        )


async def test_replace_site_products_replaces_not_accumulates(pg_tenant):
    tenant_id, site_url = pg_tenant
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    first = Product(page_url=f"{site_url}/a", name="First", source="jsonld")
    first = first.model_copy(
        update={"product_key": make_product_key(tenant_id, site_url, first.page_url, "First")}
    )
    second = Product(page_url=f"{site_url}/b", name="Second", source="jsonld")
    second = second.model_copy(
        update={"product_key": make_product_key(tenant_id, site_url, second.page_url, "Second")}
    )
    try:
        await storage.replace_site_products(tenant_id, site_url, job_id, [first])
        await storage.replace_site_products(tenant_id, site_url, job_id, [second])
        results = await storage.search_products(tenant_id)
        assert [p.name for p in results] == ["Second"]
    finally:
        pool = await storage._pool()
        await pool.execute(
            "DELETE FROM products WHERE tenant_id = $1 AND site_url = $2", tenant_id, site_url
        )


async def test_search_products_filters_by_price_range(pg_tenant):
    tenant_id, site_url = pg_tenant
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    cheap = Product(page_url=f"{site_url}/cheap", name="Cheap Thing", price_amount=Decimal("5.00"), source="jsonld")
    cheap = cheap.model_copy(update={"product_key": make_product_key(tenant_id, site_url, cheap.page_url, "Cheap Thing")})
    pricey = Product(page_url=f"{site_url}/pricey", name="Pricey Thing", price_amount=Decimal("500.00"), source="jsonld")
    pricey = pricey.model_copy(update={"product_key": make_product_key(tenant_id, site_url, pricey.page_url, "Pricey Thing")})
    try:
        await storage.replace_site_products(tenant_id, site_url, job_id, [cheap, pricey])

        cheap_only = await storage.search_products(tenant_id, price_max=Decimal("100"))
        assert [p.name for p in cheap_only] == ["Cheap Thing"]

        pricey_only = await storage.search_products(tenant_id, price_min=Decimal("100"))
        assert [p.name for p in pricey_only] == ["Pricey Thing"]
    finally:
        pool = await storage._pool()
        await pool.execute(
            "DELETE FROM products WHERE tenant_id = $1 AND site_url = $2", tenant_id, site_url
        )


async def test_upsert_products_adds_without_wiping_existing(pg_tenant):
    tenant_id, site_url = pg_tenant
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    deterministic = Product(page_url=f"{site_url}/a", name="Deterministic", source="jsonld")
    deterministic = deterministic.model_copy(
        update={"product_key": make_product_key(tenant_id, site_url, deterministic.page_url, "Deterministic")}
    )
    llm_found = Product(page_url=f"{site_url}/b", name="LLM Found", source="llm")
    llm_found = llm_found.model_copy(
        update={"product_key": make_product_key(tenant_id, site_url, llm_found.page_url, "LLM Found")}
    )
    try:
        await storage.replace_site_products(tenant_id, site_url, job_id, [deterministic])
        await storage.upsert_products(tenant_id, site_url, job_id, [llm_found])

        results = await storage.search_products(tenant_id)
        assert {p.name for p in results} == {"Deterministic", "LLM Found"}
    finally:
        pool = await storage._pool()
        await pool.execute(
            "DELETE FROM products WHERE tenant_id = $1 AND site_url = $2", tenant_id, site_url
        )
