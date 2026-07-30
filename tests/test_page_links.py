"""Tests for app/links.py (page_key, extract_links) and the storage-layer
lookup that backs GET /page-links/{tenant_id} (app/storage.py::load_page_links).

Pure-function tests need no I/O; the load_page_links tests use the pg_tenant
fixture (tests/conftest.py), which skips cleanly when Postgres isn't reachable.
"""

import uuid

import pytest

from app import storage
from app.links import extract_links, page_key
from app.models import Page

# ---- page_key ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "base", "expected"),
    [
        ("https://Example.com/Pricing/", None, "https://example.com/Pricing"),
        ("https://example.com:443/pricing?ref=x#plans", None, "https://example.com/pricing"),
        ("http://example.com:80/pricing", None, "http://example.com/pricing"),
        ("http://example.com:8080/pricing", None, "http://example.com:8080/pricing"),
        ("/pricing", "https://example.com/a/b", "https://example.com/pricing"),
        ("../pricing", "https://example.com/a/b/c", "https://example.com/a/pricing"),
        ("https://example.com", None, "https://example.com/"),
        ("#pricing", "https://example.com/x", "https://example.com/x"),
    ],
)
def test_page_key_normalization_table(url, base, expected):
    assert page_key(url, base=base) == expected


@pytest.mark.parametrize(
    "url",
    ["mailto:a@b.c", "javascript:alert(1)", "tel:+1234567890", "data:text/plain,hi", ""],
)
def test_page_key_rejects_non_http_schemes(url):
    assert page_key(url) is None


def test_page_key_rejects_unparseable():
    assert page_key("http://[invalid") is None


def test_page_key_equal_regardless_of_query_and_fragment():
    a = page_key("https://x.com/pricing?utm_source=ad")
    b = page_key("https://x.com/pricing#enterprise")
    c = page_key("https://x.com/pricing")
    assert a == b == c


# ---- extract_links ------------------------------------------------------------


def test_extract_links_resolves_relative_and_absolute_hrefs():
    html = """
    <a href="/pricing">Pricing</a>
    <a href="https://x.com/about">About</a>
    """
    links = extract_links(html, "https://x.com/")
    targets = {l.target_key for l in links}
    assert "https://x.com/pricing" in targets
    assert "https://x.com/about" in targets


@pytest.mark.parametrize(
    "href",
    ["javascript:void(0)", "mailto:hello@x.com", "tel:+15551234567", "data:text/plain,hi", "#", ""],
)
def test_extract_links_skips_non_navigational_hrefs(href):
    html = f'<a href="{href}">Click</a>'
    assert extract_links(html, "https://x.com/") == []


def test_extract_links_skips_same_page_anchor():
    html = '<a href="#section">Jump</a><a href="https://x.com/">Home again</a>'
    assert extract_links(html, "https://x.com/") == []


def test_extract_links_label_fallback_chain():
    # Visible text wins over aria-label/title/alt when present.
    assert extract_links(
        '<a href="/a" title="ignored">Visible Text</a>', "https://x.com/"
    )[0].text == "Visible Text"
    # No visible text -> aria-label.
    assert extract_links(
        '<a href="/a" aria-label="Aria Label"></a>', "https://x.com/"
    )[0].text == "Aria Label"
    # No visible text, no aria-label -> title.
    assert extract_links(
        '<a href="/a" title="Title Text"></a>', "https://x.com/"
    )[0].text == "Title Text"
    # No visible text, no aria-label, no title -> descendant img alt.
    assert extract_links(
        '<a href="/a"><img src="x.png" alt="Alt Text"></a>', "https://x.com/"
    )[0].text == "Alt Text"


def test_extract_links_drops_unlabeled_link():
    assert extract_links('<a href="/a"><img src="x.png"></a>', "https://x.com/") == []


def test_extract_links_dedupes_same_target_and_text():
    html = '<a href="/pricing">Pricing</a><a href="/pricing?ref=footer">Pricing</a>'
    links = extract_links(html, "https://x.com/")
    assert len(links) == 1


def test_extract_links_keeps_distinct_text_to_same_target():
    html = '<a href="/pricing">Pricing</a><a href="/pricing">See our plans</a>'
    links = extract_links(html, "https://x.com/")
    assert len(links) == 2


def test_extract_links_caps_at_max_links_per_page():
    html = "".join(f'<a href="/p{i}">Page {i}</a>' for i in range(250))
    links = extract_links(html, "https://x.com/")
    assert len(links) == 200


def test_extract_links_malformed_html_returns_empty():
    assert extract_links(None, "https://x.com/") == []
    assert extract_links("", "https://x.com/") == []


def test_extract_links_truncates_long_anchor_text():
    html = f'<a href="/a">{"x" * 500}</a>'
    links = extract_links(html, "https://x.com/")
    assert len(links[0].text) == 120


# ---- scraper.to_pages' post-loop uncrawled-link filter -----------------------


def test_to_pages_filters_links_to_uncrawled_destinations():
    from app.scraper import to_pages

    docs = [
        {
            "markdown": "Home page content here.",
            "html": '<a href="https://x.com/about">About</a><a href="https://external.com/">External</a>',
            "metadata": {"sourceURL": "https://x.com/"},
        },
        {
            "markdown": "About page content here.",
            "html": "<p>About us.</p>",
            "metadata": {"sourceURL": "https://x.com/about"},
        },
    ]
    pages = to_pages(docs)
    home = next(p for p in pages if p.url == "https://x.com/")
    targets = {l.target_key for l in home.links}
    assert "https://x.com/about" in targets
    assert not any("external.com" in t for t in targets)


# ---- storage.load_page_links (DB) --------------------------------------------


async def _seed_crawl(tenant_id: str, site_url: str, job_id: str, pages: list[Page]) -> None:
    await storage.create_job(job_id, site_url, tenant_id)
    await storage.mark_persisting(job_id, len(pages))
    await storage.persist_pages(job_id, pages)


async def _cleanup_jobs(*job_ids: str) -> None:
    pool = await storage._pool()
    for job_id in job_ids:
        await pool.execute("DELETE FROM jobs WHERE job_id = $1", job_id)


async def test_load_page_links_returns_captured_links(pg_tenant):
    tenant_id, site_url = pg_tenant
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    page = Page(
        url=f"{site_url}/",
        markdown="Home",
        links=[
            {"target": f"{site_url}/pricing", "target_key": f"{site_url}/pricing", "text": "Pricing"}
        ],
    )
    try:
        await _seed_crawl(tenant_id, site_url, job_id, [page])
        record = await storage.load_page_links(tenant_id, page_key(f"{site_url}/"), site_url)
        assert record is not None
        assert record.links[0].text == "Pricing"
    finally:
        await _cleanup_jobs(job_id)


async def test_load_page_links_unknown_page_returns_none(pg_tenant):
    tenant_id, site_url = pg_tenant
    record = await storage.load_page_links(tenant_id, page_key(f"{site_url}/nope"), site_url)
    assert record is None


async def test_load_page_links_null_links_column_returns_empty_list(pg_tenant):
    tenant_id, site_url = pg_tenant
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    page = Page(url=f"{site_url}/", markdown="Home")  # no links
    try:
        await _seed_crawl(tenant_id, site_url, job_id, [page])
        record = await storage.load_page_links(tenant_id, page_key(f"{site_url}/"), site_url)
        assert record is not None
        assert record.links == []
    finally:
        await _cleanup_jobs(job_id)


async def test_load_page_links_site_url_omitted_uses_most_recent_crawl(pg_tenant):
    tenant_id, site_url = pg_tenant
    old_job = f"job-{uuid.uuid4().hex[:8]}"
    new_job = f"job-{uuid.uuid4().hex[:8]}"
    old_page = Page(
        url=f"{site_url}/",
        markdown="Home v1",
        links=[{"target": f"{site_url}/old", "target_key": f"{site_url}/old", "text": "Old"}],
    )
    new_page = Page(
        url=f"{site_url}/",
        markdown="Home v2",
        links=[{"target": f"{site_url}/new", "target_key": f"{site_url}/new", "text": "New"}],
    )
    try:
        await _seed_crawl(tenant_id, site_url, old_job, [old_page])
        await _seed_crawl(tenant_id, site_url, new_job, [new_page])
        record = await storage.load_page_links(tenant_id, page_key(f"{site_url}/"))
        assert record is not None
        assert record.job_id == new_job
        assert record.links[0].text == "New"
    finally:
        await _cleanup_jobs(old_job, new_job)
