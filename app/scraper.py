"""Thin async wrapper around the Firecrawl SDK.

Isolates the rest of the app from Firecrawl's response shapes: everything here
returns plain dicts / our Pydantic models, so swapping cloud <-> self-hosted or
upgrading the SDK only touches this file.
"""

import time
from typing import Any, Optional

import httpx
from firecrawl import AsyncFirecrawl

from .actions import extract_actions
from .config import settings
from .links import extract_links, page_key
from .models import Page
from .products import extract_products
from .sections import extract_sections

_client = AsyncFirecrawl(
    api_key=settings.firecrawl_api_key,
    api_url=settings.firecrawl_api_url,
)
_http = httpx.AsyncClient(base_url=settings.firecrawl_api_url)


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Read a field that may be an attribute (SDK model) or a dict key."""
    for name in names:
        if obj is None:
            return default
        if isinstance(obj, dict):
            if name in obj and obj[name] is not None:
                return obj[name]
        else:
            val = getattr(obj, name, None)
            if val is not None:
                return val
    return default


def _scrape_options(wait_for_ms: int) -> dict:
    # markdown = clean text for ingestion; html = kept so we can recover the
    # id-anchored section structure that markdown conversion discards.
    # rawHtml = the untouched document *before* only_main_content stripping —
    # nav/header/footer, <button>s, and <script type="application/ld+json">
    # all survive here even though they never reach html/markdown (Firecrawl's
    # own pipeline is rawHtml -> html -> markdown, so requesting rawHtml
    # changes nothing about what those two contain). Feeds extract_links
    # (app/links.py) and extract_products/extract_actions (app/products.py,
    # app/actions.py) — never persisted itself, only its extracted structure
    # is. wait_for lets client-side-rendered pages hydrate before we capture
    # them.
    return {
        "formats": ["markdown", "html", "rawHtml"],
        "only_main_content": True,
        "wait_for": wait_for_ms,
    }


async def start_crawl(
    url: str,
    limit: int,
    wait_for_ms: int,
    include_paths: Optional[list[str]] = None,
    exclude_paths: Optional[list[str]] = None,
) -> str:
    """Kick off a crawl and return Firecrawl's crawl id (our job_id)."""
    kwargs: dict[str, Any] = {
        "url": url,
        "limit": limit,
        "scrape_options": _scrape_options(wait_for_ms),
    }
    if include_paths:
        kwargs["include_paths"] = include_paths
    if exclude_paths:
        kwargs["exclude_paths"] = exclude_paths

    job = await _client.start_crawl(**kwargs)
    job_id = _get(job, "id", "job_id")
    if not job_id:
        raise RuntimeError(f"Firecrawl did not return a crawl id: {job!r}")
    return str(job_id)


async def get_status(job_id: str) -> dict:
    """Return normalized crawl status plus the raw page documents (if any)."""
    status = await _client.get_crawl_status(job_id)
    raw_status = str(_get(status, "status", default="scraping"))
    normalized = {
        "scraping": "scraping",
        "active": "scraping",
        "waiting": "scraping",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(raw_status, "scraping")

    return {
        "status": normalized,
        "completed": int(_get(status, "completed", default=0) or 0),
        "total": int(_get(status, "total", default=0) or 0),
        "data": _get(status, "data", default=[]) or [],
    }


def to_pages(documents: list[Any]) -> list[Page]:
    """Normalize Firecrawl documents into our Page model.

    Deduped by URL, keeping the longest markdown per URL: Firecrawl can hand
    back the same `sourceURL` twice in one crawl's document list (a redirect
    followed to the same canonical page, an http/https or trailing-slash
    variant that normalizes to the same URL on Firecrawl's side). Two Page
    rows sharing one url produce identical chunk_ids downstream (chunker
    hashes tenant|site_url|page_url|section_id|idx — url is the same for
    both), so the second INSERT would violate the chunks primary key and
    fail the whole ingest. Keeping the longest markdown also means a thin
    redirect stub never wins over the real page it points to.
    """
    by_url: dict[str, Page] = {}
    for doc in documents:
        markdown = _get(doc, "markdown", default="") or ""
        html = _get(doc, "html", "raw_html", "rawHtml", default="") or ""
        # The untouched pre-strip document — deliberately a SEPARATE read
        # from `html` above. That fallback chain exists in case "html" were
        # ever absent from the response; here we want the raw document
        # specifically, and the installed firecrawl-py SDK normalizes the
        # wire field "rawHtml" to the attribute/key "raw_html" (confirmed
        # against the SDK source), so that spelling is checked first.
        raw_html = _get(doc, "raw_html", "rawHtml", default="") or ""
        metadata = _get(doc, "metadata", default={}) or {}
        url = _get(metadata, "source_url", "sourceURL", "url", default="") or ""
        title = _get(metadata, "title", "og_title", "ogTitle")
        description = _get(metadata, "description", "og_description", "ogDescription")
        if not markdown.strip():
            continue
        url_str = str(url)
        existing = by_url.get(url_str)
        if existing is not None and len(existing.markdown) >= len(markdown):
            continue
        by_url[url_str] = Page(
            url=url_str,
            title=title,
            description=description,
            markdown=markdown,
            # Content structure comes from the main-content-only html: a
            # nav/header/footer section chunked as if it were page content
            # would just be noise. Links/products/actions instead want the
            # untouched raw_html — see app/links.py's (now resolved) v1
            # limitation note, app/products.py, and app/actions.py.
            sections=extract_sections(html),
            links=extract_links(raw_html, url_str),
            products=extract_products(raw_html, url_str),
            page_actions=extract_actions(raw_html, url_str),
        )

    # Restrict each page's links to destinations we actually crawled. Every
    # navigation target the agent can ever pick comes from a crawled page's
    # chunks (ingestion/service.py only chunks job_pages.pages), so this
    # cannot drop a usable click target — it only drops external links,
    # uncrawled paths, and asset links, shrinking the payload voice_runtime
    # fetches on the cold-start path. Must run after the loop above: the set
    # of crawled URLs (and therefore of duplicate-URL "winners") is only
    # final once every document has been seen.
    known_keys = {page_key(u) for u in by_url}
    for page in by_url.values():
        page.links = [link for link in page.links if link.target_key in known_keys]

    return list(by_url.values())


async def map_site(url: str, limit: int) -> list[str]:
    """Preview the URLs Firecrawl would discover, without scraping them."""
    res = await _client.map(url=url, limit=limit)
    links = _get(res, "links", "urls", default=[]) or []
    out: list[str] = []
    for link in links:
        if isinstance(link, str):
            out.append(link)
        else:
            u = _get(link, "url", "href")
            if u:
                out.append(str(u))
    return out


_REACHABLE_CACHE_SECONDS = 45
_reachable_cache: tuple[float, bool] | None = None


async def reachable() -> bool:
    """Cheap health probe against the Firecrawl instance: hits its liveness
    stub (no crawl/queue work, no outbound request to a third-party site)
    and caches the result briefly. /health is this app's one unauthenticated
    endpoint, so keeping each probe near-free matters regardless of who —
    or how often — is calling it."""
    global _reachable_cache
    now = time.monotonic()
    if _reachable_cache is not None and now - _reachable_cache[0] < _REACHABLE_CACHE_SECONDS:
        return _reachable_cache[1]
    try:
        resp = await _http.get("/v0/health/liveness", timeout=5.0)
        ok = resp.status_code == 200
    except Exception:
        ok = False
    _reachable_cache = (now, ok)
    return ok
