"""LLM-fallback product extraction for a page with neither JSON-LD nor Open
Graph product signal (app/products.py) — a custom-built or older-platform
storefront. Async (an Anthropic call), so it can't live alongside the
synchronous scrape-time extractors; wired into app/ingestion/service.py's
ingest_job, after chunking/embedding — mirrors app/ingestion/questions.py's
doc2query shape (batched, concurrency-limited, fail-open) and
app/salescript/graph.py's _create_structured LLM call pattern.

Gated on a cheap price-like signal in the page's markdown (the same
currency-symbol-plus-digit heuristic app/ingestion/chunker.py's
_classify_content_type already uses for its own "pricing" label) — without
this, every non-product page (About, FAQ, Contact) would burn an LLM call
for nothing. Also skips any page app/products.py already found a
deterministic product on, so this only ever adds, never duplicates.

Fail-open throughout, same reasoning as questions.py: a failure here (no API
key, rate limit, malformed output) results in fewer/no LLM-extracted
products, never a failed ingest. Deterministic products are the source of
truth for a tenant that has them; this is a best-effort enhancement for
tenants that don't.
"""

import asyncio
import json
import logging
import re
from typing import Optional

from anthropic import AsyncAnthropic

from .. import storage
from ..config import settings
from ..models import JobPages, Product
from ..products import make_product_key

logger = logging.getLogger(__name__)

_client: Optional[AsyncAnthropic] = None

# Concurrent LLM requests per ingest job — mirrors questions.py's own cap.
_MAX_CONCURRENT_REQUESTS = 4

# A page's markdown is truncated to this many chars in the prompt — enough
# to describe one product, without paying for a full long-form page.
_PROMPT_CHARS = 4000

# Same heuristic app/ingestion/chunker.py's _classify_content_type already
# uses to label a chunk "pricing" — a cheap pre-LLM gate, not a duplicated
# cross-module dependency (this file has no reason to import chunker.py for
# one regex this trivial).
_PRICE_LIKE_RE = re.compile(r"[$€£¥]\s?\d")

# Anthropic's structured-output mode is verified elsewhere in this codebase
# (app/salescript/graph.py, app/ingestion/questions.py) to work reliably with
# plain required string/number/boolean fields — no nullable-union types, to
# stay on the same well-tested ground. "no product" / "unknown field" is
# represented as an empty string / zero and interpreted as None below, same
# convention app/links.py's _anchor_text already uses ("" means "no usable
# value", not "the value is the empty string").
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_product_page": {"type": "boolean"},
        "name": {"type": "string"},
        "price_amount": {"type": "number"},
        "price_currency": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["is_product_page", "name", "price_amount", "price_currency", "description"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You look at one page of a website's content and decide whether it is a "
    "single product's detail page (a specific item for sale, not a category "
    "listing, blog post, comparison table, or generic marketing page). If it "
    "is, extract the product's name, price as a plain number with no "
    "currency symbol (0 if not stated), its ISO currency code if "
    "determinable (empty string if not), and a one-sentence description "
    "(empty string if not stated). Never invent a price, name, or detail "
    "that isn't stated on the page — if the page doesn't clearly describe "
    "one specific product, set is_product_page to false and leave name/"
    "description as empty strings and price_amount as 0."
)


def enabled() -> bool:
    return bool(settings.product_llm_fallback_enabled and settings.anthropic_api_key)


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.product_llm_fallback_timeout_seconds,
        )
    return _client


def _looks_like_a_product_page(markdown: str) -> bool:
    """Cheap gate before ever spending an LLM call."""
    return bool(_PRICE_LIKE_RE.search(markdown))


async def _extract_one(
    page_url: str, markdown: str, semaphore: asyncio.Semaphore
) -> Optional[Product]:
    """One LLM call for one candidate page. None on any failure, a
    non-product verdict, or a missing name — never raises."""
    async with semaphore:
        try:
            response = await _get_client().messages.create(
                model=settings.product_llm_fallback_model,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
                messages=[{"role": "user", "content": markdown[:_PROMPT_CHARS]}],
            )
        except Exception:
            logger.warning(
                "product LLM fallback request failed", extra={"page_url": page_url}, exc_info=True
            )
            return None

    try:
        text = next((b.text for b in response.content if b.type == "text"), "")
        data = json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(data, dict):
        return None

    if not data.get("is_product_page"):
        return None
    name = data.get("name", "").strip()
    if not name:
        return None

    price_amount = data.get("price_amount") or None
    price_currency = data.get("price_currency", "").strip() or None
    description = data.get("description", "").strip() or None
    return Product(
        page_url=page_url,
        name=name,
        price_amount=price_amount,
        price_currency=price_currency,
        description=description,
        source="llm",
    )


async def enrich_products(
    tenant_id: str, site_url: str, job_id: str, job_pages: JobPages
) -> int:
    """Runs the LLM fallback for every page in this crawl that has neither a
    deterministic product nor an obviously-missing price-like signal.
    Fail-open throughout — see module docstring. Returns how many products
    were added."""
    if not enabled():
        return 0

    pool = await storage._pool()
    rows = await pool.fetch(
        "SELECT DISTINCT page_url FROM products WHERE tenant_id = $1 AND site_url = $2",
        tenant_id,
        site_url,
    )
    already_has_product = {row["page_url"] for row in rows}

    candidates = [
        page
        for page in job_pages.pages
        if page.url not in already_has_product
        and page.markdown
        and _looks_like_a_product_page(page.markdown)
    ]
    if not candidates:
        return 0

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)
    results = await asyncio.gather(
        *(_extract_one(page.url, page.markdown, semaphore) for page in candidates),
        return_exceptions=True,
    )

    products: list[Product] = []
    failures = 0
    for result in results:
        if isinstance(result, BaseException):
            failures += 1
            logger.warning("product LLM fallback task failed: %s", result)
        elif result is not None:
            products.append(
                result.model_copy(
                    update={
                        "product_key": make_product_key(
                            tenant_id, site_url, result.page_url, result.name
                        )
                    }
                )
            )
    if failures:
        logger.warning(
            "product LLM fallback: %d/%d candidate pages failed; continuing without them",
            failures,
            len(candidates),
        )

    if products:
        await storage.upsert_products(tenant_id, site_url, job_id, products)
    return len(products)
