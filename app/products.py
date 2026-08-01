"""Extract structured product data from a page's raw HTML, for the voice
shopping agent (GET /products/{tenant_id}) and product-aware conversation.

Extraction priority, each falling through to the next:
1. JSON-LD `schema.org/Product` — the highest-value signal by far: Shopify,
   WooCommerce, BigCommerce, and Magento all emit it by default.
2. Open Graph product tags (`og:type="product"` + `product:*` meta tags) —
   fallback for sites without JSON-LD.

A page with neither gets a separate, async LLM-fallback pass elsewhere (see
`app/ingestion/product_enrichment.py`) — that path needs a network call, so
it can't live in this module alongside the synchronous scrape-time
extractors (mirrors why app/links.py stays sync and
app/ingestion/questions.py's doc2query is a separate async module).

Deliberately does NOT parse microdata/RDFa (`itemscope`/`itemprop`) — real
DOM-tree-walking complexity for coverage JSON-LD/OG already capture on the
large majority of real storefronts. Revisit if a real tenant's site needs it.

`app/scraper.py::to_pages` feeds this the untouched `rawHtml`, same as
`app/links.py::extract_links` — a JSON-LD block or OG meta tag stripped by
`only_main_content` would otherwise be invisible here too.
"""

import hashlib
import json
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from bs4 import BeautifulSoup

from .models import Product

# Generous cap so one pathological page (a huge category listing, say) can't
# blow up extraction — mirrors app/links.py's MAX_LINKS_PER_PAGE.
MAX_PRODUCTS_PER_PAGE = 50


def make_product_key(tenant_id: str, site_url: str, page_url: str, sku_or_name: str) -> str:
    """Deterministic id for a product row, computed at persistence time
    (app/storage.py) — not by extract_products, which has no tenant_id/
    site_url in scope (it runs inside app/scraper.py::to_pages, before a
    tenant/site is attached). Mirrors
    app/ingestion/chunker.py::make_chunk_id's exact hashing convention:
    site_url is part of the key (not just page_url) because products are
    replaced by (tenant_id, site_url) — see app/storage.py::replace_site_products —
    so two crawls of the same tenant seeded at different URLs must produce
    independent product sets, never colliding primary keys."""
    raw = f"{tenant_id}|{site_url}|{page_url}|{sku_or_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _iter_jsonld_objects(soup: BeautifulSoup):
    """Every JSON-LD object on the page, flattening lists and `@graph`
    wrappers (a page can bundle several distinct schema.org types under one
    `@graph` array). Never raises: a malformed <script> tag is skipped, not
    fatal to the rest of the page's extraction."""
    for tag in soup.find_all("script", type="application/ld+json"):
        if not tag.string:
            continue
        try:
            data = json.loads(tag.string)
        except (json.JSONDecodeError, TypeError, RecursionError):
            continue
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            if isinstance(graph, list):
                for entry in graph:
                    if isinstance(entry, dict):
                        yield entry
            else:
                yield item


def _is_product_type(obj: dict) -> bool:
    type_field = obj.get("@type")
    types = type_field if isinstance(type_field, list) else [type_field]
    return any(isinstance(t, str) and t.strip().lower() == "product" for t in types)


def _parse_price(value: Any) -> Optional[Decimal]:
    """schema.org/OG prices arrive as a string, a number, or aren't present
    at all — never trust the shape. A price like "Contact us" or "" must
    degrade to None, not raise."""
    if value is None:
        return None
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


def _first_offer(obj: dict) -> dict:
    offers = obj.get("offers")
    if isinstance(offers, list):
        return next((o for o in offers if isinstance(o, dict)), {})
    return offers if isinstance(offers, dict) else {}


def _first_image(obj: dict) -> Optional[str]:
    image = obj.get("image")
    if isinstance(image, str):
        return image
    if isinstance(image, list):
        return next((i for i in image if isinstance(i, str)), None)
    if isinstance(image, dict):
        url = image.get("url")
        return url if isinstance(url, str) else None
    return None


def _clean_str(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _product_from_jsonld(obj: dict, page_url: str) -> Optional[Product]:
    name = _clean_str(obj.get("name"))
    if name is None:
        return None
    offer = _first_offer(obj)
    sku = obj.get("sku") or obj.get("mpn")
    return Product(
        page_url=page_url,
        name=name,
        sku=str(sku) if sku else None,
        price_amount=_parse_price(offer.get("price")),
        price_currency=_clean_str(offer.get("priceCurrency")),
        availability=_clean_str(offer.get("availability")),
        image_url=_first_image(obj),
        description=_clean_str(obj.get("description")),
        source="jsonld",
    )


def _extract_jsonld_products(soup: BeautifulSoup, page_url: str) -> list[Product]:
    products: list[Product] = []
    for obj in _iter_jsonld_objects(soup):
        if not _is_product_type(obj):
            continue
        product = _product_from_jsonld(obj, page_url)
        if product is None:
            continue
        products.append(product)
        if len(products) >= MAX_PRODUCTS_PER_PAGE:
            break
    return products


def _meta_content(soup: BeautifulSoup, prop: str) -> Optional[str]:
    tag = soup.find("meta", attrs={"property": prop})
    if tag is None:
        return None
    return _clean_str(tag.get("content"))


def _extract_og_product(soup: BeautifulSoup, page_url: str) -> list[Product]:
    """Only fires when og:type is literally "product" — a generic og:image/
    og:title pair on a non-product page (every page has those) must not be
    mistaken for a product."""
    if _meta_content(soup, "og:type") != "product":
        return []
    name = _meta_content(soup, "og:title")
    if name is None:
        return []
    return [
        Product(
            page_url=page_url,
            name=name,
            price_amount=_parse_price(_meta_content(soup, "product:price:amount")),
            price_currency=_meta_content(soup, "product:price:currency"),
            availability=_meta_content(soup, "product:availability"),
            image_url=_meta_content(soup, "og:image"),
            description=_meta_content(soup, "og:description"),
            source="og",
        )
    ]


def extract_products(html: str, page_url: str) -> list[Product]:
    """Products on this page: JSON-LD first, falling back to Open Graph
    product tags only if JSON-LD found nothing (a page rarely has useful
    signal in both — trying OG unconditionally would risk a worse-quality
    duplicate of what JSON-LD already gave a confident answer for). Never
    raises; malformed or absent HTML -> []."""
    if not html:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []

    products = _extract_jsonld_products(soup, page_url)
    if not products:
        products = _extract_og_product(soup, page_url)
    return products
