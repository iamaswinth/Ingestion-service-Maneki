"""Extract in-content links from a page's HTML, for click-based navigation.

The voice agent normally moves visitors between pages by telling the widget
to set `location.href`, which reloads the page and kills the LiveKit
connection. When the target page is directly linked from the page the
visitor is currently on, the agent can instead ask the widget to click that
real `<a>` element — on a client-side-routed (SPA) target site, the site's
own router intercepts the click and the connection survives. This module is
the ingestion-side half: capturing which pages link to which, so
voice_runtime can make that decision without ever inventing a selector
itself (mirrors how it already never invents an anchor for #id navigation).

`page_key()` is the *sole* definition of "same destination page" on this
side of the wire. It has two other implementations that MUST stay
byte-identical: `voice_runtime/urls.py::page_key` (Python) and
`widget/src/navigation.ts::pageKey` (TypeScript) — see either for the
worked-example table. Changing the semantics here without updating both is a
cross-repo break.

`app/scraper.py::to_pages` feeds this the untouched `rawHtml` Firecrawl
returns, not the `only_main_content`-stripped `html` — so site nav, category
menus, and the header cart link (the links most likely to be SPA-routed) are
captured too, not just in-content links. This was a known v1 limitation
(only in-content links were visible) until `rawHtml` was added to
`app/scraper.py`'s requested formats.
"""

import posixpath
from typing import Optional
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from .models import PageLink

_DEFAULT_PORTS = {"http": 80, "https": 443}

# Generous caps so one pathological page (a huge sitemap-as-content page,
# say) can't blow up the payload voice_runtime fetches on the cold-start path.
MAX_LINKS_PER_PAGE = 200
MAX_ANCHOR_TEXT_CHARS = 120

_SKIP_HREF_PREFIXES = ("javascript:", "mailto:", "tel:", "data:")


def page_key(url: str, base: Optional[str] = None) -> Optional[str]:
    """Normalized "same destination page" key. See module docstring.

    scheme://host[:nondefault-port]path — http(s) only, host lowercased,
    default port dropped, dot-segments collapsed, trailing slash stripped
    (except for the root path), query string and fragment discarded
    entirely. Returns None for anything that isn't a resolvable http(s) URL.

    Path case is preserved (case-sensitivity is server-dependent — folding
    it would let "/About" incorrectly match "/about" on a server where those
    are two different documents) and the path is not percent-decoded.
    """
    if not url:
        return None
    try:
        resolved = urljoin(base, url) if base else url
        parts = urlsplit(resolved)
        scheme = parts.scheme.lower()
        if scheme not in _DEFAULT_PORTS:
            return None
        host = parts.hostname
        if not host:
            return None
        port = parts.port  # raises ValueError on a malformed port
    except ValueError:
        return None

    netloc = host if port is None or port == _DEFAULT_PORTS[scheme] else f"{host}:{port}"

    path = parts.path or "/"
    # posixpath.normpath collapses "." / ".." segments the way a browser
    # resolves them and, as a side effect, also strips a trailing "/" (both
    # wanted here). Known minor discrepancy from the WHATWG URL parser used
    # on the widget side: normpath also collapses repeated internal slashes
    # ("//foo" -> "/foo"), which real URL parsers don't. Pathological enough
    # in real anchor hrefs (not relative-reference dot-segments, which
    # urljoin already resolves above) that it isn't worth a hand-rolled
    # RFC 3986 remove_dot_segments implementation just to avoid it.
    normalized = posixpath.normpath(path)
    if normalized == ".":
        normalized = "/"
    elif not normalized.startswith("/"):
        normalized = "/" + normalized

    return f"{scheme}://{netloc}{normalized}"


def _anchor_text(a) -> str:
    """Visible text, falling back through aria-label -> title -> a
    descendant image's alt text. Empty means "no usable label" — the caller
    drops the link rather than surfacing an unlabeled one a visitor could
    never recognize as "the pricing page" etc."""
    text = " ".join(a.get_text(" ", strip=True).split())
    if text:
        return text
    for attr in ("aria-label", "title"):
        value = a.get(attr)
        if value and value.strip():
            return " ".join(value.split())
    img = a.find("img")
    if img is not None:
        alt = img.get("alt")
        if alt and alt.strip():
            return " ".join(alt.split())
    return ""


def extract_links(html: str, page_url: str) -> list[PageLink]:
    """Every in-content anchor on this page that points at a different page
    (same-page `#id`/`#:~:text=` anchors are excluded — the widget's
    existing scroll branch already owns those), with the visible text a
    visitor would see on it. Never raises; malformed HTML -> []."""
    if not html:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []

    own_key = page_key(page_url)
    seen: set[tuple[str, str]] = set()
    links: list[PageLink] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href == "#":
            continue
        if href.lower().startswith(_SKIP_HREF_PREFIXES):
            continue

        key = page_key(href, base=page_url)
        if key is None or key == own_key:
            continue

        text = _anchor_text(a)
        if not text:
            continue
        text = text[:MAX_ANCHOR_TEXT_CHARS]

        dedupe_key = (key, text.casefold())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        links.append(PageLink(target=urljoin(page_url, href), target_key=key, text=text))
        if len(links) >= MAX_LINKS_PER_PAGE:
            break

    return links
