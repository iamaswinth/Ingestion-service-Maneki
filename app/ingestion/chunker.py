"""Turn a scraped Page into retrievable Chunks.

Validated against two real scrapes with very different shapes:
- supermemory.ai: 8 clean HTML-id sections (catalog-section, pricing, faq, ...)
- vobiz.ai: a client-side-rendered site whose only "section" is a useless
  `id="root"` wrapper around the entire page, almost no markdown headings,
  and dozens of image tags.

So there are two paths. If the scraped `sections.json` tree has at least one
section with a real (non-generic) id that doesn't just re-wrap the whole page,
we chunk section-by-section (`_chunks_from_sections`). Otherwise we fall back
to chunking the flat page markdown by heading, and if there aren't enough
headings either, by paragraph (`_chunks_from_flat_markdown`).

Every chunk gets a `navigation` target no matter which path produced it:
a real id anchor (`#pricing`), a browser text-fragment anchor
(`#:~:text=...`) when there's no id, or the bare page URL as a last resort.
"""

import hashlib
import re
from typing import Optional
from urllib.parse import quote

from ..config import settings
from ..models import Chunk, ContentType, Page, Section

_GENERIC_ID_PREFIXES = (
    "root",
    "app",
    "page",
    "main",
    "content",
    "wrapper",
    "container",
    "__next",
    "layout",
    "body",
    "site",
)

_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Link text can be empty after image-stripping (e.g. a linked logo:
# `[![Jupiter](logo.png)](jupiter.money)` loses its image first, leaving
# `[](jupiter.money)`) — match `[^\]]*` (zero or more) so those are caught too.
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_HEADING_RE = re.compile(r"(?m)^(#{1,3})\s+(.+)$")

# Fraction of a page's total markdown a single top-level section can occupy
# before we consider it a wrapper rather than real content.
_DOMINANCE_THRESHOLD = 0.8


def clean_text(markdown: str) -> str:
    """Strip image tags, unwrap link text, collapse whitespace.

    This is what gets embedded and spoken — voice output should never
    contain "![Jupiter](https://.../jupiter.png)".
    """
    text = _IMAGE_RE.sub("", markdown)
    text = _LINK_RE.sub(lambda m: m.group(1), text)  # empty capture -> link vanishes entirely
    text = _INLINE_WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _is_generic_id(section_id: Optional[str]) -> bool:
    if not section_id:
        return True
    normalized = section_id.strip().lower()
    return any(normalized.startswith(prefix) for prefix in _GENERIC_ID_PREFIXES)


def _text_fragment(text: str) -> Optional[str]:
    words = text.strip().split()
    if len(words) < 3:
        return None
    return quote(" ".join(words[:8]))


def _make_anchor(page_url: str, section_id: Optional[str], text: str) -> tuple[str, str]:
    if section_id and not _is_generic_id(section_id):
        return "id", f"{page_url}#{section_id}"
    fragment = _text_fragment(text)
    if fragment:
        return "text", f"{page_url}#:~:text={fragment}"
    return "page", page_url


# A bare "$" fires on code samples (`$ npm install`), shell prompts, and any
# page that mentions a currency symbol in passing. Require an actual amount —
# a currency symbol immediately followed by a digit — before calling a chunk
# "pricing" on text evidence alone.
_PRICE_AMOUNT_RE = re.compile(r"[$€£¥]\s?\d")


def _classify_content_type(
    title: Optional[str], text: str, is_first: bool = False
) -> ContentType:
    t = (title or "").lower()
    if "faq" in t or "question" in t:
        return "faq"
    question_lines = sum(1 for line in text.splitlines() if line.strip().endswith("?"))
    if question_lines >= 2:
        return "faq"
    if "pricing" in t or "price" in t or _PRICE_AMOUNT_RE.search(text):
        return "pricing"
    if "testimonial" in t or "review" in t:
        return "testimonial"
    if text.strip().startswith(('"', "“", ">")):
        return "testimonial"
    if is_first:
        return "hero"
    if any(k in t for k in ("feature", "how it works", "use case", "integration")):
        return "feature"
    return "generic"


def _split_long_text(text: str, target: int, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + target, len(text))
        if end < len(text):
            break_at = text.rfind("\n\n", start, end)
            if break_at == -1 or break_at <= start:
                break_at = text.rfind(". ", start, end)
            if break_at != -1 and break_at > start:
                end = break_at + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        # A natural break point can land much closer to `start` than
        # `overlap` (e.g. a short intro paragraph before a long table) —
        # `end - overlap` then undershoots `start` entirely. Retreating to
        # start + 1 in that case doesn't apply overlap, it re-emits nearly
        # the same piece shifted by one character, over and over, until
        # `start` eventually crawls past the whole short piece. Skip the
        # overlap instead of degenerating into that crawl: advance to `end`
        # so the next piece starts fresh with no overlap.
        next_start = end - overlap
        start = next_start if next_start > start else end
    return [p for p in parts if p]


def make_chunk_id(
    tenant_id: str, site_url: str, page_url: str, section_id: Optional[str], suffix: str
) -> str:
    """Deterministic id for a chunk.

    `site_url` is part of the key, not just `page_url`: chunks are replaced by
    (tenant_id, site_url) (see store.replace_site_chunks), so two crawls of the
    same tenant seeded at different URLs — "https://acme.com/" and
    "https://acme.com/docs", or an http/https or www/apex variant of one site —
    are two independent chunk sets. Without site_url in the hash, any page
    reachable from both seeds produces the same chunk_id under two site_urls,
    the delete for one never removes the other's rows, and the INSERT dies on
    the chunk_id primary key, failing the whole ingest.
    """
    raw = f"{tenant_id}|{site_url}|{page_url}|{section_id or ''}|{suffix}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _walk_sections(sections: list[Section], parent_id: Optional[str] = None):
    for sec in sections:
        yield sec, parent_id
        yield from _walk_sections(sec.children, sec.id)


def _resolve_section(
    sec: Section,
) -> tuple[str, list[tuple[Section, Optional[str], str]]]:
    """Post-order walk returning (text_for_this_section, standalone_descendants).

    `app/sections.py` strips every nested id'd child out of its parent's own
    markdown, so parent and child never duplicate text. That pairs badly with a
    flat `len(cleaned) < chunk_min_chars: continue` filter: a child whose text
    sits between sections.py's `_MIN_TEXT_CHARS` (20) and `chunk_min_chars` (40)
    is removed from its parent *and* rejected on its own, so it lands in no
    chunk at all. Real FAQ markup is exactly that shape — one id per question
    and one per answer, each a single short sentence — which silently dropped a
    site's entire FAQ, the highest-value content a sales agent has.

    So instead of dropping an undersized section, roll its text up into the
    nearest ancestor big enough to stand alone. A section that already clears
    the floor is emitted on its own and keeps its own id anchor; only content
    that could never have been a chunk gets merged upward.
    """
    parts = [clean_text(sec.markdown)]
    standalone: list[tuple[Section, Optional[str], str]] = []
    for child in sec.children:
        child_text, child_standalone = _resolve_section(child)
        if len(child_text) >= settings.chunk_min_chars:
            standalone.append((child, sec.id, child_text))
        elif child_text:
            parts.append(child_text)
        # A grandchild that stands on its own does so regardless of whether its
        # own parent was absorbed — its text is already excluded from
        # `child_text`, so this never duplicates content.
        standalone.extend(child_standalone)
    return "\n\n".join(p for p in parts if p), standalone


def _resolve_sections(page: Page) -> list[tuple[Section, Optional[str], str]]:
    """Every section that carries enough content to be its own chunk, in
    document order, paired with its parent id and rolled-up text."""
    resolved: list[tuple[Section, Optional[str], str]] = []
    for sec in page.sections:
        text, standalone = _resolve_section(sec)
        if len(text) >= settings.chunk_min_chars:
            resolved.append((sec, None, text))
        resolved.extend(standalone)
    return resolved


def _page_has_usable_sections(page: Page) -> bool:
    """Walks the *whole* section tree, not just the top level — page builders
    (Framer, Next.js, ...) routinely wrap everything in one generic top-level
    container (id="main"/"root"/"__next") while giving real, well-named ids
    to the sections nested underneath it (e.g. id="features"). Checking only
    page.sections would see that single generic wrapper, conclude there's
    nothing usable, and fall back to flat-markdown chunking for the entire
    page — discarding perfectly good nested section ids in the process.
    """
    if not page.sections:
        return False
    total_chars = max(len(page.markdown), 1)
    for sec, _parent_id in _walk_sections(page.sections):
        if _is_generic_id(sec.id):
            continue
        if sec.chars <= _DOMINANCE_THRESHOLD * total_chars:
            return True
    return False


def _build_chunk(
    *,
    tenant_id: str,
    job_id: str,
    site_url: str,
    page: Page,
    section_id: Optional[str],
    parent_section_id: Optional[str],
    title: Optional[str],
    text: str,
    idx: int,
    is_first: bool = False,
) -> Chunk:
    anchor_type, navigation = _make_anchor(page.url, section_id, text)
    content_type = _classify_content_type(title, text, is_first=is_first)
    # The page's meta description is usually the crispest one-line statement of
    # what the site sells, and it is the only page-level context the first chunk
    # can carry. Fold it into the first chunk's embedded/lexically-searched text
    # (never into `text`, which is what gets spoken).
    prefix = f"{page.title or ''} — {title or ''}"
    if is_first and page.description:
        prefix = f"{prefix} — {page.description}"
    embedding_text = f"{prefix}: {text}".strip(" —:")
    return Chunk(
        chunk_id=make_chunk_id(tenant_id, site_url, page.url, section_id, str(idx)),
        tenant_id=tenant_id,
        job_id=job_id,
        site_url=site_url,
        page_url=page.url,
        section_id=section_id,
        parent_section_id=parent_section_id,
        anchor_type=anchor_type,
        navigation=navigation,
        title=title,
        content_type=content_type,
        chunk_index=idx,
        text=text,
        embedding_text=embedding_text,
    )


def _chunks_from_sections(page: Page, job_id: str, tenant_id: str, site_url: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec, parent_id, cleaned in _resolve_sections(page):
        pieces = _split_long_text(
            cleaned,
            settings.chunk_target_chars,
            settings.chunk_max_chars,
            settings.chunk_overlap_chars,
        )
        # _split_long_text's break-point search can land close to `start`,
        # producing a tiny leftover piece (seen in the wild as single-word/
        # single-character fragments from pages with lots of short
        # animation-reveal "paragraphs") — apply the same min-length floor
        # _chunks_from_flat_markdown already applies to its pieces, so a long
        # section doesn't slip a batch of near-empty chunks past chunk_min_chars.
        idx = 0
        for piece in pieces:
            if len(piece) < settings.chunk_min_chars:
                continue
            chunks.append(
                _build_chunk(
                    tenant_id=tenant_id,
                    job_id=job_id,
                    site_url=site_url,
                    page=page,
                    section_id=sec.id,
                    parent_section_id=parent_id,
                    title=sec.title,
                    text=piece,
                    idx=idx,
                    # `is_first` was previously only ever passed on the
                    # flat-markdown path, so a page with real HTML sections —
                    # the common, well-structured case — could never produce a
                    # "hero" chunk. The voice agent uses content_type to pick an
                    # opening line, so that asymmetry made the good markup the
                    # worse experience.
                    is_first=not chunks,
                )
            )
            idx += 1
    return chunks


def _split_by_headings(markdown: str) -> list[tuple[Optional[str], str]]:
    matches = list(_HEADING_RE.finditer(markdown))
    if not matches:
        return []
    sections: list[tuple[Optional[str], str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        sections.append((m.group(2).strip(), markdown[start:end]))
    return sections


def _split_by_paragraphs(markdown: str, target: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", markdown) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if buf and len(buf) + len(p) + 2 > target:
            chunks.append(buf.strip())
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _chunks_from_flat_markdown(
    page: Page, job_id: str, tenant_id: str, site_url: str
) -> list[Chunk]:
    cleaned_full = clean_text(page.markdown)
    heading_sections = _split_by_headings(cleaned_full)

    pieces_with_title: list[tuple[Optional[str], str]] = []
    if len(heading_sections) >= 2:
        for heading, body in heading_sections:
            cleaned_body = clean_text(body)
            for sub in _split_long_text(
                cleaned_body,
                settings.chunk_target_chars,
                settings.chunk_max_chars,
                settings.chunk_overlap_chars,
            ):
                pieces_with_title.append((heading, sub))
    else:
        pieces_with_title = [
            (None, p) for p in _split_by_paragraphs(cleaned_full, settings.chunk_target_chars)
        ]

    chunks: list[Chunk] = []
    idx = 0
    for heading, piece in pieces_with_title:
        if len(piece) < settings.chunk_min_chars:
            continue
        chunks.append(
            _build_chunk(
                tenant_id=tenant_id,
                job_id=job_id,
                site_url=site_url,
                page=page,
                section_id=None,
                parent_section_id=None,
                title=heading or page.title,
                text=piece,
                idx=idx,
                is_first=(idx == 0),
            )
        )
        idx += 1
    return chunks


def chunk_page(page: Page, job_id: str, tenant_id: str, site_url: str) -> list[Chunk]:
    if _page_has_usable_sections(page):
        chunks = _chunks_from_sections(page, job_id, tenant_id, site_url)
        if chunks:
            return chunks
    return _chunks_from_flat_markdown(page, job_id, tenant_id, site_url)
