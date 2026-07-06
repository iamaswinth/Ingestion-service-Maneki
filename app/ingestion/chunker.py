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


def _classify_content_type(
    title: Optional[str], text: str, is_first: bool = False
) -> ContentType:
    t = (title or "").lower()
    if "faq" in t or "question" in t:
        return "faq"
    question_lines = sum(1 for line in text.splitlines() if line.strip().endswith("?"))
    if question_lines >= 2:
        return "faq"
    if "pricing" in t or "price" in t or "$" in text:
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
        start = max(end - overlap, start + 1)
    return [p for p in parts if p]


def _make_chunk_id(tenant_id: str, page_url: str, section_id: Optional[str], idx: int) -> str:
    raw = f"{tenant_id}|{page_url}|{section_id or ''}|{idx}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _walk_sections(sections: list[Section], parent_id: Optional[str] = None):
    for sec in sections:
        yield sec, parent_id
        yield from _walk_sections(sec.children, sec.id)


def _page_has_usable_sections(page: Page) -> bool:
    if not page.sections:
        return False
    total_chars = max(len(page.markdown), 1)
    for sec in page.sections:
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
    embedding_text = f"{page.title or ''} — {title or ''}: {text}".strip(" —:")
    return Chunk(
        chunk_id=_make_chunk_id(tenant_id, page.url, section_id, idx),
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
    for sec, parent_id in _walk_sections(page.sections):
        cleaned = clean_text(sec.markdown)
        if len(cleaned) < settings.chunk_min_chars:
            continue
        pieces = _split_long_text(
            cleaned,
            settings.chunk_target_chars,
            settings.chunk_max_chars,
            settings.chunk_overlap_chars,
        )
        for idx, piece in enumerate(pieces):
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
                )
            )
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
