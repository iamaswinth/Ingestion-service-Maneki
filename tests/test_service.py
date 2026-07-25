"""Unit tests for app/ingestion/service.py's pure helpers -- no I/O, no live
infra. ingest_job() itself is exercised end-to-end only via the live API
(see README/CLAUDE.md dev workflow); these pin the two behaviors that don't
need a database to verify.
"""

from app.ingestion.service import _dedupe_chunks
from app.models import Chunk


def _chunk(*, chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        tenant_id="t",
        job_id="j",
        site_url="https://x.com/",
        page_url="https://x.com/p",
        anchor_type="page",
        navigation="https://x.com/p",
        text=text,
        embedding_text=text,
    )


def test_dedupe_drops_exact_duplicate_text_keeping_first():
    # Repeated CTA/cookie-notice/footer-in-content blocks survive
    # only_main_content and get chunked once per page they appear on.
    a = _chunk(chunk_id="a", text="Book a demo with our sales team today.")
    b = _chunk(chunk_id="b", text="Some unique content that appears only once.")
    c = _chunk(chunk_id="c", text="Book a demo with our sales team today.")

    kept = _dedupe_chunks([a, b, c])

    assert [c.chunk_id for c in kept] == ["a", "b"]


def test_dedupe_is_whitespace_and_case_insensitive():
    a = _chunk(chunk_id="a", text="Book a Demo")
    b = _chunk(chunk_id="b", text="book   a\ndemo")

    kept = _dedupe_chunks([a, b])

    assert [c.chunk_id for c in kept] == ["a"]


def test_dedupe_keeps_distinct_text_untouched():
    a = _chunk(chunk_id="a", text="First distinct passage.")
    b = _chunk(chunk_id="b", text="Second distinct passage.")

    kept = _dedupe_chunks([a, b])

    assert len(kept) == 2
