"""Golden tests for app/ingestion/chunker.py — pure functions, no I/O, no
live infra. Tests both the public chunk_page() entry point end-to-end and
several private helpers directly (Python doesn't enforce privacy, and
pinning e.g. _split_long_text's exact split points is cheaper and more
precise than only exercising it through the full pipeline).
"""

import hashlib
from urllib.parse import quote

from app import config as config_module
from app.ingestion import chunker
from app.models import Page, Section

# ---- clean_text ----------------------------------------------------------


def test_clean_text_strips_images():
    assert chunker.clean_text("![alt text](http://example.com/img.png)") == ""


def test_clean_text_unwraps_links():
    assert chunker.clean_text("[click here](http://example.com)") == "click here"


def test_clean_text_image_wrapped_link_vanishes_entirely():
    # Image stripping runs first, leaving `[](url)`; the link regex's empty
    # capture group then deletes that too — a linked logo disappears fully.
    assert chunker.clean_text("[![Jupiter](logo.png)](https://jupiter.money)") == ""


def test_clean_text_collapses_inline_whitespace():
    assert chunker.clean_text("a   b\t\tc") == "a b c"


def test_clean_text_collapses_blank_lines():
    assert chunker.clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_clean_text_strips_leading_trailing_whitespace():
    assert chunker.clean_text("   hello world   ") == "hello world"


# ---- _is_generic_id --------------------------------------------------------


def test_is_generic_id_prefix_matches():
    assert chunker._is_generic_id("root") is True
    assert chunker._is_generic_id("content-wrapper") is True
    # Prefix-match trap: these look like real content ids but start with a
    # generic prefix ("page", "site") so are still treated as generic.
    assert chunker._is_generic_id("pages-hero") is True
    assert chunker._is_generic_id("sitemap") is True


def test_is_generic_id_missing_or_specific():
    assert chunker._is_generic_id(None) is True
    assert chunker._is_generic_id("") is True
    assert chunker._is_generic_id("pricing") is False
    assert chunker._is_generic_id("faq") is False


# ---- _make_anchor (navigation 3-way priority) ------------------------------


def test_make_anchor_id_when_section_id_present_and_specific():
    anchor_type, nav = chunker._make_anchor(
        "https://x.com/p", "pricing", "irrelevant text body here"
    )
    assert anchor_type == "id"
    assert nav == "https://x.com/p#pricing"


def test_make_anchor_text_fragment_when_no_specific_id():
    text = "This chunk has more than three words in it, plus extras"
    anchor_type, nav = chunker._make_anchor("https://x.com/p", None, text)
    expected_fragment = quote(" ".join(text.split()[:8]))
    assert anchor_type == "text"
    assert nav == f"https://x.com/p#:~:text={expected_fragment}"


def test_make_anchor_falls_back_to_bare_page_url():
    # Generic id AND fewer than 3 words -> no id anchor, no text fragment.
    anchor_type, nav = chunker._make_anchor("https://x.com/p", "root", "two words")
    assert anchor_type == "page"
    assert nav == "https://x.com/p"


# ---- _classify_content_type precedence -------------------------------------


def test_classify_content_type_title_faq_wins_regardless_of_text():
    assert (
        chunker._classify_content_type("FAQ", "Just a plain statement, no question mark.")
        == "faq"
    )


def test_classify_content_type_two_question_lines_without_faq_title():
    text = "Do you support SSO?\nIs there a public API?\nWe cover both above."
    assert chunker._classify_content_type(None, text) == "faq"


def test_classify_content_type_dollar_sign_beats_testimonial_quote():
    # Pricing check runs before the testimonial check, so a quote-prefixed
    # testimonial that happens to mention a dollar amount is classified
    # "pricing" — a deliberate adversarial case pinning check order.
    text = '"This saved us $500 a month," said our happiest customer.'
    assert chunker._classify_content_type(None, text) == "pricing"


def test_classify_content_type_hero_only_when_is_first():
    text = "Welcome to our product landing page today."
    assert chunker._classify_content_type(None, text, is_first=True) == "hero"
    assert chunker._classify_content_type(None, text, is_first=False) == "generic"


def test_classify_content_type_feature_keyword():
    assert chunker._classify_content_type("Key Features", "Fast and reliable.") == "feature"


# ---- _split_long_text -------------------------------------------------------


def test_split_long_text_under_max_chars_stays_one_piece():
    text = "a" * 1500
    assert chunker._split_long_text(text, target=1000, max_chars=1500, overlap=150) == [text]


def test_split_long_text_paragraph_break_and_overlap():
    block1 = "X" * 950
    block2 = "Y" * 700
    text = block1 + "\n\n" + block2  # len 1652, over max_chars(1500)

    parts = chunker._split_long_text(text, target=1000, max_chars=1500, overlap=150)

    assert len(parts) == 2
    assert parts[0] == block1
    # Second piece starts 150 chars back from the paragraph break (minus
    # whatever whitespace strip() removes), carrying real overlap content.
    assert parts[1] == text[801:].strip()
    assert parts[1].startswith("X")
    assert parts[1].endswith("Y" * 700)


def test_split_long_text_falls_back_to_sentence_boundary():
    sentence = "This is a filler sentence for testing chunk splitting logic. "
    text = sentence * 40
    assert "\n\n" not in text

    parts = chunker._split_long_text(text, target=1000, max_chars=1500, overlap=150)

    assert len(parts) >= 2
    # The sentence-boundary break point lands right after a "." (no trailing
    # space survives strip()).
    assert parts[0].endswith(".")


def test_split_long_text_hard_cut_when_no_delimiter_found():
    text = "a" * 2000  # no "\n\n", no ". " anywhere
    parts = chunker._split_long_text(text, target=1000, max_chars=1500, overlap=150)
    assert parts[0] == "a" * 1000


# ---- make_chunk_id determinism + site_url isolation -------------------------


def test_make_chunk_id_is_deterministic_sha256():
    cid = chunker.make_chunk_id("tenant1", "https://x.com/", "https://x.com/p", "pricing", "0")
    expected = hashlib.sha256(
        "tenant1|https://x.com/|https://x.com/p|pricing|0".encode("utf-8")
    ).hexdigest()[:16]
    assert cid == expected
    assert (
        chunker.make_chunk_id("tenant1", "https://x.com/", "https://x.com/p", "pricing", "0")
        == cid
    )
    assert (
        chunker.make_chunk_id("tenant1", "https://x.com/", "https://x.com/p", "pricing", "1")
        != cid
    )


def test_make_chunk_id_differs_across_site_url():
    # Same tenant, same page_url, two different crawl seeds (site_url) for
    # that tenant -- e.g. a re-crawl started from "https://x.com/" vs.
    # "https://x.com/docs" that both happen to reach "https://x.com/p". These
    # must not collide: replace_site_chunks deletes by (tenant_id, site_url),
    # so a shared chunk_id would mean the delete for one site_url can never
    # remove the other's row, and the next insert dies on the chunk_id
    # primary key -- failing the whole ingest (see chunker.make_chunk_id's
    # docstring).
    a = chunker.make_chunk_id("tenant1", "https://x.com/", "https://x.com/p", "pricing", "0")
    b = chunker.make_chunk_id("tenant1", "https://x.com/docs", "https://x.com/p", "pricing", "0")
    assert a != b


# ---- _page_has_usable_sections dominance-threshold boundary -----------------


def test_dominance_threshold_boundary():
    # chars just over 80% of total page markdown -> section is a wrapper,
    # not usable.
    page_over = Page(
        url="https://x.com/p",
        markdown="x" * 1000,
        sections=[Section(id="highlights", tag="div", chars=801, markdown="y" * 801)],
    )
    assert chunker._page_has_usable_sections(page_over) is False

    # Exactly at 80% (<=, inclusive) -> usable.
    page_at = Page(
        url="https://x.com/p",
        markdown="x" * 1000,
        sections=[Section(id="highlights", tag="div", chars=800, markdown="y" * 800)],
    )
    assert chunker._page_has_usable_sections(page_at) is True


# ---- chunk_page end-to-end: clean id'd-sections path -------------------------


def test_chunk_page_routes_through_sections_when_ids_are_usable():
    pricing_md = (
        "# Pricing\n\nWe charge $10 per month for the pro plan and offer a "
        "generous free tier as well for smaller teams."
    )
    faq_md = (
        "# FAQ\n\nDo you offer refunds? Yes, within 30 days.\n\n"
        "Is there a free trial? Yes, for 14 days on any plan."
    )
    sections = [
        Section(id="pricing", tag="section", title="Pricing", chars=len(pricing_md), markdown=pricing_md),
        Section(id="faq", tag="section", title="FAQ", chars=len(faq_md), markdown=faq_md),
    ]
    page = Page(
        url="https://x.com/page",
        title="X",
        markdown="\n\n".join([pricing_md, faq_md]),
        sections=sections,
    )

    chunks = chunker.chunk_page(page, job_id="job1", tenant_id="tenant1", site_url="https://x.com")

    assert len(chunks) == 2
    by_id = {c.section_id: c for c in chunks}
    assert by_id["pricing"].anchor_type == "id"
    assert by_id["pricing"].navigation == "https://x.com/page#pricing"
    assert by_id["pricing"].content_type == "pricing"
    assert by_id["faq"].content_type == "faq"
    assert all(c.tenant_id == "tenant1" and c.job_id == "job1" for c in chunks)


# ---- chunk_page end-to-end: generic-wrapper -> flat markdown fallback -------


def test_chunk_page_falls_back_to_flat_markdown_for_generic_wrapper():
    md = (
        "# Welcome\n\nWe build great software for teams everywhere today.\n\n"
        "## Features\n\nFast, reliable, and secure infrastructure for your "
        "team's needs.\n\n"
        "## Contact\n\nReach us any time via email or chat support for help."
    )
    sections = [Section(id="root", tag="div", chars=len(md), markdown=md)]
    page = Page(url="https://y.com/", title="Y", markdown=md, sections=sections)

    chunks = chunker.chunk_page(page, job_id="job2", tenant_id="tenant2", site_url="https://y.com")

    assert len(chunks) == 3
    assert all(c.section_id is None for c in chunks)
    titles = [c.title for c in chunks]
    assert titles == ["Welcome", "Features", "Contact"]
    # Only the flat-markdown path's first chunk can be classified "hero".
    assert chunks[0].content_type == "hero"
    assert chunks[1].content_type == "feature"


# ---- parent_section_id: immediate-parent-only nesting ------------------------


def test_parent_section_id_is_immediate_parent_only():
    faq_a_md = (
        "The refund window is 30 days from the original purchase date for "
        "all subscription plans, no questions asked."
    )
    faq_q_md = (
        "Do you offer refunds for annual plans purchased within the last "
        "few weeks of this billing cycle?"
    )
    faq_md = (
        "Frequently asked questions about our refund, billing, and "
        "subscription cancellation policies for all customers."
    )
    faq_a = Section(id="faq-a-01", tag="div", title="Answer", chars=len(faq_a_md), markdown=faq_a_md)
    faq_q = Section(
        id="faq-q-01", tag="div", title="Question", chars=len(faq_q_md), markdown=faq_q_md, children=[faq_a]
    )
    faq = Section(id="faq", tag="section", title="FAQ", chars=len(faq_md), markdown=faq_md, children=[faq_q])

    page = Page(
        url="https://x.com/faq",
        markdown="\n\n".join([faq_md, faq_q_md, faq_a_md] * 2),
        sections=[faq],
    )

    chunks = chunker.chunk_page(page, job_id="job3", tenant_id="tenant3", site_url="https://x.com")

    by_id = {c.section_id: c for c in chunks}
    assert by_id["faq"].parent_section_id is None
    assert by_id["faq-q-01"].parent_section_id == "faq"
    assert by_id["faq-a-01"].parent_section_id == "faq-q-01"


# ---- chunk_min_chars: per-piece filtering + index renumbering ---------------


# ---- undersized nested sections roll up instead of vanishing ----------------


def test_undersized_faq_sections_roll_up_into_parent():
    """Real FAQ markup is one id per question and one per answer, each a
    single short sentence -- shorter than chunk_min_chars (40) on its own,
    but also stripped out of its parent's own markdown by
    app/sections.py::extract_sections (parent and child never duplicate
    text). Before the roll-up fix, a section in that gap landed in *no*
    chunk at all: too big to keep in the parent, too small to stand alone.
    This reproduces that shape and asserts every question and answer still
    reaches some chunk.
    """
    faq_q1 = Section(id="faq-q-1", tag="div", title="Do you support HIPAA?",
                      chars=25, markdown="### Do you support HIPAA?")
    faq_a1 = Section(id="faq-a-1", tag="div", title="Faq A 1",
                      chars=28, markdown="Yes, we are HIPAA compliant.")
    faq_q2 = Section(id="faq-q-2", tag="div", title="Can I cancel anytime?",
                      chars=25, markdown="### Can I cancel anytime?")
    faq_a2 = Section(id="faq-a-2", tag="div", title="Faq A 2",
                      chars=29, markdown="Yes. Cancel anytime, no fees.")
    faq = Section(
        id="faq", tag="section", title="FAQ", chars=29,
        markdown="## Frequently Asked Questions",
        children=[faq_q1, faq_a1, faq_q2, faq_a2],
    )
    page = Page(
        url="https://x.com/",
        title="X",
        markdown="# FAQ\n\nDo you support HIPAA?\n\nYes, we are HIPAA compliant.\n\n"
        "Can I cancel anytime?\n\nYes. Cancel anytime, no fees.",
        sections=[faq],
    )

    chunks = chunker.chunk_page(page, job_id="j", tenant_id="t", site_url="https://x.com/")

    all_text = " ".join(c.text for c in chunks)
    assert "HIPAA compliant" in all_text
    assert "no fees" in all_text
    # None of the four undersized children got their own chunk_min_chars-
    # violating chunk -- they were absorbed into the "faq" parent instead.
    assert all(len(c.text) >= 40 for c in chunks)


def test_child_that_clears_the_floor_still_gets_its_own_chunk():
    # A nested section with real substance (>= chunk_min_chars on its own)
    # must NOT be swept into its parent -- only genuinely-too-small content
    # gets rolled up.
    child_md = "This child section has plenty of its own real content here."
    child = Section(id="pricing-detail", tag="div", title="Detail", chars=len(child_md), markdown=child_md)
    parent = Section(id="pricing", tag="section", title="Pricing", chars=10, markdown="Pricing", children=[child])
    page = Page(url="https://x.com/", markdown="Pricing\n\n" + child_md, sections=[parent])

    chunks = chunker.chunk_page(page, job_id="j", tenant_id="t", site_url="https://x.com/")

    by_id = {c.section_id: c for c in chunks}
    assert "pricing-detail" in by_id
    assert by_id["pricing-detail"].text == child_md
    assert by_id["pricing-detail"].parent_section_id == "pricing"


def test_chunk_min_chars_filters_pieces_and_renumbers_index(monkeypatch):
    monkeypatch.setattr(config_module.settings, "chunk_min_chars", 40)
    # Force _split_long_text's output regardless of real splitting math, to
    # isolate the per-piece filter + index-renumbering behavior on its own:
    # a tiny middle piece must be dropped without leaving a gap in
    # chunk_index numbering.
    monkeypatch.setattr(
        chunker,
        "_split_long_text",
        lambda text, target, max_chars, overlap: ["A" * 100, "tiny", "B" * 100],
    )

    sec_md = "# Section\n\n" + "z" * 200
    page = Page(
        url="https://x.com/p",
        markdown=sec_md + "\n\n" + ("extra " * 100),
        sections=[Section(id="longsec", tag="section", title="Long", chars=len(sec_md), markdown=sec_md)],
    )

    chunks = chunker.chunk_page(page, job_id="j", tenant_id="t", site_url="https://x.com")

    assert len(chunks) == 2
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert chunks[0].text == "A" * 100
    assert chunks[1].text == "B" * 100
