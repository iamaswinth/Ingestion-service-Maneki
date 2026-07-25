"""Unit tests for app/ingestion/questions.py's pure chunk-building helper --
no I/O, no live infra, no LLM calls (generate_questions itself is exercised
against the real Anthropic API only manually, per the module's fail-open
design).
"""

from app.ingestion.questions import build_question_chunks
from app.models import Chunk


def _parent(**overrides) -> Chunk:
    base = dict(
        chunk_id="parent-1",
        tenant_id="t",
        job_id="j",
        site_url="https://x.com/",
        page_url="https://x.com/pricing",
        section_id="pricing",
        anchor_type="id",
        navigation="https://x.com/pricing#pricing",
        chunk_index=0,
        text="We charge $10 per month for the pro plan.",
        embedding_text="Pricing: We charge $10 per month for the pro plan.",
    )
    base.update(overrides)
    return Chunk(**base)


def test_question_rows_carry_no_text():
    # Question rows must not duplicate the parent's full text -- a hit on one
    # is resolved back to the parent's real content via a JOIN at query time
    # (app/ingestion/store.py), not a second stored copy of it.
    parent = _parent()
    out = build_question_chunks([parent], {parent.chunk_id: ["How much does it cost?"]})

    assert len(out) == 1
    assert out[0].text == ""
    assert out[0].question == "How much does it cost?"
    assert out[0].embedding_text == "How much does it cost?"
    assert out[0].kind == "question"
    assert out[0].parent_chunk_id == parent.chunk_id


def test_question_chunk_ids_differ_across_site_url():
    # Two parents that are otherwise identical (same tenant/page/section/
    # index) but crawled under different site_urls must not produce colliding
    # question-row ids -- same collision hazard as content chunk_ids (see
    # chunker.make_chunk_id's docstring), since build_question_chunks reuses
    # that same id function.
    parent_a = _parent(chunk_id="a", site_url="https://x.com/")
    parent_b = _parent(chunk_id="b", site_url="https://x.com/docs")

    out_a = build_question_chunks([parent_a], {"a": ["question?"]})
    out_b = build_question_chunks([parent_b], {"b": ["question?"]})

    assert out_a[0].chunk_id != out_b[0].chunk_id


def test_multiple_questions_get_distinct_ids():
    parent = _parent()
    out = build_question_chunks(
        [parent], {parent.chunk_id: ["Question one?", "Question two?"]}
    )

    assert len(out) == 2
    assert out[0].chunk_id != out[1].chunk_id
