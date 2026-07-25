"""Turns an approved SalesScript into retrievable Chunk rows (kind="sales_script").

Mirrors app/ingestion/chunker.py's per-item granularity (doc2query splits one
question per row; here value_props/objection_handling split one entry per
row) so /query can surface the single most relevant talking point rather than
the whole script at once.

There's no real on-page anchor for synthesized content, so these chunks reuse
the *existing* anchor_type="page" meaning (the bare site_url) rather than
adding a new AnchorType value.

Section titles come from the tenant's Playbook (app/salescript/playbooks.py)
rather than being hard-coded here, so a portfolio's chunks are titled "Rates
& Availability" instead of "Pricing", etc. `kind="sales_script"` and
`content_type="sales_script"` stay fixed regardless of archetype — those are
DB/wire values, not display text.
"""

import hashlib

from ..models import Chunk, SalesScript
from .playbooks import Playbook


def _chunk_id(tenant_id: str, site_url: str, section_key: str, idx: int) -> str:
    raw = f"{tenant_id}|{site_url}|sales_script|{section_key}|{idx}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _make_chunk(
    *,
    tenant_id: str,
    job_id: str,
    site_url: str,
    section_key: str,
    idx: int,
    title: str,
    text: str,
    chunk_index: int,
) -> Chunk:
    return Chunk(
        chunk_id=_chunk_id(tenant_id, site_url, section_key, idx),
        tenant_id=tenant_id,
        job_id=job_id,
        site_url=site_url,
        page_url=site_url,
        section_id=f"sales_script:{section_key}:{idx}",
        parent_section_id=None,
        anchor_type="page",
        navigation=site_url,
        title=title,
        content_type="sales_script",
        chunk_index=chunk_index,
        text=text,
        embedding_text=f"Agent script — {title}: {text}",
        kind="sales_script",
        parent_chunk_id=None,
        question=None,
    )


def sales_script_to_chunks(
    script: SalesScript, *, tenant_id: str, job_id: str, site_url: str, playbook: Playbook
) -> list[Chunk]:
    labels = playbook.section_labels
    chunks: list[Chunk] = []
    i = 0

    def add(section_key: str, idx: int, title: str, text: str) -> None:
        nonlocal i
        if text.strip():
            chunks.append(
                _make_chunk(
                    tenant_id=tenant_id,
                    job_id=job_id,
                    site_url=site_url,
                    section_key=section_key,
                    idx=idx,
                    title=title,
                    text=text,
                    chunk_index=i,
                )
            )
            i += 1

    add("opening_hook", 0, labels["opening_hook"], script.opening_hook)
    add(
        "discovery_questions",
        0,
        labels["discovery_questions"],
        "\n".join(f"- ({q.stage}) {q.question}" for q in script.discovery_questions),
    )
    for idx, vp in enumerate(script.value_props):
        add(
            "value_prop",
            idx,
            f"{labels['value_prop']}: {vp.pain_point}",
            vp.value_prop,
        )
    for idx, oq in enumerate(script.objection_handling):
        text = (
            f'If the visitor says: "{oq.objection}" — respond: {oq.response}'
            if oq.covered
            else f'If the visitor says: "{oq.objection}" — not covered by site '
            f'content, flagged for owner follow-up: {oq.response}'
        )
        add("objection", idx, f"{labels['objection']}: {oq.objection}", text)
    for idx, pp in enumerate(script.proof_points):
        add("proof_point", idx, f"{labels['proof_point']}: {pp.reinforces}", pp.claim)
    add("pricing_talk_track", 0, labels["pricing_talk_track"], script.pricing_talk_track)
    add("differentiators", 0, labels["differentiators"], script.differentiators)
    add("closing_cta", 0, labels["closing_cta"], script.closing_cta)

    return chunks
