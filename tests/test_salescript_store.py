"""State-machine tests for app/salescript/store.py's claim/generating/
failed/pending_review/ready lifecycle, plus one end-to-end approve->index
test through app/salescript/service.py. No LLM/graph calls anywhere here —
store.py's functions are pure DB operations, and the approve->index test
monkeypatches embed_documents to avoid the real fastembed model.
"""

import uuid
from datetime import datetime, timedelta, timezone

from app.ingestion import store as ingestion_store
from app.models import SalesScript
from app.salescript import service as salescript_service
from app.salescript import store as salescript_store


def _minimal_profile(archetype: str = "portfolio") -> dict:
    """A minimal but schema-valid SiteProfile payload (app/models.py)."""
    return {
        "archetype": archetype,
        "reasoning": "The site is a single person's project showcase.",
        "audience": "prospective clients evaluating the person's work",
        "visitor_goals": ["see recent work", "check availability"],
        "conversion_triggers": ["seeing relevant past work", "clear rates"],
        "primary_action": "get in touch about a project",
        "tone": "warm and direct",
        "publishes_pricing": False,
    }


def _minimal_script() -> dict:
    """A minimal but schema-valid SalesScript payload — fields taken
    directly from app/salescript/chunker.py::sales_script_to_chunks's usage
    (opening_hook, discovery_questions, value_props, objection_handling,
    proof_points, pricing_talk_track, differentiators, closing_cta) plus
    the required qualification_signals field."""
    return {
        "opening_hook": "Hi, thanks for stopping by — what brought you here today?",
        "discovery_questions": [
            {"stage": "situation", "question": "What does your team use today?"}
        ],
        "value_props": [{"pain_point": "slow onboarding", "value_prop": "same-day setup"}],
        "objection_handling": [
            {"objection": "too expensive", "response": "we have a free tier", "covered": True}
        ],
        "proof_points": [{"claim": "500+ teams onboarded", "reinforces": "value prop: setup speed"}],
        "pricing_talk_track": "Plans start free, pro is $10/mo.",
        "differentiators": "We're the only one with same-day setup.",
        "closing_cta": "Want to start a free trial today?",
        "qualification_signals": ["team size", "current tool"],
    }


async def test_claim_generation_succeeds_for_fresh_pair(pg_tenant):
    tenant_id, site_url = pg_tenant
    claimed = await salescript_store.claim_generation(tenant_id, site_url, "job-1")
    assert claimed is True

    record = await salescript_store.load_current(tenant_id, site_url)
    assert record is not None
    assert record.status == "generating"


async def test_second_claim_blocked_while_fresh_and_generating(pg_tenant):
    tenant_id, site_url = pg_tenant
    assert await salescript_store.claim_generation(tenant_id, site_url, "job-1") is True
    assert await salescript_store.claim_generation(tenant_id, site_url, "job-2") is False


async def test_claim_fail_reclaim(pg_tenant):
    """The core ask: claim -> fail -> re-claim succeeds."""
    tenant_id, site_url = pg_tenant
    assert await salescript_store.claim_generation(tenant_id, site_url, "job-1") is True

    await salescript_store.save_failure(tenant_id, site_url, "simulated LLM failure")
    record = await salescript_store.load_current(tenant_id, site_url)
    assert record.status == "failed"
    assert record.error == "simulated LLM failure"

    reclaimed = await salescript_store.claim_generation(tenant_id, site_url, "job-2")
    assert reclaimed is True
    record = await salescript_store.load_current(tenant_id, site_url)
    assert record.status == "generating"
    assert record.job_id == "job-2"


async def test_stale_generating_row_can_be_reclaimed_without_failure(pg_tenant):
    """Crash-mid-generation path: a 'generating' row with no live worker
    behind it (never reached save_result/save_failure) still self-heals via
    the staleness window, distinct from the clean-failure path above."""
    tenant_id, site_url = pg_tenant
    assert await salescript_store.claim_generation(tenant_id, site_url, "job-1") is True

    pool = await salescript_store._pool()
    stale_time = datetime.now(timezone.utc) - timedelta(hours=1)
    await pool.execute(
        "UPDATE sales_scripts SET updated_at = $3 WHERE tenant_id = $1 AND site_url = $2",
        tenant_id, site_url, stale_time,
    )

    reclaimed = await salescript_store.claim_generation(tenant_id, site_url, "job-2")
    assert reclaimed is True
    record = await salescript_store.load_current(tenant_id, site_url)
    assert record.job_id == "job-2"


async def test_reconcile_interrupted_generations(pg_tenant):
    tenant_id, site_url = pg_tenant
    assert await salescript_store.claim_generation(tenant_id, site_url, "job-1") is True

    count = await salescript_store.reconcile_interrupted_generations()
    assert count >= 1

    record = await salescript_store.load_current(tenant_id, site_url)
    assert record.status == "failed"
    assert record.error == "interrupted by server restart"

    # No stuck rows left -> running it again finds nothing new for this row.
    await salescript_store.claim_generation(tenant_id, site_url, "job-2")
    await salescript_store.save_result(tenant_id, site_url, _minimal_script(), None, None, 0)
    count2 = await salescript_store.reconcile_interrupted_generations()
    record2 = await salescript_store.load_current(tenant_id, site_url)
    assert record2.status == "pending_review"  # untouched by reconcile


async def test_approve_flips_pending_review_to_ready_and_guards_wrong_state(pg_tenant):
    tenant_id, site_url = pg_tenant
    await salescript_store.claim_generation(tenant_id, site_url, "job-1")
    await salescript_store.save_result(tenant_id, site_url, _minimal_script(), None, None, 0)

    approved = await salescript_store.approve(tenant_id, site_url)
    assert approved is not None
    assert approved.status == "ready"

    # Already ready, not pending_review -> second approve is a no-op miss.
    second = await salescript_store.approve(tenant_id, site_url)
    assert second is None


async def test_save_result_round_trips_site_profile(pg_tenant):
    tenant_id, site_url = pg_tenant
    await salescript_store.claim_generation(tenant_id, site_url, "job-1")
    await salescript_store.save_result(
        tenant_id, site_url, _minimal_script(), _minimal_profile("portfolio"), None, 0
    )

    record = await salescript_store.load_current(tenant_id, site_url)
    assert record.site_profile is not None
    assert record.site_profile.archetype == "portfolio"
    assert record.site_profile.publishes_pricing is False


async def test_load_current_tolerates_null_site_profile(pg_tenant):
    """A row saved before this column existed (or a save_result call that
    passes None) must still load — a pre-existing sales_scripts row with
    site_profile IS NULL is exactly this case."""
    tenant_id, site_url = pg_tenant
    await salescript_store.claim_generation(tenant_id, site_url, "job-1")
    await salescript_store.save_result(tenant_id, site_url, _minimal_script(), None, None, 0)

    record = await salescript_store.load_current(tenant_id, site_url)
    assert record.site_profile is None


async def test_approve_and_index_writes_sales_script_chunks(pg_tenant, monkeypatch):
    tenant_id, site_url = pg_tenant
    await salescript_store.claim_generation(tenant_id, site_url, "job-1")
    await salescript_store.save_result(tenant_id, site_url, _minimal_script(), None, None, 0)

    # pgvector's cosine distance (<=>) is undefined for an all-zero vector,
    # so the fake embedding must be a real (non-zero) unit vector.
    fake_vector = [1.0] + [0.0] * 383
    fake_calls = []

    def _fake_embed_documents(texts: list[str]) -> list[list[float]]:
        fake_calls.append(texts)
        return [fake_vector for _ in texts]

    monkeypatch.setattr(salescript_service, "embed_documents", _fake_embed_documents)

    try:
        record, indexed = await salescript_service.approve_and_index(tenant_id, site_url)

        assert record.status == "ready"
        assert indexed > 0
        assert fake_calls, "embed_documents should have been called instead of the real model"

        hits = await ingestion_store.search(
            tenant_id=tenant_id, embedding=fake_vector, question="pricing",
            top_k=indexed, hybrid=False,
        )
        assert any(h.text == _minimal_script()["pricing_talk_track"] for h in hits)
    finally:
        await ingestion_store.delete_site_chunks(tenant_id, site_url)
        pool = await ingestion_store.get_pool()
        await pool.execute(
            "DELETE FROM chunks WHERE tenant_id = $1 AND site_url = $2 AND kind = 'sales_script'",
            tenant_id, site_url,
        )


async def test_approve_and_index_uses_archetype_playbook_labels(pg_tenant, monkeypatch):
    """A portfolio-archetype script's pricing_talk_track chunk should be
    titled from the portfolio playbook ("Rates & Availability"), not the
    generic/SaaS "Pricing" label — proves approve_and_index resolves the
    stored site_profile into the right Playbook before chunking."""
    tenant_id, site_url = pg_tenant
    await salescript_store.claim_generation(tenant_id, site_url, "job-1")
    await salescript_store.save_result(
        tenant_id, site_url, _minimal_script(), _minimal_profile("portfolio"), None, 0
    )

    fake_vector = [1.0] + [0.0] * 383
    monkeypatch.setattr(
        salescript_service, "embed_documents", lambda texts: [fake_vector for _ in texts]
    )

    try:
        record, indexed = await salescript_service.approve_and_index(tenant_id, site_url)
        assert indexed > 0

        hits = await ingestion_store.search(
            tenant_id=tenant_id, embedding=fake_vector, question="pricing",
            top_k=indexed, hybrid=False,
        )
        assert any(h.title == "Rates & Availability" for h in hits)
        assert not any(h.title == "Pricing" for h in hits)
    finally:
        await ingestion_store.delete_site_chunks(tenant_id, site_url)
        pool = await ingestion_store.get_pool()
        await pool.execute(
            "DELETE FROM chunks WHERE tenant_id = $1 AND site_url = $2 AND kind = 'sales_script'",
            tenant_id, site_url,
        )
