"""Orchestrates one sales-script generation run (claim -> graph -> persist,
mirroring app/main.py::_run_ingest's shape) and approval (approve -> chunk ->
embed -> index, mirroring app/ingestion/service.py::ingest_job's ownership of
chunk->embed->upsert).
"""

import asyncio
import logging
from typing import Optional

from .. import storage
from ..config import settings
from ..ingestion import store as ingestion_store
from ..ingestion.embedder import embed_documents
from ..models import SalesScriptRecord
from . import playbooks, sectioning, store
from .chunker import sales_script_to_chunks
from .errors import describe_failure
from .graph import graph

logger = logging.getLogger(__name__)


class SalesScriptNotFound(Exception):
    """No sales_scripts row exists for this tenant (+ site_url)."""


class SalesScriptWrongState(Exception):
    """A row exists but isn't in the state the caller expected (e.g. not
    pending_review when approving)."""


async def run_generation(tenant_id: str, site_url: str, job_id: str) -> None:
    """Background task body for POST /sales-script/{job_id}. Assumes the
    caller already claimed generation via store.claim_generation."""
    try:
        job_pages = await storage.load_pages(job_id, include_content=True)
        if job_pages is None or not job_pages.pages:
            raise ValueError(f"No persisted pages for job_id={job_id}")

        pages = sectioning.build_page_digests(
            job_pages, settings.sales_script_max_page_chars
        )
        if not pages:
            raise ValueError("No page content available to extract facts from")

        final_state = await asyncio.wait_for(
            graph.ainvoke(
                {
                    "tenant_id": tenant_id,
                    "job_id": job_id,
                    "site_url": site_url,
                    "pages": pages,
                    "facts": [],
                    "extract_failures": 0,
                    "extract_errors": [],
                    "profile": None,
                    "script": None,
                    "critique": None,
                    "revision_count": 0,
                    "max_revisions": settings.sales_script_max_revisions,
                }
            ),
            # Must stay under store._STALE_GENERATION (30 min): an untimed run
            # that outlives its own claim lets a retry spawn a second worker
            # that races this one on save_result/save_failure for the same row.
            timeout=settings.sales_script_run_timeout_seconds,
        )
        await store.save_result(
            tenant_id,
            site_url,
            final_state["script"],
            final_state["profile"],
            final_state["critique"],
            final_state["revision_count"],
        )
    except Exception as exc:
        logger.exception(
            "sales script generation failed for tenant_id=%s site_url=%s",
            tenant_id,
            site_url,
        )
        await store.save_failure(tenant_id, site_url, describe_failure(exc))


async def approve_and_index(
    tenant_id: str, site_url: Optional[str] = None
) -> tuple[SalesScriptRecord, int]:
    """Flip pending_review -> ready, then index the script's sections into
    the chunks table for retrieval via /query.

    Raises SalesScriptNotFound / SalesScriptWrongState for the endpoint to
    map to 404 / 409.
    """
    if site_url is None:
        current = await store.load_current(tenant_id)
        if current is None:
            raise SalesScriptNotFound(tenant_id)
        site_url = current.site_url

    record = await store.approve(tenant_id, site_url)
    if record is None:
        existing = await store.load_current(tenant_id, site_url)
        if existing is None:
            raise SalesScriptNotFound(f"{tenant_id}:{site_url}")
        raise SalesScriptWrongState(existing.status)

    # site_profile is None for rows generated before this column existed —
    # playbooks.get(None) falls back to the fully-generic "other" playbook.
    archetype = record.site_profile.archetype if record.site_profile else None
    playbook = playbooks.get(archetype)
    chunks = sales_script_to_chunks(
        record.script,
        tenant_id=tenant_id,
        job_id=record.job_id,
        site_url=site_url,
        playbook=playbook,
    )
    embeddings = (
        await asyncio.to_thread(embed_documents, [c.embedding_text for c in chunks])
        if chunks
        else []
    )
    indexed = await ingestion_store.replace_site_script_chunks(
        tenant_id, site_url, chunks, embeddings
    )
    return record, indexed
