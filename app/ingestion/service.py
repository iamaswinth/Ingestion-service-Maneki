"""Orchestrates load -> chunk -> embed -> upsert for one completed crawl job."""

import asyncio
import logging

from .. import storage
from ..models import Page, PageSummary
from . import store
from .chunker import chunk_page
from .embedder import embed_documents
from .questions import build_question_chunks, generate_questions

logger = logging.getLogger(__name__)


def _to_page(summary: PageSummary) -> Page:
    return Page(
        url=summary.url,
        title=summary.title,
        markdown=summary.markdown or "",
        sections=summary.sections or [],
    )


async def ingest_job(job_id: str) -> tuple[int, int]:
    """Chunk, embed, and upsert every persisted page for a job.

    Returns (content_chunk_count, question_count).

    Idempotent and re-crawl-safe: existing chunks for this *site* (not just this
    job) are replaced atomically with the new set, so re-running (e.g. after a
    chunker change, or a fresh crawl of a site whose content changed) never
    leaves stale rows from pages/sections that no longer exist.
    """
    job = await storage.load_job(job_id)
    if job is None:
        raise ValueError(f"Unknown job_id: {job_id}")

    job_pages = await storage.load_pages(job_id, include_content=True)

    if job_pages is None or not job_pages.pages:
        # Nothing scraped for this job — still clear out any chunks left over
        # from a previous crawl of this site (e.g. the site now 404s).
        await store.delete_site_chunks(job.tenant_id, job.url)
        logger.warning(
            "ingest produced 0 chunks: no persisted pages",
            extra={"job_id": job_id, "tenant_id": job.tenant_id, "site_url": job.url},
        )
        return 0, 0

    content_chunks = [
        chunk
        for summary in job_pages.pages
        for chunk in chunk_page(
            _to_page(summary), job_id=job_id, tenant_id=job.tenant_id, site_url=job.url
        )
    ]
    if not content_chunks:
        await store.delete_site_chunks(job.tenant_id, job.url)
        logger.warning(
            "ingest produced 0 chunks: chunker returned nothing for every page",
            extra={
                "job_id": job_id,
                "tenant_id": job.tenant_id,
                "site_url": job.url,
                "page_count": len(job_pages.pages),
            },
        )
        return 0, 0

    # Doc2query: synthetic visitor questions as extra vectors (fail-open —
    # a generation failure never blocks the ingest).
    questions = await generate_questions(content_chunks)
    question_chunks = build_question_chunks(content_chunks, questions)

    all_chunks = content_chunks + question_chunks
    texts = [c.embedding_text for c in all_chunks]
    embeddings = await asyncio.to_thread(embed_documents, texts)
    await store.replace_site_chunks(job.tenant_id, job.url, all_chunks, embeddings)

    logger.info(
        "ingest complete",
        extra={
            "job_id": job_id,
            "tenant_id": job.tenant_id,
            "site_url": job.url,
            "page_count": len(job_pages.pages),
            "content_chunk_count": len(content_chunks),
            "question_chunk_count": len(question_chunks),
        },
    )
    return len(content_chunks), len(question_chunks)
