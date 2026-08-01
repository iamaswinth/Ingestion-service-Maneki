"""Orchestrates load -> chunk -> embed -> upsert for one completed crawl job."""

import asyncio
import logging

from .. import storage
from ..config import settings
from ..models import Chunk, Page, PageSummary
from . import store
from .chunker import chunk_page
from .embedder import embed_documents
from .product_enrichment import enrich_products
from .questions import build_question_chunks, generate_questions

logger = logging.getLogger(__name__)


def _to_page(summary: PageSummary) -> Page:
    return Page(
        url=summary.url,
        title=summary.title,
        description=summary.description,
        markdown=summary.markdown or "",
        sections=summary.sections or [],
    )


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks whose normalized text exactly duplicates one already kept.

    `only_main_content` strips headers/footers, but in-content boilerplate
    (repeated CTAs, cookie/consent copy, a feature strip reused on every
    page) still survives and gets chunked once per page it appears on.
    Untreated, identical text competes with itself for /query's top_k slots
    and doc2query pays LLM tokens to write near-identical questions for the
    same passage N times.

    Keeps the first occurrence in crawl order (page list order is Firecrawl's
    crawl order, so the canonical/earliest-discovered page wins) and drops
    later duplicates. Whitespace-and-case normalized so "Book a Demo" and
    "book a demo\\n" collapse to the same key without touching the kept
    chunk's actual text.
    """
    seen: set[str] = set()
    kept: list[Chunk] = []
    dropped = 0
    for chunk in chunks:
        key = " ".join(chunk.text.split()).lower()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(chunk)
    if dropped:
        logger.info("dropped %d duplicate-text chunks", dropped)
    return kept


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

    # Independent of chunking/embedding below: this only needs job_pages'
    # markdown (already loaded, include_content=True above) and the
    # deterministic products persist_pages already wrote for this site
    # before ingest_job ever ran. Fail-open (see product_enrichment.py's
    # module docstring) — never blocks the rest of ingestion.
    try:
        enriched_count = await enrich_products(job.tenant_id, job.url, job_id, job_pages)
        if enriched_count:
            logger.info(
                "product LLM fallback added products",
                extra={"job_id": job_id, "tenant_id": job.tenant_id, "count": enriched_count},
            )
    except Exception:
        logger.warning("product LLM fallback enrichment failed", exc_info=True)

    content_chunks = [
        chunk
        for summary in job_pages.pages
        for chunk in chunk_page(
            _to_page(summary), job_id=job_id, tenant_id=job.tenant_id, site_url=job.url
        )
    ]
    if settings.dedupe_chunks:
        content_chunks = _dedupe_chunks(content_chunks)
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
