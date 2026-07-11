"""Scraping + ingestion pipeline API.

Flow:
    POST /scrape                       -> start a crawl, return job_id
    GET  /scrape/{job_id}               -> live status; persists pages once crawl completes,
                                            then auto-triggers ingestion in the background
    GET  /scrape/{job_id}/pages         -> the persisted page index
    POST /ingest/{job_id}               -> (re-)run ingestion for a job synchronously
    POST /sales-script/{job_id}         -> kick off sales-script generation for that job's site
    GET  /sales-script/{tenant_id}      -> current sales script (any status) for a tenant
    POST /sales-script/{tenant_id}/approve -> approve a pending_review script; indexes it into chunks
    POST /query                         -> hybrid (vector + lexical) search over a tenant's chunks
    GET  /map?url=                      -> preview URLs before committing to a crawl
    GET  /health                        -> Firecrawl + database reachability
"""

import asyncio
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query

from . import db, scraper, storage
from .auth import require_internal_token
from .config import settings
from .ingestion import service as ingestion_service
from .ingestion import store as ingestion_store
from .ingestion.embedder import embed_query
from .models import (
    IngestResult,
    JobCreated,
    JobPages,
    JobState,
    MapResult,
    QueryRequest,
    QueryResponse,
    SalesScriptApproveResponse,
    SalesScriptGenerateResponse,
    SalesScriptRecord,
    ScrapeRequest,
)
from .salescript import service as salescript_service
from .salescript import store as salescript_store

app = FastAPI(
    title="Firecrawl Scraping + Ingestion Pipeline",
    description="Paste a website URL; crawl, chunk, and embed it for voice-agent Q&A and navigation.",
    version="0.2.0",
)


@app.on_event("startup")
async def on_startup() -> None:
    # Don't fail app startup if the DB isn't reachable yet — /health surfaces
    # this, and scraping still works without it. Ingestion just fails until fixed.
    try:
        await ingestion_store.get_pool()
        await storage.reconcile_interrupted_jobs()
    except Exception:
        pass


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await db.close_pool()


async def _run_ingest(job_id: str) -> None:
    """Background task: ingest a completed crawl, updating the job row."""
    state = await storage.load_job(job_id)
    if state is None:
        return
    try:
        chunk_count, question_count = await ingestion_service.ingest_job(job_id)
        state = await storage.load_job(job_id) or state
        state.ingest_status = "ingested"
        state.ingested_chunks = chunk_count
        state.ingested_questions = question_count
        state.ingest_error = None
    except Exception as exc:
        state = await storage.load_job(job_id) or state
        state.ingest_status = "ingest_failed"
        state.ingest_error = str(exc)
    await storage.save_job(state)


@app.get("/health")
async def health() -> dict:
    firecrawl_ok = await scraper.reachable()
    db_ok = True
    try:
        pool = await ingestion_store.get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception:
        db_ok = False
    return {
        "status": "ok" if firecrawl_ok and db_ok else "degraded",
        "firecrawl_url": settings.firecrawl_api_url,
        "firecrawl_reachable": firecrawl_ok,
        "database_reachable": db_ok,
    }


@app.post(
    "/scrape",
    response_model=JobCreated,
    status_code=202,
    dependencies=[Depends(require_internal_token)],
)
async def start_scrape(req: ScrapeRequest) -> JobCreated:
    limit = min(req.limit or settings.default_crawl_limit, settings.max_crawl_limit)
    wait_for = min(
        req.wait_for if req.wait_for is not None else settings.default_wait_for_ms,
        settings.max_wait_for_ms,
    )
    url = str(req.url)
    try:
        job_id = await scraper.start_crawl(
            url=url,
            limit=limit,
            wait_for_ms=wait_for,
            include_paths=req.include_paths,
            exclude_paths=req.exclude_paths,
        )
    except Exception as exc:  # surface Firecrawl connectivity/validation errors
        raise HTTPException(status_code=502, detail=f"Failed to start crawl: {exc}")

    await storage.create_job(job_id, url, req.tenant_id)
    return JobCreated(job_id=job_id, url=url, tenant_id=req.tenant_id)


@app.get(
    "/scrape/{job_id}", response_model=JobState, dependencies=[Depends(require_internal_token)]
)
async def get_scrape(job_id: str, background_tasks: BackgroundTasks) -> JobState:
    state = await storage.load_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    # Once persisted, trust stored state — no need to hit Firecrawl again.
    # (Ingestion may still be running in the background; the job row reflects
    # its progress on every load, so this keeps returning fresh status.)
    if state.persisted:
        return state

    try:
        live = await scraper.get_status(job_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to read crawl status: {exc}")

    state.status = live["status"]
    state.completed_pages = live["completed"]
    state.total_pages = live["total"]

    if state.status != "completed":
        await storage.save_job(state)
        return state

    # Two requests — even across different processes/replicas — can both
    # observe persisted=False and reach here at once right as the crawl
    # finishes. This atomic UPDATE lets only one of them win the race, so
    # only one persists pages and schedules ingestion.
    claimed = await storage.mark_persisting(job_id, live["total"])
    if claimed is None:
        return await storage.load_job(job_id) or state

    try:
        pages = scraper.to_pages(live["data"])
        summaries = await storage.persist_pages(job_id, pages)
    except Exception as exc:
        await storage.revert_persisting(job_id, str(exc))
        raise HTTPException(status_code=502, detail=f"Failed to persist pages: {exc}")

    claimed.completed_pages = len(summaries)
    if settings.auto_ingest:
        claimed.ingest_status = "ingesting"
        background_tasks.add_task(_run_ingest, job_id)

    await storage.save_job(claimed)
    return claimed


@app.get(
    "/scrape/{job_id}/pages",
    response_model=JobPages,
    dependencies=[Depends(require_internal_token)],
)
async def get_scrape_pages(
    job_id: str,
    include_content: bool = Query(
        default=False, description="Include full page markdown in the response"
    ),
) -> JobPages:
    pages = await storage.load_pages(job_id, include_content=include_content)
    if pages is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return pages


@app.post(
    "/ingest/{job_id}", response_model=IngestResult, dependencies=[Depends(require_internal_token)]
)
async def trigger_ingest(job_id: str) -> IngestResult:
    """(Re-)run ingestion for a completed crawl synchronously."""
    state = await storage.load_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if not state.persisted:
        raise HTTPException(status_code=409, detail="Crawl has not completed/persisted yet")

    state.ingest_status = "ingesting"
    await storage.save_job(state)
    try:
        chunk_count, question_count = await ingestion_service.ingest_job(job_id)
    except Exception as exc:
        state = await storage.load_job(job_id) or state
        state.ingest_status = "ingest_failed"
        state.ingest_error = str(exc)
        await storage.save_job(state)
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {exc}")

    state = await storage.load_job(job_id) or state
    state.ingest_status = "ingested"
    state.ingested_chunks = chunk_count
    state.ingested_questions = question_count
    state.ingest_error = None
    await storage.save_job(state)
    return IngestResult(
        job_id=job_id, chunk_count=chunk_count, question_count=question_count
    )


@app.post(
    "/sales-script/{job_id}",
    response_model=SalesScriptGenerateResponse,
    status_code=202,
    dependencies=[Depends(require_internal_token)],
)
async def start_sales_script(
    job_id: str, background_tasks: BackgroundTasks
) -> SalesScriptGenerateResponse:
    """Kick off sales-script generation for the site this job scraped.

    Only requires the crawl to be persisted (not ingested) — fact extraction
    reads pages directly, not the chunks table. Keyed by (tenant_id, site_url)
    like the chunks table, so this can be re-triggered for the same site
    across different crawl jobs; job_id is stored for provenance only.
    """
    if not settings.sales_script_enabled:
        raise HTTPException(status_code=403, detail="Sales script generation is disabled")

    state = await storage.load_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if not state.persisted:
        raise HTTPException(status_code=409, detail="Crawl has not completed/persisted yet")

    claimed = await salescript_store.claim_generation(state.tenant_id, state.url, job_id)
    if claimed is None:
        raise HTTPException(
            status_code=409,
            detail="Sales script generation already in progress for this site",
        )

    background_tasks.add_task(
        salescript_service.run_generation, state.tenant_id, state.url, job_id
    )
    return SalesScriptGenerateResponse(
        tenant_id=state.tenant_id, site_url=state.url, job_id=job_id, status="generating"
    )


@app.get(
    "/sales-script/{tenant_id}",
    response_model=SalesScriptRecord,
    dependencies=[Depends(require_internal_token)],
)
async def get_sales_script(
    tenant_id: str,
    site_url: Optional[str] = Query(
        default=None, description="Omit to get the tenant's most-recently-updated script"
    ),
) -> SalesScriptRecord:
    """Returns the record regardless of status — a human reviewer needs
    pending_review + critique to decide; a voice-gateway consumer checks
    status == "ready" itself before using it as call context."""
    record = await salescript_store.load_current(tenant_id, site_url)
    if record is None:
        raise HTTPException(status_code=404, detail="No sales script found for this tenant")
    return record


@app.post(
    "/sales-script/{tenant_id}/approve",
    response_model=SalesScriptApproveResponse,
    dependencies=[Depends(require_internal_token)],
)
async def approve_sales_script(
    tenant_id: str,
    site_url: Optional[str] = Query(
        default=None, description="Omit to approve the tenant's most-recently-updated script"
    ),
) -> SalesScriptApproveResponse:
    """Flips a pending_review script to ready and indexes its sections into
    the chunks table (kind="sales_script") so /query can surface them."""
    try:
        record, indexed = await salescript_service.approve_and_index(tenant_id, site_url)
    except salescript_service.SalesScriptNotFound:
        raise HTTPException(status_code=404, detail="No sales script found for this tenant")
    except salescript_service.SalesScriptWrongState as exc:
        raise HTTPException(
            status_code=409, detail=f"Sales script is not pending review (status={exc})"
        )
    return SalesScriptApproveResponse(
        tenant_id=tenant_id,
        site_url=record.site_url,
        status=record.status,
        indexed_chunks=indexed,
    )


@app.post(
    "/query", response_model=QueryResponse, dependencies=[Depends(require_internal_token)]
)
async def query(req: QueryRequest) -> QueryResponse:
    embedding = await asyncio.to_thread(embed_query, req.question)
    hits = await ingestion_store.search(
        tenant_id=req.tenant_id,
        embedding=embedding,
        question=req.question,
        top_k=req.top_k,
        site_url=req.site_url,
        page_url=req.page_url,
        hybrid=req.hybrid,
        debug=req.debug,
    )
    return QueryResponse(hits=hits)


@app.get(
    "/map", response_model=MapResult, dependencies=[Depends(require_internal_token)]
)
async def map_url(
    url: str = Query(..., description="Website URL to preview"),
    limit: int = Query(default=50, ge=1, le=500),
) -> MapResult:
    try:
        links = await scraper.map_site(url, limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to map site: {exc}")
    return MapResult(url=url, count=len(links), links=links)
