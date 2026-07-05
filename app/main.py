"""Scraping pipeline API.

Flow:
    POST /scrape               -> start a crawl, return job_id
    GET  /scrape/{job_id}      -> live status; persists pages once the crawl completes
    GET  /scrape/{job_id}/pages-> the persisted page index (handoff to ingestion)
    GET  /map?url=             -> preview URLs before committing to a crawl
    GET  /health               -> is Firecrawl reachable
"""

from fastapi import FastAPI, HTTPException, Query

from . import scraper, storage
from .config import settings
from .models import (
    JobCreated,
    JobPages,
    JobState,
    MapResult,
    ScrapeRequest,
)

app = FastAPI(
    title="Firecrawl Scraping Pipeline",
    description="Paste a website URL; crawl it to clean markdown for voice-agent ingestion.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    ok = await scraper.reachable()
    return {
        "status": "ok" if ok else "degraded",
        "firecrawl_url": settings.firecrawl_api_url,
        "firecrawl_reachable": ok,
    }


@app.post("/scrape", response_model=JobCreated, status_code=202)
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

    storage.create_job(job_id, url)
    return JobCreated(job_id=job_id, url=url)


@app.get("/scrape/{job_id}", response_model=JobState)
async def get_scrape(job_id: str) -> JobState:
    state = storage.load_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    # Once persisted, trust local state — no need to hit Firecrawl again.
    if state.persisted:
        return state

    try:
        live = await scraper.get_status(job_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to read crawl status: {exc}")

    state.status = live["status"]
    state.completed_pages = live["completed"]
    state.total_pages = live["total"]

    if state.status == "completed":
        pages = scraper.to_pages(live["data"])
        summaries = storage.persist_pages(job_id, pages)
        state.completed_pages = len(summaries)
        state.persisted = True

    storage.save_job(state)
    return state


@app.get("/scrape/{job_id}/pages", response_model=JobPages)
async def get_scrape_pages(
    job_id: str,
    include_content: bool = Query(
        default=False, description="Include full page markdown in the response"
    ),
) -> JobPages:
    pages = storage.load_pages(job_id, include_content=include_content)
    if pages is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return pages


@app.get("/map", response_model=MapResult)
async def map_url(
    url: str = Query(..., description="Website URL to preview"),
    limit: int = Query(default=50, ge=1, le=500),
) -> MapResult:
    try:
        links = await scraper.map_site(url, limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to map site: {exc}")
    return MapResult(url=url, count=len(links), links=links)
