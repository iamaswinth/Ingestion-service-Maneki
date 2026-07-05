"""Disk-backed persistence for jobs and scraped pages.

Layout per job:

    data/{job_id}/
        job.json              # JobState — survives restarts, keeps API stateless
        index.json            # list[PageSummary] (metadata only)
        pages/
            001-home.md       # one markdown file per page

Deliberately filesystem-based so multiple uvicorn workers on one box share
state with no in-memory registry. The loaders/writers are the only place that
knows the layout, so moving to Postgres + object storage later is contained.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import settings
from .models import JobPages, JobState, Page, PageSummary, Section


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    return settings.data_dir / job_id


def _slug(url: str, fallback: str) -> str:
    tail = url.rstrip("/").split("/")[-1] or fallback
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", tail).strip("-").lower()
    return (slug or fallback)[:60]


def create_job(job_id: str, url: str) -> JobState:
    _job_dir(job_id).mkdir(parents=True, exist_ok=True)
    state = JobState(
        job_id=job_id,
        url=url,
        status="scraping",
        created_at=_now(),
        updated_at=_now(),
    )
    save_job(state)
    return state


def save_job(state: JobState) -> None:
    state.updated_at = _now()
    path = _job_dir(state.job_id) / "job.json"
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")


def load_job(job_id: str) -> Optional[JobState]:
    path = _job_dir(job_id) / "job.json"
    if not path.exists():
        return None
    return JobState.model_validate_json(path.read_text(encoding="utf-8"))


def persist_pages(job_id: str, pages: list[Page]) -> list[PageSummary]:
    """Write markdown files + index.json for a completed crawl. Idempotent."""
    pages_dir = _job_dir(job_id) / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[PageSummary] = []
    for i, page in enumerate(pages, start=1):
        stem = f"{i:03d}-{_slug(page.url, f'page-{i}')}"
        filename = f"{stem}.md"
        front_matter = (
            f"---\nurl: {page.url}\ntitle: {page.title or ''}\n"
            f"description: {page.description or ''}\n---\n\n"
        )
        (pages_dir / filename).write_text(front_matter + page.markdown, encoding="utf-8")

        if page.sections:
            (pages_dir / f"{stem}.sections.json").write_text(
                json.dumps(
                    [s.model_dump() for s in page.sections], indent=2, ensure_ascii=False
                ),
                encoding="utf-8",
            )

        summaries.append(
            PageSummary(
                url=page.url,
                title=page.title,
                chars=len(page.markdown),
                file=f"pages/{filename}",
                section_count=len(page.sections),
            )
        )

    index_path = _job_dir(job_id) / "index.json"
    index_path.write_text(
        json.dumps([s.model_dump(exclude={"markdown"}) for s in summaries], indent=2),
        encoding="utf-8",
    )
    return summaries


def load_pages(job_id: str, include_content: bool = False) -> Optional[JobPages]:
    job = load_job(job_id)
    if job is None:
        return None
    index_path = _job_dir(job_id) / "index.json"
    if not index_path.exists():
        return JobPages(job_id=job_id, url=job.url, page_count=0, pages=[])

    raw = json.loads(index_path.read_text(encoding="utf-8"))
    summaries = [PageSummary(**item) for item in raw]
    if include_content:
        for s in summaries:
            file_path = _job_dir(job_id) / s.file
            if file_path.exists():
                s.markdown = file_path.read_text(encoding="utf-8")
            sections_path = file_path.with_suffix(".sections.json")
            if sections_path.exists():
                raw_sections = json.loads(sections_path.read_text(encoding="utf-8"))
                s.sections = [Section(**item) for item in raw_sections]

    return JobPages(
        job_id=job_id, url=job.url, page_count=len(summaries), pages=summaries
    )
