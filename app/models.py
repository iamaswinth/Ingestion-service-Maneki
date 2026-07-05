from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl

# Our normalized job status. Firecrawl reports "scraping"/"completed"/"failed"/
# "cancelled"; we map anything else onto these.
JobStatus = Literal["scraping", "completed", "failed", "cancelled"]


class ScrapeRequest(BaseModel):
    """A website the user pasted, plus optional crawl controls."""

    url: HttpUrl
    limit: Optional[int] = Field(
        default=None, ge=1, description="Max pages to crawl (capped by MAX_CRAWL_LIMIT)"
    )
    include_paths: Optional[list[str]] = Field(
        default=None,
        description='Only crawl URLs whose path matches these regexes, e.g. ["/docs/.*"]',
    )
    exclude_paths: Optional[list[str]] = Field(
        default=None,
        description='Skip URLs whose path matches these regexes, e.g. ["/blog/.*"]',
    )
    wait_for: Optional[int] = Field(
        default=None,
        ge=0,
        description="Milliseconds to wait for JS to render each page. Defaults to "
        "DEFAULT_WAIT_FOR_MS; raise for slow client-side-rendered sites.",
    )


class JobCreated(BaseModel):
    job_id: str
    url: str
    status: JobStatus = "scraping"


class JobState(BaseModel):
    job_id: str
    url: str
    status: JobStatus
    completed_pages: int = 0
    total_pages: int = 0
    error: Optional[str] = None
    # True once completed pages have been written to disk (persist runs once).
    persisted: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class Section(BaseModel):
    """A page region identified by its HTML id (e.g. id="pricing").

    Hierarchical: `children` holds nested id'd blocks (e.g. faq -> faq-q-01).
    `markdown` is this section's *own* content, with nested sections removed,
    so parent and children never duplicate text.
    """

    id: str
    tag: str  # the HTML tag that carried the id: section / div / article ...
    title: Optional[str] = None  # first heading in the section, else humanized id
    chars: int
    markdown: str
    children: list["Section"] = []


class Page(BaseModel):
    """One scraped page — the unit handed off to the ingestion step."""

    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    markdown: str
    sections: list[Section] = []


class PageSummary(BaseModel):
    url: str
    title: Optional[str] = None
    chars: int
    file: str
    section_count: int = 0
    markdown: Optional[str] = None  # populated only when include_content=true
    sections: Optional[list[Section]] = None  # populated only when include_content=true


class JobPages(BaseModel):
    job_id: str
    url: str
    page_count: int
    pages: list[PageSummary]


class MapResult(BaseModel):
    url: str
    count: int
    links: list[str]
