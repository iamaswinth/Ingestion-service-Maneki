from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

# Our normalized job status. Firecrawl reports "scraping"/"completed"/"failed"/
# "cancelled"; we map anything else onto these.
JobStatus = Literal["scraping", "completed", "failed", "cancelled"]

# Ingestion (chunk -> embed -> upsert into pgvector) lifecycle, tracked
# separately from crawl status since it runs after the crawl completes.
IngestStatus = Literal["not_started", "ingesting", "ingested", "ingest_failed"]

ContentType = Literal[
    "hero", "feature", "pricing", "faq", "testimonial", "generic", "sales_script"
]

AnchorType = Literal["id", "text", "page"]

# Sales script generation (app/salescript/) lifecycle: a LangGraph run lands in
# pending_review; a separate human approval flips it to ready.
SalesScriptStatus = Literal[
    "not_started", "generating", "pending_review", "ready", "failed"
]


class ScrapeRequest(BaseModel):
    """A website the user pasted, plus optional crawl controls."""

    url: HttpUrl
    tenant_id: str = Field(
        ..., min_length=1, description="Owner of this crawl; scopes all ingested chunks"
    )
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
    tenant_id: str
    status: JobStatus = "scraping"


class JobState(BaseModel):
    job_id: str
    url: str
    tenant_id: str
    status: JobStatus
    completed_pages: int = 0
    total_pages: int = 0
    error: Optional[str] = None
    # True once completed pages have been written to disk (persist runs once).
    persisted: bool = False
    ingest_status: IngestStatus = "not_started"
    ingested_chunks: int = 0
    ingested_questions: int = 0
    ingest_error: Optional[str] = None
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
    description: Optional[str] = None
    chars: int
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


class Chunk(BaseModel):
    """A retrievable unit destined for the vector DB.

    `navigation` is always populated (id anchor, text-fragment anchor, or bare
    page URL) — this is what the voice agent uses to move the user around the
    site after speaking `text`.

    Two kinds of rows share this shape (doc2query):
    - kind="content": a real chunk; `embedding_text` is what gets embedded.
    - kind="question": a synthetic question a visitor might ask, pointing back
      at its content chunk via `parent_chunk_id`. It copies the parent's
      text/navigation/metadata (so a hit returns speakable content directly)
      but embeds the `question` string instead.
    """

    chunk_id: str
    tenant_id: str
    job_id: str
    site_url: str
    page_url: str
    section_id: Optional[str] = None
    parent_section_id: Optional[str] = None
    anchor_type: AnchorType
    navigation: str
    title: Optional[str] = None
    content_type: ContentType = "generic"
    chunk_index: int = 0
    text: str
    embedding_text: str
    kind: Literal["content", "question", "sales_script"] = "content"
    parent_chunk_id: Optional[str] = None
    question: Optional[str] = None


class IngestResult(BaseModel):
    job_id: str
    chunk_count: int
    question_count: int = 0


class QueryRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    site_url: Optional[str] = None
    page_url: Optional[str] = None
    # None => use settings.hybrid_search_enabled; explicit True/False overrides
    # per request, for A/B eval of hybrid vs. vector-only.
    hybrid: Optional[bool] = None
    # None => use settings.rerank_enabled; explicit True/False overrides per
    # request, for A/B eval of rerank vs. RRF/vector-only ranking.
    rerank: Optional[bool] = None
    # When true, populate QueryHit.vector_score/lexical_score/rerank_score for tuning.
    debug: bool = False


class QueryHit(BaseModel):
    text: str
    score: float
    page_url: str
    section_id: Optional[str] = None
    title: Optional[str] = None
    content_type: ContentType = "generic"
    anchor_type: AnchorType
    navigation: str
    # When a synthetic question vector produced this hit, the question that
    # matched — useful for debugging retrieval quality.
    matched_question: Optional[str] = None
    # Per-leg scores, populated only when QueryRequest.debug=True and the
    # hybrid path ran (cosine similarity / ts_rank_cd respectively).
    vector_score: Optional[float] = None
    lexical_score: Optional[float] = None
    # Cross-encoder relevance score (app/ingestion/reranker.py), populated
    # only when QueryRequest.debug=True and reranking ran.
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    hits: list[QueryHit]


# --- Sales script generation (app/salescript/) ---
# `extra="forbid"` on every model below is required so `.model_json_schema()`
# sets `additionalProperties: false` at every nesting level, which Anthropic's
# structured-output (json_schema) mode requires.


class ValueProp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pain_point: str
    value_prop: str


class ObjectionQA(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objection: str
    response: str
    # False when this is a known-important objection category (see the fixed
    # checklist in DRAFT_SCRIPT_SYSTEM_PROMPT) with no supporting facts on the
    # site — a disclosed gap for the human reviewer, not a hallucinated answer.
    covered: bool


DiscoveryStage = Literal["situation", "problem", "implication", "need_payoff"]


class DiscoveryQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: DiscoveryStage
    question: str


class ProofPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    # Free-text description of which objection or value prop this
    # reinforces, e.g. "value prop: faster onboarding" or "objection:
    # switching cost" — not an index, so it stays valid even as the model
    # revises other lists across critique loops.
    reinforces: str


class SalesScript(BaseModel):
    """The structured sales script a completed graph run produces — the unit
    a human reviews/approves, and that gets split into retrievable chunks
    (app/salescript/chunker.py) once approved."""

    model_config = ConfigDict(extra="forbid")

    opening_hook: str
    discovery_questions: list[DiscoveryQuestion]
    value_props: list[ValueProp]
    objection_handling: list[ObjectionQA]
    proof_points: list[ProofPoint]
    pricing_talk_track: str
    differentiators: str
    closing_cta: str
    # Short descriptions of what to listen for during discovery (e.g. "team
    # size", "current tool being replaced", "budget authority", "urgency/
    # timeline") for a future CRM/lead-capture handoff. Derived from the ICP,
    # not conversational script content — not indexed into chunks.
    qualification_signals: list[str]


class IcpProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ideal_customer: str
    top_pain_points: list[str]
    buying_triggers: list[str] = []


class SalesScriptCritique(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    issues: list[str] = []


class SalesScriptRecord(BaseModel):
    """One row of the sales_scripts table, keyed by (tenant_id, site_url)."""

    tenant_id: str
    site_url: str
    job_id: str
    status: SalesScriptStatus
    script: Optional[SalesScript] = None
    critique: Optional[SalesScriptCritique] = None
    revision_count: int = 0
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SalesScriptGenerateResponse(BaseModel):
    tenant_id: str
    site_url: str
    job_id: str
    status: SalesScriptStatus


class SalesScriptApproveResponse(BaseModel):
    tenant_id: str
    site_url: str
    status: SalesScriptStatus
    indexed_chunks: int
