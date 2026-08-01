from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

# Our normalized job status. Firecrawl reports "scraping"/"completed"/"failed"/
# "cancelled"; we map anything else onto these.
JobStatus = Literal["scraping", "completed", "failed", "cancelled"]

# Ingestion (chunk -> embed -> upsert into pgvector) lifecycle, tracked
# separately from crawl status since it runs after the crawl completes.
IngestStatus = Literal["not_started", "ingesting", "ingested", "ingest_failed"]

ContentType = Literal[
    "hero",
    "feature",
    "pricing",
    "faq",
    "testimonial",
    "about",
    "contact",
    "project",
    "generic",
    "sales_script",
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


class PageLink(BaseModel):
    """An in-content <a> on a crawled page pointing at a different page of
    the same site (app/links.py::extract_links). Backs click-based
    navigation: the voice agent only asks the widget to click a real link
    when the target is one of these, never a selector it invented itself.
    """

    target: str  # absolute URL as resolved at crawl time (query/hash kept)
    target_key: str  # app/links.py::page_key(target) — the matching key
    text: str  # normalized visible anchor text (or aria-label/title/alt)


# Which extraction path produced a Product — "jsonld" (schema.org/Product,
# app/products.py) and "og" (Open Graph product:* tags, same module) are both
# deterministic; "llm" is the async Haiku fallback
# (app/ingestion/product_enrichment.py) for a page with neither signal. Bare
# str would lose the ability to filter/debug by source, but a Literal keeps
# call sites honest about which paths actually exist.
ProductSource = Literal["jsonld", "og", "llm"]


class Product(BaseModel):
    """A product extracted from one crawled page — app/products.py's
    deterministic JSON-LD/OG extraction, or app/ingestion/product_enrichment.py's
    LLM fallback. Backs GET /products/{tenant_id}: catalog search for
    "suggest alternatives" and grounding "what's in stock" in real data
    instead of retrieval prose.

    `product_key` is deliberately Optional here: extraction
    (app/products.py::extract_products) runs inside app/scraper.py::to_pages,
    which has no tenant_id/site_url in scope to hash into a key. It is filled
    in by app/storage.py at persistence time (see
    app/products.py::make_product_key) and is always populated by the time a
    Product reaches an API response.
    """

    product_key: Optional[str] = None
    page_url: str
    name: str
    sku: Optional[str] = None
    price_amount: Optional[Decimal] = None
    price_currency: Optional[str] = None
    # Raw schema.org (e.g. "https://schema.org/InStock") or OG
    # (product:availability) value, intentionally not normalized to a fixed
    # enum — platforms don't agree on one, and a bare str never invalidates
    # on a value this code hasn't seen yet.
    availability: Optional[str] = None
    image_url: Optional[str] = None
    description: Optional[str] = None
    # Size/color/variant labels — no fixed shape across platforms.
    attributes: dict[str, str] = {}
    source: ProductSource


# What a captured control does. "other" is a deliberate catch-all (mirrors
# SiteArchetype's "other") — an action's usefulness as a click target isn't
# limited to the kinds named here.
ActionKind = Literal[
    "add_to_cart",
    "buy_now",
    "checkout",
    "view_cart",
    "select_variant",
    "quantity",
    "search",
    "filter",
    "other",
]


class PageAction(BaseModel):
    """A clickable control captured on one page (app/actions.py). Backs
    click-based cart actions the same way PageLink backs click-based
    navigation — the agent may only ever act on a handle ingestion actually
    verified exists on the page, never invent a selector."""

    kind: ActionKind
    label: str  # accessible name: text -> aria-label -> title -> img alt
    role: str  # "button" | "link" | "input" | "select"
    selectors: list[str]  # priority-ordered fallback bundle, most-specific first
    # Only ever populated for a single-product page — see app/actions.py's
    # module docstring for the documented multi-product-page limitation.
    product_key: Optional[str] = None
    section_id: Optional[str] = None


class Page(BaseModel):
    """One scraped page — the unit handed off to the ingestion step."""

    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    markdown: str
    sections: list[Section] = []
    links: list[PageLink] = []
    products: list[Product] = []
    page_actions: list[PageAction] = []


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


class PageLinksRecord(BaseModel):
    """Links captured from one page of a tenant's most recent completed
    crawl (app/storage.py::load_page_links)."""

    job_id: str
    site_url: str
    page_url: str
    links: list[PageLink] = []


class PageLinksResponse(BaseModel):
    tenant_id: str
    site_url: str
    page_url: str
    job_id: str
    links: list[PageLink]


class PageActionsRecord(BaseModel):
    """Clickable controls captured from one page of a tenant's most recent
    completed crawl (app/storage.py::load_page_actions) — mirrors
    PageLinksRecord exactly."""

    job_id: str
    site_url: str
    page_url: str
    page_actions: list[PageAction] = []


class PageActionsResponse(BaseModel):
    tenant_id: str
    site_url: str
    page_url: str
    job_id: str
    page_actions: list[PageAction]


class ProductsResponse(BaseModel):
    """GET /products/{tenant_id} — catalog search backing "suggest
    alternatives" and grounding "what's in stock" in real data. Not scoped
    to one page/job like PageLinksResponse/PageActionsResponse: a product
    search spans the whole tenant catalog."""

    tenant_id: str
    products: list[Product]


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
    # Constrain to one content_type (e.g. "faq") or one kind
    # ("content"/"question"/"sales_script") — orthogonal filters, both
    # None by default (no constraint). See
    # app/ingestion/store.py::_build_conditions for precedence when kind
    # is given explicitly.
    content_type: Optional[ContentType] = None
    kind: Optional[Literal["content", "question", "sales_script"]] = None


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
    # False when this is a known-important concern category (see the
    # archetype's playbook.concern_checklist in app/salescript/playbooks.py)
    # with no supporting facts on the site — a disclosed gap for the human
    # reviewer, not a hallucinated answer.
    covered: bool


# SPIN stages (situation/problem/implication/need_payoff) fit an enterprise
# sales discovery flow; context/goal/fit are the neutral equivalents used by
# non-sales playbooks (portfolio, creator_media, nonprofit, etc — see
# app/salescript/playbooks.py). Which stages a given script actually uses is
# decided by the playbook, not by this type — it's widened, not narrowed, so
# older data with the SPIN stages still validates.
DiscoveryStage = Literal[
    "situation", "problem", "implication", "need_payoff", "context", "goal", "fit"
]


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
    # timeline" for a SaaS site; "project scope", "timeline", "budget range"
    # for a portfolio) for a future CRM/lead-capture handoff. Derived from the
    # SiteProfile, not conversational script content — not indexed into chunks.
    qualification_signals: list[str]


# What kind of site this is — decides which playbook (app/salescript/
# playbooks.py) shapes the draft/critique prompts. "other" is a deliberate,
# fully-generic fallback rather than forcing a bad fit onto a site that
# doesn't match any of the named archetypes.
SiteArchetype = Literal[
    "saas_product",
    "ecommerce",
    "local_service",
    "professional_services",
    "portfolio",
    "creator_media",
    "nonprofit",
    "other",
]


class SiteProfile(BaseModel):
    """What profile_site (app/salescript/graph.py) infers about the crawled
    site before drafting — replaces the old IcpProfile, which assumed every
    tenant was a company with a buying committee to profile. This is inferred
    from the crawl itself, not read from tenant_config (ingestion has no
    dependency on api-gateway)."""

    model_config = ConfigDict(extra="forbid")

    archetype: SiteArchetype
    # One sentence justifying the archetype call, for the human reviewer.
    reasoning: str
    # Who visits this site (e.g. "recruiters and prospective clients
    # evaluating the designer's work").
    audience: str
    # What visitors came to do (was: top_pain_points — "pain point" presumes
    # a problem being sold against, which doesn't fit e.g. a portfolio).
    visitor_goals: list[str]
    # What makes a visitor act (was: buying_triggers).
    conversion_triggers: list[str]
    # The one thing this site most wants a visitor to do (book a call, start
    # a project enquiry, buy, donate, subscribe, ...).
    primary_action: str
    # How the site talks about itself (e.g. "playful and informal", "terse
    # and technical", "warm and reassuring") — grounds the script's voice.
    tone: str
    # Whether the site actually states prices/rates anywhere, vs. only
    # "contact us" — lets draft_script write a confident bridge instead of a
    # vague deflection when there's genuinely nothing to quote.
    publishes_pricing: bool


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
    # Kept in its own column rather than folded into `script` — see
    # app/salescript/store.py. None for rows generated before this field
    # existed; callers should treat that the same as archetype="other".
    site_profile: Optional[SiteProfile] = None
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
