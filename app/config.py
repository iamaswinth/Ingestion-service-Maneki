import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Base URL of the Firecrawl instance. Defaults to the local self-hosted
    # stack; set to https://api.firecrawl.dev to use the cloud service.
    firecrawl_api_url: str = "http://localhost:3002"

    # Optional for self-hosted (auth is bypassed). Required only for cloud.
    # The SDK still wants a non-empty string, so we default to a placeholder.
    firecrawl_api_key: str = "self-hosted"

    # Default page limit per crawl if the request does not specify one.
    default_crawl_limit: int = 25

    # Hard ceiling so a user cannot kick off an unbounded crawl.
    max_crawl_limit: int = 100

    # Milliseconds to wait for client-side JS to render before capturing a page.
    # Many modern sites (React/Next/Vue) return an empty shell without this.
    # Self-hosted Firecrawl has no smart-wait, so a fixed wait is the reliable fix.
    default_wait_for_ms: int = 3000

    # Ceiling on per-request wait override, to bound crawl time.
    max_wait_for_ms: int = 15000

    # Postgres+pgvector connection string (Neon in prod, local Docker for dev).
    database_url: str = (
        "postgresql://postgres:postgres@localhost:5433/postgres"
    )

    # Shared service-to-service secret required from any internal caller
    # (the API gateway's forward proxy and the voice runtime both present
    # it — see app/auth.py). This app has no other auth of its own, so an
    # unset token means every request is rejected, not "auth is skipped."
    internal_service_token: str = ""

    # Local, free embedding model. 384 dims — must match the `vector(384)`
    # column in app/ingestion/store.py if ever changed.
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Explicit cache path (rather than fastembed's OS-temp-dir default) so the
    # model baked into the Docker image at build time is guaranteed to still be
    # there at runtime, even if the platform mounts /tmp as ephemeral tmpfs.
    embedding_cache_dir: str = ".fastembed_cache"

    # Automatically ingest a crawl into the vector DB once it finishes scraping.
    auto_ingest: bool = True

    # Background loop that finishes crawls no client has polled since Firecrawl
    # completed them (see _poll_open_jobs in app/main.py). Off switch for tests.
    scrape_poll_enabled: bool = True
    scrape_poll_interval_seconds: int = 15

    # Chunking knobs (app/ingestion/chunker.py).
    chunk_target_chars: int = 1000
    chunk_max_chars: int = 1500
    chunk_overlap_chars: int = 150
    chunk_min_chars: int = 40

    # Drop chunks whose normalized text exactly duplicates one already kept for
    # this (tenant_id, site_url) — repeated CTA/cookie/footer-in-content blocks
    # that only_main_content doesn't strip. Off switch for a site that
    # legitimately needs identical text indexed under multiple nav targets.
    dedupe_chunks: bool = True

    # --- Doc2query: AI-generated questions per chunk (app/ingestion/questions.py) ---
    # Anthropic API key. Empty => question generation is silently skipped and
    # the pipeline behaves exactly as before (pure dense retrieval).
    anthropic_api_key: str = ""
    question_gen_enabled: bool = True
    question_gen_model: str = "claude-haiku-4-5"
    # 3->2: diminishing returns per extra question, and fewer question rows
    # per chunk directly relieves the hnsw.ef_search dilution below (each
    # doc2query row is a distinct HNSW candidate competing with real content
    # for the same fixed ef_search budget).
    questions_per_chunk: int = 2
    # How many chunks to pack into one LLM request.
    question_gen_batch_size: int = 12
    # Anthropic SDK default is 600s; a stalled batch shouldn't hold an ingest
    # hostage for ten minutes. Fail-open design (see questions.py) means a
    # timeout here just means fewer questions, never a failed ingest.
    question_gen_timeout_seconds: float = 60.0

    # --- Product LLM fallback (app/ingestion/product_enrichment.py) ---
    # For a page with neither JSON-LD nor Open Graph product signal
    # (app/products.py) — reuses anthropic_api_key above. Silently skipped
    # (same fail-open pattern as doc2query) with no key set.
    product_llm_fallback_enabled: bool = True
    product_llm_fallback_model: str = "claude-haiku-4-5"
    product_llm_fallback_timeout_seconds: float = 60.0

    # --- Hybrid retrieval: dense vector + Postgres full-text, fused via RRF ---
    hybrid_search_enabled: bool = True
    hybrid_rrf_k: int = 60
    hybrid_vector_weight: float = 1.0
    hybrid_lexical_weight: float = 1.0
    # Mirrors the vector leg's existing top_k*8 oversample factor.
    hybrid_candidate_multiplier: int = 8
    hybrid_candidate_floor: int = 50

    # --- Reranking: cross-encoder re-scores retrieval results before top_k ---
    rerank_enabled: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Extra candidates fetched (from whichever retrieval path ran) before
    # truncating to the requested top_k, so the reranker has real room to
    # promote a good match that RRF/cosine ranked lower — same intent as
    # hybrid_candidate_multiplier, opposite direction from hybrid_candidate_floor
    # (this bounds a maximum, not a minimum, so a top_k=20 request doesn't
    # balloon rerank cost).
    rerank_candidate_multiplier: int = 4
    rerank_candidate_ceiling: int = 50

    # Max hits from any one (page_url, section_id) in a /query response. Ranking
    # alone can return top_k slices of a single section (e.g. five different
    # sentences of the pricing block); this caps that so the agent sees breadth
    # across the tenant's content instead. Applied after reranking.
    max_hits_per_section: int = 2

    # HNSW query-time recall knob (app/db.py). pgvector's index default
    # (ef_search=40) is fixed regardless of how many candidates a query asks
    # for, and is a *global* scan budget before any per-tenant WHERE filter is
    # applied — once the chunks table holds several tenants, a small tenant's
    # rows can fall outside the top-40 nearest globally and searches return
    # too few (or zero) hits despite the tenant's data being present and
    # correct. Set above the largest candidate pool a single query can request
    # end to end (top_k=20 with reranking -> fetch_k=50 -> hybrid candidate_n
    # up to 400) so the tenant filter has real room to be satisfied even at
    # the extreme end of the request range; ef_search only costs query
    # latency, not storage.
    hnsw_ef_search: int = 400

    # --- Sales script generation (app/salescript/) ---
    # An explicit, user-triggered action (unlike doc2query's silent fail-open)
    # so disabling it should surface as an error, not a no-op.
    sales_script_enabled: bool = True
    sales_script_extract_model: str = "claude-haiku-4-5"  # extract_facts tier
    sales_script_model: str = "claude-sonnet-5"  # profile_site/draft/critique tier
    sales_script_max_revisions: int = 2
    sales_script_max_page_chars: int = 8000
    sales_script_page_batch_size: int = 1
    sales_script_max_concurrent_extractions: int = 4
    # Lowered from an earlier 6: the SDK's backoff is min(0.5 * 2**n, 8s), so
    # retries past ~4 buy only 8s each — a poor way to wait out a multi-minute
    # capacity event. Patience now lives in graph.py's node-level RetryPolicy
    # (_LLM_RETRY), whose intervals can be much longer and, unlike the SDK's
    # DEBUG-level retries, are logged at INFO and visible in LangSmith. This
    # multiplies with RetryPolicy.max_attempts (4 x 3 = 12 HTTP attempts/stage),
    # so don't raise both independently.
    sales_script_anthropic_max_retries: int = 3
    # Per-attempt HTTP timeout. The SDK default is 600s, which at even a modest
    # retry count is tens of minutes for one stage — past _STALE_GENERATION
    # (store.py, 30 min), after which a second worker can claim the same row.
    # Sized above a non-streaming 8192-token Sonnet generation (draft_script),
    # unlike questions.py's 60s, which would false-positive on a large draft.
    sales_script_timeout_seconds: float = 240.0
    # Hard ceiling on one whole generation run, enforced in
    # service.py::run_generation via asyncio.wait_for. Must stay comfortably
    # below store._STALE_GENERATION (30 min) so a slow worker can never
    # outlive its own claim and race a re-claimed second worker on the same row.
    sales_script_run_timeout_seconds: float = 1500.0
    # Floor for graph.py's fail-open fact extraction (extract_facts_one):
    # fraction of pages that must extract successfully before profile_site
    # will proceed. Below this the run fails loudly rather than writing a
    # confident script from the few facts that survived.
    sales_script_min_extract_success_ratio: float = 0.5

    # LangSmith tracing for the sales-script graph. Read as our own settings
    # (not left to pydantic-settings' env_file loading, which never mutates
    # os.environ) and forwarded below so the langsmith/langgraph libraries —
    # which read these via os.environ directly — see them regardless of
    # whether the process already had them exported.
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "firecrawl-sales-script"

    # Deployment environment. "production"/"staging" make app/startup_checks.py
    # strict: an unset secret becomes a refusal to start instead of a silent
    # default. Anything else (the default) only warns, so local dev and tests
    # keep working with no .env at all.
    environment: str = "development"

    # --- Observability ---
    log_level: str = "INFO"
    log_json: bool = True
    # Fail-open like anthropic_api_key/langchain_api_key above: an unset DSN
    # means Sentry is simply never initialized, not an error.
    sentry_dsn: str = ""
    sentry_environment: str = "development"
    sentry_traces_sample_rate: float = 0.0


settings = Settings()

if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
