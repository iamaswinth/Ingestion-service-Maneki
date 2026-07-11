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

    # Automatically ingest a crawl into the vector DB once it finishes scraping.
    auto_ingest: bool = True

    # Chunking knobs (app/ingestion/chunker.py).
    chunk_target_chars: int = 1000
    chunk_max_chars: int = 1500
    chunk_overlap_chars: int = 150
    chunk_min_chars: int = 40

    # --- Doc2query: AI-generated questions per chunk (app/ingestion/questions.py) ---
    # Anthropic API key. Empty => question generation is silently skipped and
    # the pipeline behaves exactly as before (pure dense retrieval).
    anthropic_api_key: str = ""
    question_gen_enabled: bool = True
    question_gen_model: str = "claude-haiku-4-5"
    questions_per_chunk: int = 3
    # How many chunks to pack into one LLM request.
    question_gen_batch_size: int = 12

    # --- Hybrid retrieval: dense vector + Postgres full-text, fused via RRF ---
    hybrid_search_enabled: bool = True
    hybrid_rrf_k: int = 60
    hybrid_vector_weight: float = 1.0
    hybrid_lexical_weight: float = 1.0
    # Mirrors the vector leg's existing top_k*8 oversample factor.
    hybrid_candidate_multiplier: int = 8
    hybrid_candidate_floor: int = 50

    # --- Sales script generation (app/salescript/) ---
    # An explicit, user-triggered action (unlike doc2query's silent fail-open)
    # so disabling it should surface as an error, not a no-op.
    sales_script_enabled: bool = True
    sales_script_extract_model: str = "claude-haiku-4-5"  # extract_facts tier
    sales_script_model: str = "claude-sonnet-5"  # derive_icp/draft/critique tier
    sales_script_max_revisions: int = 2
    sales_script_max_page_chars: int = 8000
    sales_script_page_batch_size: int = 1
    sales_script_max_concurrent_extractions: int = 4

    # LangSmith tracing for the sales-script graph. Read as our own settings
    # (not left to pydantic-settings' env_file loading, which never mutates
    # os.environ) and forwarded below so the langsmith/langgraph libraries —
    # which read these via os.environ directly — see them regardless of
    # whether the process already had them exported.
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "firecrawl-sales-script"


settings = Settings()

if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
