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


settings = Settings()
