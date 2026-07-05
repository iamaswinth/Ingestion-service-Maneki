from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Base URL of the Firecrawl instance. Defaults to the local self-hosted
    # stack; set to https://api.firecrawl.dev to use the cloud service.
    firecrawl_api_url: str = "http://localhost:3002"

    # Optional for self-hosted (auth is bypassed). Required only for cloud.
    # The SDK still wants a non-empty string, so we default to a placeholder.
    firecrawl_api_key: str = "self-hosted"

    # Where scraped pages are persisted (relative to project root).
    data_dir: Path = Path("data")

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


settings = Settings()
