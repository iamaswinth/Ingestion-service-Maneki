# Firecrawl Scraping + Ingestion Pipeline

Scrape a pasted website URL to clean markdown (self-hosted Firecrawl), chunk it
by HTML section, embed locally, and store in pgvector for a voice agent to
query and use for on-page navigation.

## Model routing reminder

- **Planning** (design discussions, reviewing architecture, writing a plan
  before touching code): use **Fable**.
- **Writing/editing code**: switch to **Sonnet** or **Opus** first.

This isn't automatic — switch models yourself with `/model` before starting
each phase.

## Dev server rule

**Never leave background servers running after testing.** Always stop any
`uvicorn`/Docker process you started for a test before ending a task. The user
runs the dev server themselves with:

```
uvicorn app.main:app --reload --port 8000
```

If you start it to verify something, kill it again before you finish. Check
with:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
```

## Stack

- FastAPI app in `app/` (Python, venv at `.venv/`)
- Self-hosted Firecrawl stack (Docker Compose) in sibling folder `../firecrawl-selfhost`, port 3002
- Postgres + pgvector for job/page/chunk storage (Neon in prod, local Docker for dev/test)
- Local embeddings via `fastembed` (`BAAI/bge-small-en-v1.5`, 384 dims) — no API key, no cost

## Where things land

- Scraped output: `jobs` + `pages` tables in Postgres (job status, page markdown + section tree)
- Ingested chunks: `chunks` table in Postgres, filtered by `tenant_id`

## Running things

```powershell
# Firecrawl stack
docker compose -f ..\firecrawl-selfhost\docker-compose.yaml up -d

# This API
.\.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

Docs at `http://localhost:8000/docs`.
