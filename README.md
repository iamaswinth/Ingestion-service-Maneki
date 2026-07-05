# Firecrawl Scraping Pipeline

Phase 1 of a "paste a website URL → voice-agent knowledge base" product.

A FastAPI service: submit a URL, it crawls the site to clean markdown via a
**self-hosted Firecrawl** stack (zero cost, no API key), and persists each page
in a normalized format that the later ingestion phase (chunk → embed → vector
store) will consume.

```
POST /scrape ──► FastAPI (:8000) ──► Firecrawl self-hosted (:3002)
                     │                  API ─ Redis queue ─ worker(s) ─ Playwright
                     ▼
        data/{job_id}/pages/*.md   ◄── handoff to ingestion
```

## Prerequisites

- Docker Desktop (running)
- Python 3.11+

## 1. Start self-hosted Firecrawl

The Firecrawl repo is cloned as a sibling folder `../firecrawl-selfhost` with a
minimal `.env` (`USE_DB_AUTHENTICATION=false`, `PORT=3002`). From that folder:

```bash
docker compose build
docker compose up -d
```

Firecrawl is then at `http://localhost:3002`; the queue admin UI is at
`http://localhost:3002/admin/CHANGEME/queues`. Smoke test:

```bash
curl -X POST http://localhost:3002/v2/scrape \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://firecrawl.dev","formats":["markdown"]}'
```

**Scaling concurrency** (when you outgrow one worker): raise
`NUM_WORKERS_PER_QUEUE` / `MAX_CONCURRENT_JOBS` in `.env`, or run more API
containers with `docker compose up -d --scale api=N`.

## 2. Run this API

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # defaults already point at the local stack
uvicorn app.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/scrape` | Start a crawl. Body: `{"url", "limit"?, "include_paths"?, "exclude_paths"?}`. Returns `job_id`. |
| `GET` | `/scrape/{job_id}` | Live status; persists pages to disk once the crawl completes. |
| `GET` | `/scrape/{job_id}/pages` | Persisted page index. `?include_content=true` for full markdown. |
| `GET` | `/map?url=` | Preview discovered URLs before committing to a crawl. |
| `GET` | `/health` | Firecrawl reachability. |

### Example

```bash
# start
curl -X POST http://localhost:8000/scrape \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://docs.firecrawl.dev","limit":5}'
# -> {"job_id":"abc-123","url":"...","status":"scraping"}

# poll until status == "completed"
curl http://localhost:8000/scrape/abc-123

# read persisted pages
curl http://localhost:8000/scrape/abc-123/pages
```

Output lands in `data/{job_id}/`: `job.json` (state), `index.json` (page
metadata), `pages/NNN-slug.md` (clean markdown, one file per page).

## Design notes

- **Stateless API.** Job state lives on disk (`data/{job_id}/job.json`), not in
  memory, so multiple uvicorn workers on one box share state safely.
- **`job_id` = Firecrawl's crawl id.** No separate mapping to maintain.
- **Cloud/self-hosted switch** is one env var (`FIRECRAWL_API_URL`).
- **Upgrade path** when you outgrow one machine: job store → Postgres, pages →
  object storage (S3/GCS). Both are isolated to `app/storage.py`.

## Out of scope (next phase)

Ingestion: chunking, embeddings (free/local via sentence-transformers or
Ollama), vector store, and a query endpoint for the voice agent.
