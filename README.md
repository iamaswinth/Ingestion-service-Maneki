# Firecrawl Scraping + Ingestion Pipeline

A "paste a website URL → voice-agent knowledge base" pipeline: a FastAPI
service that crawls a site to clean, section-aware markdown via a
**self-hosted Firecrawl** stack (zero cost, no API key), then chunks and
embeds it into **pgvector** for a voice agent to retrieve answers from and
navigate the site with.

```
POST /scrape {url, tenant_id} ──► FastAPI (:8000) ──► Firecrawl self-hosted (:3002)
                                        │                 API ─ Redis queue ─ worker(s) ─ Playwright
                                        ▼
                       Postgres: jobs + pages tables (job status, page markdown/sections)
                                        │  (auto-triggered once crawl completes)
                                        ▼
                    chunk (by HTML section) → embed (local, free) → upsert
                                        ▼
                              Postgres + pgvector `chunks` table
                                        │
POST /query {tenant_id, question} ─────┘──► cosine search, tenant-scoped
                                             → text to speak + navigation URL
```

Each HTML-id section (`id="pricing"`, `id="faq"`, ...) becomes its own chunk,
so a retrieved chunk carries both the answer text **and** a ready-to-use
navigation target (`https://site.com/#pricing`) — the voice agent speaks the
former and the frontend can act on the latter.

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

## 2. Start Postgres + pgvector

Prod uses a free [Neon](https://neon.tech) project (just paste its connection
string into `DATABASE_URL` — we run `CREATE EXTENSION vector` ourselves, no
manual setup needed; append `?ssl=require` to the connection string, since
Neon requires SSL). For local dev, run pgvector in Docker:

```bash
docker run -d --name pgvector -p 5433:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
```

## 3. Run this API

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # defaults already point at the local stack
uvicorn app.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`. The first embedding call
downloads the model (~130MB, one time).

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/scrape` | Start a crawl. Body: `{"url", "tenant_id", "limit"?, "include_paths"?, "exclude_paths"?, "wait_for"?}`. Returns `job_id`. |
| `GET` | `/scrape/{job_id}` | Live status. Persists pages once the crawl completes, then auto-triggers ingestion in the background (`ingest_status`: `not_started → ingesting → ingested\|ingest_failed`). |
| `GET` | `/scrape/{job_id}/pages` | Persisted page index. `?include_content=true` for full markdown + sections. |
| `POST` | `/ingest/{job_id}` | Manually (re-)run ingestion for a completed crawl. Idempotent. |
| `POST` | `/query` | Body: `{"tenant_id", "question", "top_k"?, "site_url"?, "page_url"?}`. Vector search scoped to the tenant. |
| `GET` | `/map?url=` | Preview discovered URLs before committing to a crawl. |
| `GET` | `/health` | Firecrawl + database reachability (diagnostic; Firecrawl check is cached ~45s). |
| `GET` | `/health/live` | DB-only liveness probe — point a load balancer's health check here, not at `/health`. |

### Example

```bash
# start a crawl for a tenant
curl -X POST http://localhost:8000/scrape \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://docs.firecrawl.dev","tenant_id":"acme","limit":5}'
# -> {"job_id":"abc-123","url":"...","tenant_id":"acme","status":"scraping"}

# poll until status=="completed" and ingest_status=="ingested"
curl http://localhost:8000/scrape/abc-123

# ask the voice agent's question
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"acme","question":"how much does it cost?"}'
# -> {"hits":[{"text":"...","score":0.87,"page_url":"...","section_id":"pricing",
#              "anchor_type":"id","navigation":"https://.../#pricing", ...}]}
```

Job status and page content live in Postgres (`jobs` and `pages` tables) —
retrieve them via `GET /scrape/{job_id}` (status) and
`GET /scrape/{job_id}/pages?include_content=true` (markdown + section tree,
one row per page).

## Chunking

Sites vary wildly in how much HTML structure survives to their `sections.json`
— some have clean semantic ids (`#pricing`, `#faq`), others (typically
client-side-rendered SPAs) offer nothing but a single `id="root"` wrapper
around the whole page. `app/ingestion/chunker.py` handles both:

1. If the page has real, reasonably-sized id'd sections, each becomes a chunk
   (long ones are split with overlap; nested sections like FAQ answers keep a
   `parent_section_id` link).
2. Otherwise it falls back to splitting the flat markdown by heading, and if
   there aren't enough headings, by paragraph.

Every chunk gets a `navigation` target regardless of path: a real `#section-id`
anchor when available, a browser text-fragment anchor (`#:~:text=...`)
when there's no id, or the bare page URL as a last resort. Markdown images and
link brackets are stripped before embedding/storage so voice output never
includes `![Jupiter](https://.../jupiter.png)`.

## Retrieval

`POST /query` runs three stages, each independently toggleable per-request
(`hybrid`/`rerank` on `QueryRequest`) or via `.env`:

1. **Vector search** — cosine similarity over `fastembed` embeddings (pgvector).
2. **Hybrid fusion** — Postgres full-text search fused with the vector leg via
   Reciprocal Rank Fusion (RRF), so lexical matches (exact terms) and semantic
   matches both contribute (`HYBRID_*` settings). Falls back to vector-only if
   the hybrid query errors.
3. **Reranking** — a local cross-encoder (`fastembed`, `RERANK_MODEL`, default
   `Xenova/ms-marco-MiniLM-L-6-v2` — no API key, same footprint as the
   embedding model) re-scores the fused candidates by actually reading the
   query against each one's text, then the final `top_k` is sliced off *after*
   reranking. Set `RERANK_ENABLED=false` to skip this stage.

## Doc2query (AI-generated questions)

At ingest time, Claude Haiku (`claude-haiku-4-5`) generates ~3 questions each
chunk answers ("how much does it cost?"). Each question is embedded as its own
vector pointing back to the parent chunk (`kind='question'`,
`parent_chunk_id`), because incoming voice questions match stored *questions*
far better than raw passage text. `/query` dedupes so each content chunk
appears at most once; `matched_question` on a hit shows which question won.

Set `ANTHROPIC_API_KEY` in `.env` to enable. **Fail-open:** if the key is
missing or generation fails, ingest still completes with content chunks only
(pure dense retrieval, exactly the pre-doc2query behavior).

## Multi-tenancy

`tenant_id` is required on `/scrape` and every chunk row carries it. `/query`
always filters `WHERE tenant_id = ...` — this is the entire isolation
boundary, so any new query path must preserve it. Re-ingesting a job deletes
its previous chunks first (`DELETE ... WHERE tenant_id=... AND job_id=...`),
so re-running ingestion is safe and idempotent.

## Deploying

```bash
docker build -t ingestion .
docker run -p 8000:8000 --env-file .env ingestion
```

The image bakes the fastembed model in at build time (see `app/ingestion/embedder.py`), so
a fresh container never pays a model download on its first `/query`. `WEB_CONCURRENCY`
controls the number of `uvicorn` worker processes in prod (default `2`) — safe to raise
per-replica; see "Stateless API" below for why concurrent workers/replicas don't race.

CI (`.github/workflows/ci.yml`) runs `pytest` on every push/PR to `main` — no live
Postgres/Firecrawl needed, since the test client never triggers app startup (see
`tests/test_auth.py`).

## Observability

Logs are JSON on stdout by default (`LOG_JSON=true`) — one object per line, structured for
whatever log aggregator the eventual host provides, matching the 12-factor "log to stdout"
pattern the `Dockerfile` already assumes. Error tracking via Sentry is opt-in
(`SENTRY_DSN`) and a no-op until set. When a tenant reports "the voice agent doesn't know
anything about my site," the first place to look is the per-job ingest summary logged by
`app/ingestion/service.py::ingest_job` — it logs a `WARNING` on both zero-chunk paths (no
persisted pages, or the chunker produced nothing) and an `INFO` "ingest complete" line with
page/chunk counts on success, all tagged with `job_id`/`tenant_id`/`site_url`.

## Design notes

- **Stateless API.** Job/page state lives in Postgres (`jobs`/`pages` tables),
  not in memory or on local disk, so multiple uvicorn workers or container
  replicas share state safely with no shared volume. The scraping->persisted
  transition is claimed with an atomic `UPDATE ... WHERE persisted = false`,
  so it's race-safe across replicas too.
- **`job_id` = Firecrawl's crawl id.** No separate mapping to maintain.
- **Cloud/self-hosted Firecrawl switch** is one env var (`FIRECRAWL_API_URL`).
- **Embeddings are local and free** (`fastembed`, `BAAI/bge-small-en-v1.5`,
  CPU/ONNX) — no API key, no per-query cost.
- **Upgrade path** when you outgrow one machine: page markdown lives in a
  Postgres `TEXT` column, fine at current scale — revisit only if page
  size/volume ever justifies moving it to object storage (S3/GCS). Isolated
  to `app/storage.py`.

## Out of scope (next phase)

The voice agent itself (LLM + STT/TTS) that calls `/query` and drives
navigation from the `navigation` field.
