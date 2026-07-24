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

## Schema migrations

The schema lives in `migrations/*.sql`, applied by `app/migrate.py`:

```powershell
python -m app.migrate            # apply everything pending
python -m app.migrate --status   # what's applied, what's pending
```

Deliberately not Alembic — there's no ORM here, so autogenerate has nothing to
introspect and every migration would be hand-written raw SQL regardless. See
`app/migrate.py`'s docstring for what the runner does guarantee (ordering,
exactly-once, per-migration atomicity, advisory-locked concurrent deploys,
immutable applied history).

**Rules:**
- Filenames are zero-padded (`0002_...`, not `2_...`) — ordering is a string sort.
- **Never edit a migration that has been applied.** The runner stores each file's
  checksum and refuses to run if one changed; add a new migration instead.
- There is no downgrade. Recover forward.
- New schema changes go in a new migration file — **not** as another
  `ALTER TABLE ... IF NOT EXISTS` appended to a `_SCHEMA_SQL` string. Several of
  those accumulated in `app/ingestion/store.py` before migrations existed and
  were re-executed on every process start; `0001_baseline.sql` absorbed them.

Three modules own a slice of the schema (`app/storage.py` → `jobs`/`pages`,
`app/ingestion/store.py` → `chunks`, `app/salescript/store.py` → `sales_scripts`).
Each still creates its own tables in dev/test, but in `production`/`staging` they
only *verify* (see `app/schema_guard.py`) and raise `SchemaNotMigrated` if the
migration step was skipped — so prod DB credentials don't need CREATE rights.

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
