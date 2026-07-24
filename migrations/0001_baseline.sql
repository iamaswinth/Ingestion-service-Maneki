-- Baseline: the schema as it existed before migrations were introduced,
-- consolidated from the three modules that each owned a piece of it —
-- app/storage.py (jobs, pages), app/ingestion/store.py (chunks),
-- app/salescript/store.py (sales_scripts).
--
-- Written with IF NOT EXISTS throughout so it is safe to apply to a database
-- that already has these objects (every dev machine, at the time this was
-- added). Later migrations need no such caution — they run exactly once, in
-- order, and should say plainly what they change.
--
-- The ALTER/UPDATE statements near the bottom were previously re-run on every
-- process start as inline "idempotent migrations". They are preserved here
-- verbatim so an existing database reaches the same state; new databases get
-- them as part of this one baseline.

-- ---------------------------------------------------------------- jobs/pages

CREATE TABLE IF NOT EXISTS jobs (
  job_id             TEXT PRIMARY KEY,
  url                TEXT NOT NULL,
  tenant_id          TEXT NOT NULL,
  status             TEXT NOT NULL,
  completed_pages    INT NOT NULL DEFAULT 0,
  total_pages        INT NOT NULL DEFAULT 0,
  error              TEXT,
  persisted          BOOLEAN NOT NULL DEFAULT false,
  ingest_status      TEXT NOT NULL DEFAULT 'not_started',
  ingested_chunks    INT NOT NULL DEFAULT 0,
  ingested_questions INT NOT NULL DEFAULT 0,
  ingest_error       TEXT,
  created_at         timestamptz NOT NULL DEFAULT now(),
  updated_at         timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS jobs_tenant_idx ON jobs (tenant_id);
CREATE INDEX IF NOT EXISTS jobs_ingest_status_idx ON jobs (ingest_status);
CREATE INDEX IF NOT EXISTS jobs_open_idx ON jobs (status) WHERE persisted = false;

CREATE TABLE IF NOT EXISTS pages (
  job_id         TEXT NOT NULL REFERENCES jobs (job_id) ON DELETE CASCADE,
  page_index     INT NOT NULL,
  url            TEXT NOT NULL,
  title          TEXT,
  description    TEXT,
  markdown       TEXT NOT NULL,
  chars          INT NOT NULL,
  section_count  INT NOT NULL DEFAULT 0,
  sections       JSONB,
  created_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (job_id, page_index)
);

CREATE INDEX IF NOT EXISTS pages_job_idx ON pages (job_id);

-- -------------------------------------------------------------------- chunks

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id          TEXT PRIMARY KEY,
  tenant_id         TEXT NOT NULL,
  job_id            TEXT NOT NULL,
  site_url          TEXT NOT NULL,
  page_url          TEXT NOT NULL,
  section_id        TEXT,
  parent_section_id TEXT,
  anchor_type       TEXT NOT NULL,
  navigation        TEXT NOT NULL,
  title             TEXT,
  content_type      TEXT,
  chunk_index       INT NOT NULL,
  text              TEXT NOT NULL,
  embedding         vector(384) NOT NULL,
  created_at        timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks (tenant_id);
CREATE INDEX IF NOT EXISTS chunks_tenant_page_idx ON chunks (tenant_id, page_url);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
  ON chunks USING hnsw (embedding vector_cosine_ops);

-- Doc2query: 'question' rows are synthetic visitor questions embedded as
-- extra vectors pointing back at their content chunk via parent_chunk_id.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'content';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS parent_chunk_id TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS question TEXT;

-- Hybrid retrieval: embedding_text was previously computed at ingest time but
-- never persisted; storing it lets full-text search run over the same text the
-- dense embedding was built from (title-prefixed content, or the raw synthetic
-- question for kind='question' rows).
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_text TEXT;

-- Two-arg to_tsvector with a literal config name is the documented pattern for
-- a generated column (the config is resolved at DDL time). COALESCE to `text`
-- so lexical search still works for any row before backfill runs.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector tsvector
  GENERATED ALWAYS AS (to_tsvector('english', coalesce(embedding_text, text))) STORED;

CREATE INDEX IF NOT EXISTS chunks_search_vector_idx
  ON chunks USING GIN (search_vector);

-- Backfill rows ingested before embedding_text existed. Exact for
-- kind='question' (question text is already stored verbatim); an approximation
-- for kind='content' since the page-level title isn't persisted per-chunk
-- (only the section title is). WHERE-guarded so this is a no-op scan once
-- every row has been backfilled.
UPDATE chunks
SET embedding_text = CASE
    WHEN kind = 'question' THEN question
    ELSE coalesce(title || ': ' || text, text)
END
WHERE embedding_text IS NULL;

-- ------------------------------------------------------------- sales_scripts

CREATE TABLE IF NOT EXISTS sales_scripts (
  id             BIGSERIAL PRIMARY KEY,
  tenant_id      TEXT NOT NULL,
  site_url       TEXT NOT NULL,
  job_id         TEXT NOT NULL,
  status         TEXT NOT NULL DEFAULT 'not_started',
  script         JSONB,
  critique       JSONB,
  revision_count INT NOT NULL DEFAULT 0,
  error          TEXT,
  created_at     timestamptz NOT NULL DEFAULT now(),
  updated_at     timestamptz NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, site_url)
);

CREATE INDEX IF NOT EXISTS sales_scripts_tenant_idx ON sales_scripts (tenant_id);
CREATE INDEX IF NOT EXISTS sales_scripts_status_idx ON sales_scripts (status);
