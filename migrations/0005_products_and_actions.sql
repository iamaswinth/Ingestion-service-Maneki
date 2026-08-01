-- Structured commerce data for the voice shopping agent: products as
-- first-class entities (GET /products/{tenant_id} — catalog search for
-- "suggest alternatives" and grounding "what's in stock" in real data
-- instead of retrieval prose) and the clickable-action inventory per page
-- (GET /page-actions/{tenant_id} — what the agent may click, mirroring how
-- 0004_page_links.sql made click-based navigation possible).
--
-- Extracted by app/products.py (JSON-LD schema.org/Product, falling back to
-- Open Graph product tags) and app/actions.py (buttons/inputs/selects,
-- classified by label + a Shopify-specific /cart/add form-action signal),
-- fed the untouched rawHtml Firecrawl returns (app/scraper.py::to_pages) —
-- same rawHtml that resolved 0004's nav-links limitation.
--
-- product_key mirrors chunks.chunk_id's role: a stable hash of
-- (tenant_id, site_url, page_url, sku-or-name) computed at persistence time
-- (app/products.py::make_product_key, called from app/storage.py) — not by
-- the extractor itself, which never has tenant_id/site_url in scope (it runs
-- inside app/scraper.py::to_pages, before a tenant/site is attached).
--
-- No backfill for page_actions, same reasoning 0004 already established for
-- links/url_key: a pre-migration row's page_actions is NULL, which every
-- caller treats as "no actions captured" -> today's behavior. The next
-- crawl of a site repopulates it.

CREATE TABLE IF NOT EXISTS products (
  product_key    TEXT PRIMARY KEY,
  tenant_id      TEXT NOT NULL,
  job_id         TEXT NOT NULL,
  site_url       TEXT NOT NULL,
  page_url       TEXT NOT NULL,
  name           TEXT NOT NULL,
  sku            TEXT,
  price_amount   NUMERIC,
  price_currency TEXT,
  availability   TEXT,
  image_url      TEXT,
  description    TEXT,
  attributes     JSONB,
  source         TEXT NOT NULL,
  embedding      vector(384),
  created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS products_tenant_idx      ON products (tenant_id);
CREATE INDEX IF NOT EXISTS products_tenant_site_idx ON products (tenant_id, site_url);
CREATE INDEX IF NOT EXISTS products_tenant_page_idx ON products (tenant_id, page_url);
CREATE INDEX IF NOT EXISTS products_embedding_idx
  ON products USING hnsw (embedding vector_cosine_ops);

ALTER TABLE pages ADD COLUMN IF NOT EXISTS page_actions JSONB;
