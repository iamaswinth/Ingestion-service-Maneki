-- Click-based navigation (single-hop): the voice agent asks the widget to
-- click a real <a> on the visitor's current page instead of assigning
-- location.href, so an SPA-routed target site intercepts it client-side and
-- the LiveKit connection survives. That decision needs to know which pages
-- are directly linked from the page the visitor is on, which nothing in
-- this schema recorded before now.
--
-- url_key is page_key(url) (app/links.py) — origin + path, default port and
-- trailing slash normalized, query and fragment dropped. Written by
-- app/storage.py::persist_pages, never computed in SQL: the exact same
-- normalization has to be reproducible in voice_runtime's Python and the
-- widget's TypeScript, and the only way to guarantee that is one spec with
-- hand-written implementations in each, not a Postgres expression nobody
-- else can mirror.
--
-- No backfill. url_key cannot be derived faithfully in SQL, and a pre-
-- migration row's links/url_key are NULL, which every caller already
-- treats as "no links available" -> the agent falls back to today's full-
-- reload navigation. The next crawl of a site repopulates both columns.

ALTER TABLE pages ADD COLUMN IF NOT EXISTS links   JSONB;
ALTER TABLE pages ADD COLUMN IF NOT EXISTS url_key TEXT;

CREATE INDEX IF NOT EXISTS pages_url_key_idx ON pages (url_key);
