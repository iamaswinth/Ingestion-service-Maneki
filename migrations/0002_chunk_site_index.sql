-- Every ingest's delete+replace (app/ingestion/store.py::replace_site_chunks,
-- replace_site_script_chunks, delete_site_chunks) filters chunks by exactly
-- (tenant_id, site_url). Until now the only composite index on this table
-- was (tenant_id, page_url), so those deletes ran against the tenant-only
-- index and rechecked every one of that tenant's rows for a site_url match.

CREATE INDEX IF NOT EXISTS chunks_tenant_site_idx ON chunks (tenant_id, site_url);
