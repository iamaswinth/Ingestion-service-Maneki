-- profile_site (app/salescript/graph.py) now classifies what kind of site
-- was crawled (SiteArchetype — saas_product, ecommerce, portfolio, ...)
-- before draft_script writes anything, so the script can be shaped by a
-- matching playbook (app/salescript/playbooks.py) instead of always
-- assuming a B2B software sale. The result is stored alongside the script
-- itself, not folded into it: `script` validates against SalesScript with
-- extra="forbid", so adding a field inside that JSONB would break every
-- pre-existing row's validation on next read.
--
-- NULL for rows generated before this column existed, or for a generation
-- that failed before profile_site completed — app/salescript/store.py and
-- app/salescript/service.py treat NULL the same as archetype="other".

ALTER TABLE sales_scripts ADD COLUMN IF NOT EXISTS site_profile JSONB;
