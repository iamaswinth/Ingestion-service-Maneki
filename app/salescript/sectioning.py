"""Turns a crawl's persisted pages into the page digests fed to the graph's
fact-extraction fan-out (app/salescript/graph.py::extract_facts_one).

Reads full page markdown rather than pre-segmented chunks: a price and its
footnote can land in different chunks after ingestion's ~1000-char splitting,
but fact extraction wants that context intact. `clean_text` (already used by
app/ingestion/chunker.py) strips the same image/link markdown noise here.
"""

from ..ingestion.chunker import clean_text
from ..models import JobPages


def build_page_digests(job_pages: JobPages, max_chars: int) -> list[dict]:
    digests = []
    for page in job_pages.pages:
        content = clean_text(page.markdown or "")[:max_chars]
        if not content:
            continue
        digests.append(
            {
                "url": page.url,
                "title": page.title,
                "content": content,
            }
        )
    return digests
