"""Extract id-anchored sections from a page's HTML.

Firecrawl's markdown is a flat text projection that drops all HTML ids/classes.
To tag content with the section it came from (e.g. id="pricing"), we parse the
HTML that Firecrawl returns alongside the markdown and rebuild a section tree.

A "section" is a structural element (section/div/article/...) that carries an
`id` and holds a meaningful amount of text. Sections nest: a top-level section
(pricing, faq, ...) may contain nested id'd children (faq-q-01, faq-a-01), which
become `children`. Each node's markdown is its *own* content — nested children
are stripped out before conversion — so parent and child never duplicate text.
"""

import copy

from bs4 import BeautifulSoup
from markdownify import markdownify as _to_markdown

from .models import Section

# Only these tags are treated as section wrappers; ignore id'd <span>/<img> etc.
_STRUCTURAL_TAGS = {
    "section",
    "div",
    "article",
    "main",
    "header",
    "footer",
    "aside",
    "nav",
}

# Skip id'd elements with less than this much text (icons, tracking attrs, ...).
_MIN_TEXT_CHARS = 20


def _qualifies(el) -> bool:
    if not el.get("id") or el.name not in _STRUCTURAL_TAGS:
        return False
    return len(el.get_text(strip=True)) >= _MIN_TEXT_CHARS


def _to_md(node) -> str:
    return _to_markdown(str(node), heading_style="ATX", strip=["script", "style"]).strip()


def _humanize(section_id: str) -> str:
    return section_id.replace("-", " ").replace("_", " ").strip().title()


def _title(own_node, section_id: str) -> str:
    for heading in own_node.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = heading.get_text(strip=True)
        if text:
            return text
    return _humanize(section_id)


def extract_sections(html: str) -> list[Section]:
    if not html:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []

    # Component-generated markup (Framer, and others) routinely stamps the
    # same `id` on multiple elements — breakpoint or animation-state variants
    # of one logical block, one often nested inside the other. Without this
    # dedup, each duplicate becomes its own Section nested under the last,
    # and since one is a near-clone of the other, decompose() only partially
    # separates them: a page can end up with 100+ chained "sections" sharing
    # one section_id, each markdown just a few characters off from the last.
    # Keep only the richest (most text) element per distinct id string.
    by_id: dict[str, object] = {}
    for el in soup.find_all(id=True):
        if not _qualifies(el):
            continue
        section_id = el.get("id")
        existing = by_id.get(section_id)
        if existing is None or len(el.get_text(strip=True)) > len(
            existing.get_text(strip=True)
        ):
            by_id[section_id] = el
    qualifying = list(by_id.values())
    qset = {id(el) for el in qualifying}

    def nearest_qualifying_ancestor(el):
        for parent in el.parents:
            if id(parent) in qset:
                return parent
        return None

    # Bucket each qualifying element under its nearest qualifying ancestor.
    children_of: dict[int, list] = {id(el): [] for el in qualifying}
    tops = []
    for el in qualifying:
        parent = nearest_qualifying_ancestor(el)
        if parent is None:
            tops.append(el)
        else:
            children_of[id(parent)].append(el)

    def build(el) -> Section:
        directs = children_of[id(el)]
        # Own content = this element minus its nested qualifying children.
        clone = copy.copy(el)
        for child in directs:
            child_id = child.get("id")
            match = clone.find(id=child_id)
            if match is not None:
                match.decompose()
        own_md = _to_md(clone)
        return Section(
            id=el.get("id"),
            tag=el.name,
            title=_title(clone, el.get("id")),
            chars=len(own_md),
            markdown=own_md,
            children=[build(child) for child in directs],
        )

    return [build(top) for top in tops]
