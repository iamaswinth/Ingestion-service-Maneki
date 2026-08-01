"""Extract clickable controls from a page's raw HTML — the action inventory
a voice shopping agent may act on (GET /page-actions/{tenant_id}).

Mirrors app/links.py's contract exactly: the agent may only ever click a
control ingestion actually verified exists on the page, never invent a
selector itself. What this module captures is a *bundle* of priority-ordered
selector strings per control, not one brittle string — real sites hash their
class names (Tailwind/CSS-in-JS), so the widget (a later phase) tries these
in order and verifies the resolved element's accessible name matches before
acting.

`product_key` on a captured PageAction is deliberately left unset here (see
its docstring in app/models.py) — associating an action with a specific
product requires knowing whether the *page* has exactly one product, which
this module doesn't have visibility into (extract_actions only sees one
page's HTML, not the Product list app/products.py separately extracted from
the same page). That association is resolved in app/storage.py at
persistence time, alongside product_key resolution itself.

Known v1 limitation: a multi-product listing/category page would need
DOM-proximity matching between an action and its nearest product card to
associate them correctly — real additional complexity for uncertain payoff.
For now, actions on a page with zero or multiple products simply ship with
product_key=None rather than guessing wrong. Only the common single-product
product-detail-page case gets an association (in app/storage.py).

`app/scraper.py::to_pages` feeds this the untouched `rawHtml`, same as
app/links.py::extract_links and app/products.py::extract_products.
"""

import re
from typing import Optional

from bs4 import BeautifulSoup

from .models import ActionKind, PageAction

# Generous cap mirroring app/links.py's MAX_LINKS_PER_PAGE — one pathological
# page (a filter sidebar with fifty checkboxes, say) can't blow up the payload.
MAX_ACTIONS_PER_PAGE = 100
MAX_LABEL_CHARS = 120

# Ordered — first match wins. Deliberately simple regexes on the accessible
# name, not an LLM: cheap, testable, and covers the overwhelming majority of
# real storefront copy ("Add to Cart", "Add to Bag", "Buy It Now", "Proceed
# to Checkout", ...).
_KIND_PATTERNS: list[tuple[ActionKind, re.Pattern]] = [
    ("add_to_cart", re.compile(r"add\s*to\s*(cart|bag)", re.IGNORECASE)),
    ("buy_now", re.compile(r"\bbuy\s*(it\s*)?now\b", re.IGNORECASE)),
    ("checkout", re.compile(r"\b(proceed\s*to\s*)?check\s*out\b", re.IGNORECASE)),
    ("view_cart", re.compile(r"\b(view|go\s*to)\s*cart\b", re.IGNORECASE)),
    ("search", re.compile(r"\bsearch\b", re.IGNORECASE)),
    ("filter", re.compile(r"\bfilter\b", re.IGNORECASE)),
]

# Shopify's own add-to-cart forms POST here by convention — a very
# high-signal, deterministic tell independent of whatever the button's own
# label says (some themes render it as just an icon with no visible text).
_CART_ADD_ACTION_RE = re.compile(r"/cart/add\b", re.IGNORECASE)

_DATA_ID_ATTRS = ("data-testid", "data-test", "data-cy", "data-qa")


def _accessible_name(el) -> str:
    """Visible label for a control — the semantics differ enough by element
    type that one text-content-then-aria-label chain (app/links.py's
    _anchor_text) doesn't fit all of them:
    - <input type="submit"|"button">: `value` IS the rendered label.
    - <input type="number"|"text"|...> (e.g. a quantity field): `value` is
      the control's *current content*, not a label — a quantity box holding
      "1" must not be mislabeled "1"; aria-label/title is what actually
      describes it.
    - <select>: its own accessible name (aria-label/title) takes priority
      over its concatenated option text, which describes the available
      choices, not the control itself. Falls back to option text only if
      nothing else is available, rather than reporting no label at all.
    - everything else (button, a, ...): text content, then aria-label/title.
    """
    if el.name == "input" and el.get("type") in ("submit", "button"):
        value = el.get("value")
        if value and value.strip():
            return " ".join(value.split())
    elif el.name != "select":
        text = " ".join(el.get_text(" ", strip=True).split())
        if text:
            return text
    for attr in ("aria-label", "title"):
        value = el.get(attr)
        if value and value.strip():
            return " ".join(value.split())
    if el.name == "select":
        text = " ".join(el.get_text(" ", strip=True).split())
        if text:
            return text
    return ""


def _role_of(el) -> str:
    declared = el.get("role")
    if declared in ("button", "link", "input", "select"):
        return declared
    return {"a": "link", "input": "input", "select": "select"}.get(el.name, "button")


def _classify_kind(el, label: str) -> ActionKind:
    if el.name == "select":
        return "select_variant"
    if el.name == "input" and el.get("type") == "number":
        return "quantity"
    for kind, pattern in _KIND_PATTERNS:
        if pattern.search(label):
            return kind
    return "other"


def _selector_bundle(el) -> list[str]:
    """Priority-ordered, most-specific first. #id and a real test-id
    attribute survive a redesign that reshuffles class names; aria-label is
    the last structural fallback before the widget falls back further still
    to role+accessible-name search (see app/models.py::PageAction)."""
    selectors: list[str] = []
    el_id = el.get("id")
    if el_id:
        selectors.append(f"#{el_id}")
    for attr in _DATA_ID_ATTRS:
        value = el.get(attr)
        if value:
            selectors.append(f'[{attr}="{value}"]')
    aria_label = el.get("aria-label")
    if aria_label and aria_label.strip():
        selectors.append(f'[aria-label="{aria_label.strip()}"]')
    return selectors


def _nearest_section_id(el) -> Optional[str]:
    for parent in el.parents:
        parent_id = parent.get("id") if hasattr(parent, "get") else None
        if parent_id:
            return parent_id
    return None


def _build_action(el, kind: ActionKind, label: str) -> Optional[PageAction]:
    label = label[:MAX_LABEL_CHARS]
    selectors = _selector_bundle(el)
    if not label and not selectors:
        # No accessible name AND no stable selector — nothing a visitor
        # could ask for by description, and nothing the widget could later
        # resolve reliably either. Not worth capturing.
        return None
    return PageAction(
        kind=kind,
        label=label,
        role=_role_of(el),
        selectors=selectors,
        section_id=_nearest_section_id(el),
    )


def extract_actions(html: str, page_url: str) -> list[PageAction]:
    """Every clickable control on this page worth the agent knowing about.
    Never raises; malformed or absent HTML -> []."""
    if not html:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []

    actions: list[PageAction] = []
    seen_elements: set[int] = set()

    # Pass 1: Shopify-style add-to-cart forms — a form's action attribute is
    # a stronger, deterministic signal than any button's own label text.
    for form in soup.find_all("form", action=True):
        if not _CART_ADD_ACTION_RE.search(form["action"]):
            continue
        submit = next(
            (b for b in form.find_all("button") if b.get("type") != "reset"), None
        )
        if submit is None:
            submit = form.find("input", attrs={"type": ["submit", "button"]})
        if submit is None:
            continue
        label = _accessible_name(submit)
        action = _build_action(submit, "add_to_cart", label)
        if action is not None:
            actions.append(action)
            seen_elements.add(id(submit))
            if len(actions) >= MAX_ACTIONS_PER_PAGE:
                return actions

    # Pass 2: everything else — buttons, submit/button inputs, selects,
    # number inputs, and anchors explicitly marked role="button". A reset
    # button clears a form; it's never something a shopping agent should
    # click, deterministic form-detection nonsense aside.
    candidates = soup.find_all("button") + soup.find_all(
        "input", attrs={"type": ["submit", "button", "number"]}
    ) + soup.find_all("select") + soup.find_all("a", attrs={"role": "button"})

    for el in candidates:
        if id(el) in seen_elements:
            continue
        if el.get("type") == "reset":
            continue
        label = _accessible_name(el)
        kind = _classify_kind(el, label)
        action = _build_action(el, kind, label)
        if action is None:
            continue
        actions.append(action)
        seen_elements.add(id(el))
        if len(actions) >= MAX_ACTIONS_PER_PAGE:
            break

    return actions
