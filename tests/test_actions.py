"""Tests for app/actions.py (extract_actions) and the storage-layer lookup
that backs GET /page-actions/{tenant_id} (app/storage.py::load_page_actions).

Pure-function tests need no I/O; the load_page_actions tests use the
pg_tenant fixture (tests/conftest.py), which skips cleanly when Postgres
isn't reachable. Mirrors tests/test_page_links.py's structure exactly.
"""

import uuid

from app import storage
from app.actions import extract_actions
from app.models import Page, PageAction

# ---- extract_actions: kind classification -----------------------------------


def test_classifies_add_to_cart_from_label():
    html = '<button>Add to Cart</button>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].kind == "add_to_cart"


def test_classifies_add_to_bag_variant():
    html = '<button>Add to Bag</button>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].kind == "add_to_cart"


def test_classifies_buy_now():
    # Empty text content so aria-label is the actual accessible name used —
    # a button with real visible text uses that first (matches
    # app/links.py::_anchor_text's established text-before-aria-label
    # convention), so this isn't testing a conflicting-signal case.
    html = '<button aria-label="Buy It Now"></button>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].kind == "buy_now"


def test_classifies_checkout():
    html = '<button>Proceed to Checkout</button>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].kind == "checkout"


def test_classifies_view_cart():
    html = '<button>View Cart</button>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].kind == "view_cart"


def test_classifies_search():
    html = '<button aria-label="Search"></button>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].kind == "search"


def test_classifies_filter():
    html = '<button>Filter results</button>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].kind == "filter"


def test_unrecognized_label_classifies_as_other():
    html = '<button>Learn More</button>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].kind == "other"


def test_select_element_classifies_as_select_variant():
    html = '<select aria-label="Size"><option>S</option><option>M</option></select>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].kind == "select_variant"
    # aria-label wins over concatenated option text as the label.
    assert actions[0].label == "Size"


def test_select_without_aria_label_falls_back_to_option_text():
    html = '<select id="color"><option>Red</option><option>Blue</option></select>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].label == "Red Blue"


def test_number_input_classifies_as_quantity_not_labeled_by_its_value():
    # A quantity box holding "1" must not be mislabeled "1" — that's its
    # current content, not a description of the control.
    html = '<input type="number" value="1" aria-label="Quantity">'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].kind == "quantity"
    assert actions[0].label == "Quantity"


def test_submit_input_uses_value_as_its_label():
    html = '<input type="submit" value="Add to Cart">'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].kind == "add_to_cart"
    assert actions[0].label == "Add to Cart"


def test_anchor_with_button_role_is_captured():
    html = '<a role="button" href="/checkout">Checkout</a>'
    actions = extract_actions(html, "https://x.com/p")
    assert len(actions) == 1
    assert actions[0].role == "button"
    assert actions[0].kind == "checkout"


def test_plain_anchor_without_button_role_is_not_captured():
    html = '<a href="/about">About</a>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions == []


# ---- extract_actions: reset buttons excluded --------------------------------


def test_reset_button_is_excluded():
    html = '<button type="reset">Reset</button>'
    assert extract_actions(html, "https://x.com/p") == []


def test_reset_input_is_excluded():
    html = '<input type="reset" value="Clear">'
    assert extract_actions(html, "https://x.com/p") == []


# ---- extract_actions: Shopify /cart/add form detection ----------------------


def test_shopify_cart_add_form_detected_regardless_of_button_label():
    # Real Shopify themes sometimes render this as an icon with no visible
    # text — the form action is the deterministic signal, not the label.
    html = '<form action="/cart/add" method="post"><button aria-label="icon-only"></button></form>'
    actions = extract_actions(html, "https://x.com/p")
    assert len(actions) == 1
    assert actions[0].kind == "add_to_cart"


def test_cart_add_form_is_not_double_counted_in_the_generic_pass():
    html = '<form action="/cart/add"><button id="add-btn">Add to Cart</button></form>'
    actions = extract_actions(html, "https://x.com/p")
    assert len(actions) == 1


def test_form_with_unrelated_action_is_not_treated_as_add_to_cart():
    html = '<form action="/search"><button>Go</button></form>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].kind == "other"


# ---- extract_actions: selector bundle ---------------------------------------


def test_selector_bundle_priority_order():
    html = '<button id="add-btn" data-testid="add-to-cart-btn" aria-label="Add">Add to Cart</button>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].selectors == [
        "#add-btn",
        '[data-testid="add-to-cart-btn"]',
        '[aria-label="Add"]',
    ]


def test_no_selector_but_has_label_is_still_captured():
    html = '<button>Random Button</button>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].selectors == []
    assert actions[0].label == "Random Button"


# ---- extract_actions: section_id --------------------------------------------


def test_section_id_from_nearest_ancestor():
    html = '<section id="product-main"><button>Add to Cart</button></section>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].section_id == "product-main"


def test_section_id_is_none_with_no_ancestor_id():
    html = '<div><button>Add to Cart</button></div>'
    actions = extract_actions(html, "https://x.com/p")
    assert actions[0].section_id is None


# ---- extract_actions: caps, dedupe-free malformed/empty input ---------------


def test_extract_actions_never_raises_on_unclosed_tags():
    # bs4's html.parser is lenient by design; this asserts the *contract*
    # (a list back, no exception), not any particular recovered structure.
    result = extract_actions("<div><button>Add to Cart", "https://x.com/p")
    assert isinstance(result, list)


def test_extract_actions_empty_or_none_html_returns_empty():
    assert extract_actions("", "https://x.com/p") == []
    assert extract_actions(None, "https://x.com/p") == []


def test_extract_actions_caps_at_max_actions_per_page():
    from app.actions import MAX_ACTIONS_PER_PAGE

    html = "".join(f'<button id="btn-{i}">Add to Cart {i}</button>' for i in range(150))
    actions = extract_actions(html, "https://x.com/p")
    assert len(actions) == MAX_ACTIONS_PER_PAGE


# ---- storage.load_page_actions (DB) -----------------------------------------


async def _seed_crawl(tenant_id: str, site_url: str, job_id: str, pages: list[Page]) -> None:
    await storage.create_job(job_id, site_url, tenant_id)
    await storage.mark_persisting(job_id, len(pages))
    await storage.persist_pages(job_id, tenant_id, site_url, pages)


async def _cleanup_jobs(*job_ids: str) -> None:
    pool = await storage._pool()
    for job_id in job_ids:
        await pool.execute("DELETE FROM jobs WHERE job_id = $1", job_id)


async def test_load_page_actions_returns_captured_actions(pg_tenant):
    tenant_id, site_url = pg_tenant
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    from app.links import page_key

    page = Page(
        url=f"{site_url}/product",
        markdown="Buy it.",
        page_actions=[
            PageAction(kind="add_to_cart", label="Add to Cart", role="button", selectors=["#add"])
        ],
    )
    try:
        await _seed_crawl(tenant_id, site_url, job_id, [page])
        record = await storage.load_page_actions(tenant_id, page_key(page.url), site_url)
        assert record is not None
        assert record.job_id == job_id
        assert record.page_url == page.url
        assert len(record.page_actions) == 1
        assert record.page_actions[0].kind == "add_to_cart"
        assert record.page_actions[0].selectors == ["#add"]
    finally:
        await _cleanup_jobs(job_id)


async def test_load_page_actions_unknown_page_returns_none(pg_tenant):
    tenant_id, site_url = pg_tenant
    from app.links import page_key

    record = await storage.load_page_actions(tenant_id, page_key(f"{site_url}/nope"), site_url)
    assert record is None


async def test_load_page_actions_null_column_returns_empty_list(pg_tenant):
    tenant_id, site_url = pg_tenant
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    from app.links import page_key

    page = Page(url=f"{site_url}/no-actions", markdown="Nothing clickable here.")
    try:
        await _seed_crawl(tenant_id, site_url, job_id, [page])
        record = await storage.load_page_actions(tenant_id, page_key(page.url), site_url)
        assert record is not None
        assert record.page_actions == []
    finally:
        await _cleanup_jobs(job_id)
