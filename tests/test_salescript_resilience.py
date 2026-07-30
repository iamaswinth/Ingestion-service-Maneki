"""Tests for the 529/overloaded-error resilience added to app/salescript/errors.py
and app/salescript/graph.py: readable failure messages, the transient-error
retry predicate, per-page fail-open fact extraction, the extraction floor,
and the fan-out/fan-in behavior end to end. No network and no Postgres —
every LLM call is monkeypatched; real anthropic.* exception classes are
constructed directly so the isinstance ladders are exercised for real.
"""

import json

import anthropic
import httpx
import pytest

from app.salescript import graph
from app.salescript.errors import FactExtractionFloorError, describe_failure


def _minimal_profile(archetype: str = "portfolio") -> dict:
    """A minimal but schema-valid SiteProfile payload (app/models.py)."""
    return {
        "archetype": archetype,
        "reasoning": "The site is a single person's project showcase.",
        "audience": "prospective clients evaluating the person's work",
        "visitor_goals": ["see recent work", "check availability"],
        "conversion_triggers": ["seeing relevant past work", "clear rates"],
        "primary_action": "get in touch about a project",
        "tone": "warm and direct",
        "publishes_pricing": False,
    }


def _minimal_script() -> dict:
    """A minimal but schema-valid SalesScript payload, mirroring
    tests/test_salescript_store.py::_minimal_script."""
    return {
        "opening_hook": "Hi, thanks for stopping by — what brought you here today?",
        "discovery_questions": [
            {"stage": "situation", "question": "What does your team use today?"}
        ],
        "value_props": [{"pain_point": "slow onboarding", "value_prop": "same-day setup"}],
        "objection_handling": [
            {"objection": "too expensive", "response": "we have a free tier", "covered": True}
        ],
        "proof_points": [
            {"claim": "500+ teams onboarded", "reinforces": "value prop: setup speed"}
        ],
        "pricing_talk_track": "Plans start free, pro is $10/mo.",
        "differentiators": "We're the only one with same-day setup.",
        "closing_cta": "Want to start a free trial today?",
        "qualification_signals": ["team size", "current tool"],
    }


def _status_error(cls, status: int, *, request_id: str = None, message: str = None):
    """Builds a real anthropic APIStatusError subclass instance with a real
    httpx.Response underneath it, so describe_failure/_is_transient_llm_error
    exercise the actual SDK types instead of stand-ins."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    headers = {"request-id": request_id} if request_id else {}
    response = httpx.Response(status, request=request, headers=headers)
    body = (
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        if status == 529
        else None
    )
    return cls(message or f"Error code: {status}", response=response, body=body)


def _connection_error(timeout: bool = False):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    if timeout:
        return anthropic.APITimeoutError(request=request)
    return anthropic.APIConnectionError(message="Connection failed.", request=request)


# --------------------------------------------------------------------------
# describe_failure
# --------------------------------------------------------------------------


def test_describe_failure_529_is_readable_and_hides_raw_dict():
    exc = _status_error(anthropic.OverloadedError, 529, request_id="req_abc123")
    message = describe_failure(exc)
    assert "overloaded" in message.lower()
    assert "529" in message
    assert "try" in message.lower()
    # Regression guard for the actual reported bug: the raw error dict must
    # not leak into the dashboard-facing string.
    assert "overloaded_error" not in message
    assert "{'type'" not in message


def test_describe_failure_preserves_request_id():
    exc = _status_error(anthropic.OverloadedError, 529, request_id="req_xyz789")
    assert "req_xyz789" in describe_failure(exc)


def test_describe_failure_auth_error_says_retry_wont_help():
    exc = _status_error(anthropic.AuthenticationError, 401)
    message = describe_failure(exc)
    assert "retrying will not help" in message.lower()
    assert "try again in a few minutes" not in message.lower()


def test_describe_failure_timeout_classified_as_timeout_not_connection():
    exc = _connection_error(timeout=True)
    message = describe_failure(exc)
    assert "timed out" in message.lower()
    assert "network error" not in message.lower()


def test_describe_failure_connection_error_without_timeout():
    exc = _connection_error(timeout=False)
    message = describe_failure(exc)
    assert "network error" in message.lower() or "reach the anthropic api" in message.lower()


def test_describe_failure_server_error_vs_client_error():
    server = describe_failure(_status_error(anthropic.APIStatusError, 503))
    assert "server error" in server.lower()

    client = describe_failure(_status_error(anthropic.BadRequestError, 400))
    assert "configuration or code problem" in client.lower()


def test_describe_failure_floor_error_returned_verbatim():
    exc = FactExtractionFloorError("Fact extraction failed for 8 of 10 pages...")
    assert describe_failure(exc) == str(exc)


def test_describe_failure_detail_suffix_is_bounded():
    exc = _status_error(anthropic.APIStatusError, 503, message="x" * 5000)
    assert len(describe_failure(exc)) < 500


# --------------------------------------------------------------------------
# _is_transient_llm_error
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_exc,expected",
    [
        (lambda: _status_error(anthropic.OverloadedError, 529), True),
        (lambda: _status_error(anthropic.APIStatusError, 503), True),
        (lambda: _status_error(anthropic.APIStatusError, 500), True),
        (lambda: _status_error(anthropic.RateLimitError, 429), True),
        (lambda: _status_error(anthropic.APIStatusError, 409), True),
        (lambda: _status_error(anthropic.APIStatusError, 408), True),
        (lambda: _connection_error(timeout=True), True),
        (lambda: _connection_error(timeout=False), True),
        (lambda: _status_error(anthropic.BadRequestError, 400), False),
        (lambda: _status_error(anthropic.AuthenticationError, 401), False),
        (lambda: _status_error(anthropic.PermissionDeniedError, 403), False),
        (lambda: _status_error(anthropic.NotFoundError, 404), False),
        (lambda: _status_error(anthropic.UnprocessableEntityError, 422), False),
        (lambda: ValueError("bad json"), False),
        (lambda: json.JSONDecodeError("msg", "doc", 0), False),
        (lambda: FactExtractionFloorError("too many failures"), False),
    ],
)
def test_is_transient_llm_error(make_exc, expected):
    assert graph._is_transient_llm_error(make_exc()) is expected


# --------------------------------------------------------------------------
# extract_facts_one fail-open
# --------------------------------------------------------------------------


async def test_extract_facts_one_fails_open_on_api_error(monkeypatch):
    async def fake_call(page):
        raise _status_error(anthropic.OverloadedError, 529)

    monkeypatch.setattr(graph, "_call_extract_facts", fake_call)
    result = await graph.extract_facts_one({"page": {"url": "https://example.com/a"}})

    assert result.get("facts") is None or result["facts"] == []
    assert result["extract_failures"] == 1
    assert len(result["extract_errors"]) == 1
    assert "https://example.com/a" in result["extract_errors"][0]


async def test_extract_facts_one_fails_open_on_malformed_json(monkeypatch):
    async def fake_call(page):
        return "{not valid json"

    monkeypatch.setattr(graph, "_call_extract_facts", fake_call)
    result = await graph.extract_facts_one({"page": {"url": "https://example.com/b"}})

    assert result["extract_failures"] == 1
    assert "https://example.com/b" in result["extract_errors"][0]


async def test_extract_facts_one_happy_path_filters_blank_and_nonstring(monkeypatch):
    async def fake_call(page):
        return json.dumps({"facts": ["real fact", "  ", "", 42, None, "another fact"]})

    monkeypatch.setattr(graph, "_call_extract_facts", fake_call)
    result = await graph.extract_facts_one({"page": {"url": "https://example.com/c"}})

    assert result["facts"] == ["real fact", "another fact"]
    assert "extract_failures" not in result


# --------------------------------------------------------------------------
# _check_extraction_floor
# --------------------------------------------------------------------------


def _state(total: int, failed: int, facts: list) -> dict:
    return {
        "pages": [{"url": f"p{i}"} for i in range(total)],
        "extract_failures": failed,
        "facts": facts,
        "extract_errors": [f"p{i}: OverloadedError: boom" for i in range(failed)],
    }


def test_extraction_floor_passes_at_boundary():
    # 10 pages, 5 failed -> 50% success == the 0.5 ratio, should pass.
    graph._check_extraction_floor(_state(10, 5, ["fact"] * 5))


def test_extraction_floor_raises_when_majority_failed():
    with pytest.raises(FactExtractionFloorError) as exc_info:
        graph._check_extraction_floor(_state(10, 6, ["fact"] * 4))
    assert "6 of 10" in str(exc_info.value)


def test_extraction_floor_raises_when_no_facts_even_if_no_failures():
    with pytest.raises(FactExtractionFloorError):
        graph._check_extraction_floor(_state(10, 0, []))


def test_extraction_floor_single_page_failure_raises():
    # Guards max(1, ...): a ratio of 0.5 on 1 page would round down to 0
    # without the floor, which would wrongly let a single failed page pass.
    with pytest.raises(FactExtractionFloorError):
        graph._check_extraction_floor(_state(1, 1, []))


def test_extraction_floor_samples_at_most_three_errors():
    state = _state(100, 60, ["fact"] * 40)
    with pytest.raises(FactExtractionFloorError) as exc_info:
        graph._check_extraction_floor(state)
    message = str(exc_info.value)
    assert message.count("OverloadedError") == 3


# --------------------------------------------------------------------------
# End-to-end fail-open through the real Pregel Send fan-out
# --------------------------------------------------------------------------


def _fake_create_structured(schema: dict):
    title = schema.get("title")
    if title == "SiteProfile":
        return json.dumps(_minimal_profile())
    if title == "SalesScript":
        return json.dumps(_minimal_script())
    if title == "SalesScriptCritique":
        return json.dumps({"passed": True})
    raise AssertionError(f"unexpected schema in fake _create_structured: {title}")


def _pages(n: int) -> list:
    return [{"url": f"https://example.com/page{i}", "title": f"Page {i}"} for i in range(n)]


async def test_graph_survives_one_page_overloaded(monkeypatch):
    async def fake_call_extract_facts(page):
        if page["url"] == "https://example.com/page1":
            raise _status_error(anthropic.OverloadedError, 529)
        return json.dumps({"facts": [f"fact from {page['url']}"]})

    async def fake_create_structured(*, model, max_tokens, system, schema, user_content):
        return _fake_create_structured(schema)

    monkeypatch.setattr(graph, "_call_extract_facts", fake_call_extract_facts)
    monkeypatch.setattr(graph, "_create_structured", fake_create_structured)

    g = graph.build_graph()
    final_state = await g.ainvoke(
        {
            "tenant_id": "t1",
            "job_id": "job-1",
            "site_url": "https://example.com",
            "pages": _pages(5),
            "facts": [],
            "extract_failures": 0,
            "extract_errors": [],
            "profile": None,
            "script": None,
            "critique": None,
            "revision_count": 0,
            "max_revisions": 2,
        }
    )

    assert final_state["script"] is not None
    assert final_state["extract_failures"] == 1
    surviving = [f for f in final_state["facts"] if "page1" not in f]
    assert len(surviving) == 4


async def test_graph_raises_floor_error_when_all_pages_fail(monkeypatch):
    async def fake_call_extract_facts(page):
        raise _status_error(anthropic.OverloadedError, 529)

    async def fake_create_structured(*, model, max_tokens, system, schema, user_content):
        return _fake_create_structured(schema)

    monkeypatch.setattr(graph, "_call_extract_facts", fake_call_extract_facts)
    monkeypatch.setattr(graph, "_create_structured", fake_create_structured)

    g = graph.build_graph()
    with pytest.raises(FactExtractionFloorError):
        await g.ainvoke(
            {
                "tenant_id": "t1",
                "job_id": "job-1",
                "site_url": "https://example.com",
                "pages": _pages(5),
                "facts": [],
                "extract_failures": 0,
                "extract_errors": [],
                "profile": None,
                "script": None,
                "critique": None,
                "revision_count": 0,
                "max_revisions": 2,
            }
        )
