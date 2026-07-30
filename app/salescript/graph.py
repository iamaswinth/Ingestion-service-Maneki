"""The sales-script LangGraph agent.

    START --(Send fan-out, one per page)--> extract_facts_one x N (Haiku)
        --(fan-in via `facts` reducer)--> profile_site (Sonnet)
        --> draft_script (Sonnet) <-------------------+
        --> critique (Sonnet) ---(revise, bounded)-----+
        --(finalize)--> END

`profile_site` classifies what kind of site this is (SiteArchetype) before
anything gets written — draft_script and critique resolve the matching
app/salescript/playbooks.py entry and build their system prompts from it, so
the script a portfolio gets is shaped differently from one a SaaS company
gets, without changing the graph topology.

Runs in-process (invoked from app/salescript/service.py via BackgroundTasks),
not a separate LangGraph Server — see the plan doc for why. No checkpointer:
each run is a single request/response cycle, not a resumable session.

Calls `anthropic.AsyncAnthropic` directly, same idiom as
app/ingestion/questions.py — no langchain-core dependency needed for this.
"""

import asyncio
import json
import logging
import math
import operator
from typing import Annotated, Literal, Optional, TypedDict

import anthropic
import httpx
from anthropic import AsyncAnthropic
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send
from langsmith import traceable

from ..config import settings
from ..models import SalesScript, SalesScriptCritique, SiteProfile
from . import playbooks, prompts
from .errors import FactExtractionFloorError

logger = logging.getLogger(__name__)

_client: Optional[AsyncAnthropic] = None
_extract_semaphore = asyncio.Semaphore(settings.sales_script_max_concurrent_extractions)

_EXTRACT_FACTS_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
    "additionalProperties": False,
}


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            max_retries=settings.sales_script_anthropic_max_retries,
            # The SDK default is a 600s read timeout, applied per attempt — a
            # hung stage plus retries can outlive store._STALE_GENERATION
            # (30 min), letting a retry re-claim the row out from under a
            # still-running worker. 240s clears a non-streaming 8192-token
            # Sonnet draft_script generation with headroom.
            timeout=httpx.Timeout(settings.sales_script_timeout_seconds, connect=5.0),
        )
    return _client


def _text_of(response) -> str:
    """Raises a clear error (with stop_reason) instead of the cryptic
    'coroutine raised StopIteration' that a bare `next()` with no default
    produces when a response has no text block — e.g. when it was truncated
    before any text was written (stop_reason='max_tokens')."""
    for block in response.content:
        if block.type == "text":
            return block.text
    raise ValueError(f"No text content block in response (stop_reason={response.stop_reason})")


async def _create_structured(
    *, model: str, max_tokens: int, system: str, schema: dict, user_content: str
) -> str:
    """One structured (json_schema) call, with one retry at double the token
    budget if the first attempt was truncated by hitting max_tokens — whether
    that cut it off before any text was written, or mid-generation, leaving a
    partial/invalid JSON string. The latter is the more common case in
    practice: a verbose generation (a bigger site producing more objections/
    value props/proof points to write, say) usually gets partway through the
    JSON before running out of budget, not stopped at zero characters, so
    checking only "no text block at all" misses most real truncations and
    lets broken JSON reach the caller's json.loads.

    A single verbose generation can occasionally overrun even a generous
    static max_tokens — this is ordinary LLM output-length variance, not a
    bug worth failing the whole graph run over, so retry once with headroom
    before surfacing an error.
    """
    budget = max_tokens
    for attempt in range(2):
        response = await _get_client().messages.create(
            model=model,
            max_tokens=budget,
            system=system,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user_content}],
        )
        try:
            text = _text_of(response)
        except ValueError:
            if attempt == 1:
                raise
            budget *= 2
            continue
        if response.stop_reason == "max_tokens" and attempt == 0:
            budget *= 2
            continue
        return text


class SalesScriptState(TypedDict):
    tenant_id: str
    job_id: str
    site_url: str
    pages: list[dict]
    facts: Annotated[list[str], operator.add]
    extract_failures: Annotated[int, operator.add]
    extract_errors: Annotated[list[str], operator.add]
    profile: Optional[dict]
    script: Optional[dict]
    critique: Optional[dict]
    revision_count: int
    max_revisions: int


def _dispatch_extract(state: SalesScriptState) -> list[Send]:
    return [
        Send("extract_facts_one", {"page": page, "tenant_id": state["tenant_id"]})
        for page in state["pages"]
    ]


@traceable(name="extract_facts_llm", run_type="llm")
async def _call_extract_facts(page: dict) -> str:
    async with _extract_semaphore:
        return await _create_structured(
            model=settings.sales_script_extract_model,
            max_tokens=1536,
            system=prompts.EXTRACT_FACTS_SYSTEM_PROMPT,
            schema=_EXTRACT_FACTS_SCHEMA,
            user_content=prompts.build_extract_facts_prompt(page),
        )


async def extract_facts_one(payload: dict) -> dict:
    """Fail-open per page, the Send-fan-out equivalent of
    app/ingestion/questions.py::generate_questions's
    `gather(..., return_exceptions=True)`: each page is its own node task, so
    letting one page's 529 (or a json.loads failure on its output) raise here
    would abort the whole superstep and discard every sibling page's facts —
    and with no checkpointer there is nothing to resume from, so the retry
    re-extracts all N pages. A minority of pages missing costs the script a
    few facts; `_check_extraction_floor` in profile_site turns "most pages
    failed" back into a loud, honest failure.

    Catches Exception, not BaseException: asyncio.CancelledError must stay
    uncaught so the run-level deadline in service.py::run_generation can
    actually stop this node.
    """
    page = payload["page"]
    try:
        text = await _call_extract_facts(page)
        facts = [
            f for f in json.loads(text).get("facts", []) if isinstance(f, str) and f.strip()
        ]
        return {"facts": facts}
    except Exception as exc:
        logger.warning(
            "fact extraction failed for url=%s: %s: %s", page.get("url"), type(exc).__name__, exc
        )
        return {
            "extract_failures": 1,
            "extract_errors": [f"{page.get('url')}: {type(exc).__name__}: {exc}"[:200]],
        }


@traceable(name="profile_site_llm", run_type="llm")
async def _call_profile_site(facts: list[str]) -> str:
    return await _create_structured(
        model=settings.sales_script_model,
        max_tokens=3072,
        system=prompts.PROFILE_SITE_SYSTEM_PROMPT,
        schema=SiteProfile.model_json_schema(),
        user_content=prompts.build_profile_site_prompt(facts),
    )


def _check_extraction_floor(state: SalesScriptState) -> None:
    """Fan-in gate. extract_facts_one is fail-open per page, which without a
    floor would happily hand profile_site the 2 facts scraped off the one
    page that happened to succeed and produce a confidently-wrong script the
    owner has no way to tell apart from a good one. Fail loudly instead.

    Counts failed *pages*, not facts: a nav-only page returning zero facts is
    a success, not a failure, and len(facts) alone can't tell the two apart.
    """
    total = len(state.get("pages") or ())
    failed = state.get("extract_failures") or 0
    facts = state.get("facts") or []
    needed = max(1, math.ceil(total * settings.sales_script_min_extract_success_ratio))
    if facts and (total - failed) >= needed:
        if failed:
            logger.warning(
                "fact extraction: %d/%d pages failed; continuing with %d facts",
                failed,
                total,
                len(facts),
            )
        return
    sample = "; ".join((state.get("extract_errors") or [])[:3])
    raise FactExtractionFloorError(
        f"Fact extraction failed for {failed} of {total} pages and collected "
        f"{len(facts)} facts — too few to write a script from. First errors: {sample}"
    )


async def profile_site(state: SalesScriptState) -> dict:
    _check_extraction_floor(state)
    text = await _call_profile_site(state["facts"])
    return {"profile": json.loads(text)}


@traceable(name="draft_script_llm", run_type="llm")
async def _call_draft_script(prompt: str, system: str) -> str:
    return await _create_structured(
        model=settings.sales_script_model,
        # SalesScript grew (proof_points, qualification_signals, staged
        # discovery_questions, a fixed 5+-category objection checklist) —
        # 6144 was sized for the smaller pre-expansion schema and now
        # truncates mid-JSON on a normal-sized script, not just large sites.
        max_tokens=8192,
        system=system,
        schema=SalesScript.model_json_schema(),
        user_content=prompt,
    )


async def draft_script(state: SalesScriptState) -> dict:
    playbook = playbooks.get(state["profile"].get("archetype"))
    is_revision = state["critique"] is not None
    if is_revision:
        prompt = prompts.build_revise_prompt(
            state["facts"], state["profile"], state["script"], state["critique"]
        )
    else:
        prompt = prompts.build_draft_prompt(state["facts"], state["profile"])
    system = prompts.build_draft_system_prompt(playbook, state["profile"])
    text = await _call_draft_script(prompt, system)
    return {
        "script": json.loads(text),
        "revision_count": state["revision_count"] + (1 if is_revision else 0),
    }


@traceable(name="critique_llm", run_type="llm")
async def _call_critique(facts: list[str], script: dict, system: str) -> str:
    return await _create_structured(
        model=settings.sales_script_model,
        max_tokens=6144,
        system=system,
        schema=SalesScriptCritique.model_json_schema(),
        user_content=prompts.build_critique_prompt(facts, script),
    )


async def critique_node(state: SalesScriptState) -> dict:
    playbook = playbooks.get(state["profile"].get("archetype"))
    system = prompts.build_critique_system_prompt(playbook)
    text = await _call_critique(state["facts"], state["script"], system)
    return {"critique": json.loads(text)}


def _should_revise(state: SalesScriptState) -> Literal["revise", "finalize"]:
    critique = state["critique"]
    if critique and critique.get("passed"):
        return "finalize"
    if state["revision_count"] >= state["max_revisions"]:
        return "finalize"
    return "revise"


def _is_transient_llm_error(exc: Exception) -> bool:
    """retry_on for _LLM_RETRY below.

    Deliberately narrower than langgraph's default_retry_on, whose final
    statement is `return True` for anything it doesn't recognise: an
    anthropic.APIStatusError is neither an httpx.HTTPStatusError nor one of
    the ValueError/RuntimeError/OSError family it special-cases, so the
    default policy would burn attempts and backoff retrying a 400
    invalid_request or a 401 that is never going to succeed. Mirrors the
    SDK's own _should_retry (5xx, 408, 409, 429) plus connection/timeout
    failures. Returns False for FactExtractionFloorError and JSON/ValueError
    failures — those aren't transient and shouldn't eat a retry budget.
    """
    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return False


# Long intervals rather than langgraph's 0.5/2.0 defaults: a 529 capacity wave
# lasts tens of seconds to minutes, and the SDK has already spent several
# seconds retrying before this exception even surfaces to the node. This is a
# background task with no latency budget to protect (see the
# sales_script_anthropic_max_retries comment in config.py) — the same
# reasoning applied at the layer that can actually act on it.
_LLM_RETRY = RetryPolicy(
    initial_interval=8.0,
    backoff_factor=3.0,
    max_interval=90.0,
    max_attempts=4,
    jitter=True,
    retry_on=_is_transient_llm_error,
)


def build_graph():
    builder = StateGraph(SalesScriptState)
    # No retry_policy here: extract_facts_one is fail-open (see its
    # docstring) and never raises, so a RetryPolicy on this node would be
    # dead code. Its retry layer is the SDK's max_retries; kept modest since
    # this node runs once per page and a fan-out-wide RetryPolicy would just
    # add load to an API that's already telling every page it's overloaded.
    builder.add_node("extract_facts_one", extract_facts_one)
    builder.add_node("profile_site", profile_site, retry_policy=_LLM_RETRY)
    builder.add_node("draft_script", draft_script, retry_policy=_LLM_RETRY)
    builder.add_node("critique", critique_node, retry_policy=_LLM_RETRY)

    builder.add_conditional_edges(START, _dispatch_extract, ["extract_facts_one"])
    builder.add_edge("extract_facts_one", "profile_site")
    builder.add_edge("profile_site", "draft_script")
    builder.add_edge("draft_script", "critique")
    builder.add_conditional_edges(
        "critique", _should_revise, {"revise": "draft_script", "finalize": END}
    )
    return builder.compile()


# Module-level singleton; also the langgraph.json entrypoint for local Studio dev.
graph = build_graph()
