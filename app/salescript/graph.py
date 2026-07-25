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
import operator
from typing import Annotated, Literal, Optional, TypedDict

from anthropic import AsyncAnthropic
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langsmith import traceable

from ..config import settings
from ..models import SalesScript, SalesScriptCritique, SiteProfile
from . import playbooks, prompts

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
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
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
    text = await _call_extract_facts(payload["page"])
    facts = [f for f in json.loads(text).get("facts", []) if isinstance(f, str) and f.strip()]
    return {"facts": facts}


@traceable(name="profile_site_llm", run_type="llm")
async def _call_profile_site(facts: list[str]) -> str:
    return await _create_structured(
        model=settings.sales_script_model,
        max_tokens=3072,
        system=prompts.PROFILE_SITE_SYSTEM_PROMPT,
        schema=SiteProfile.model_json_schema(),
        user_content=prompts.build_profile_site_prompt(facts),
    )


async def profile_site(state: SalesScriptState) -> dict:
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


def build_graph():
    builder = StateGraph(SalesScriptState)
    builder.add_node("extract_facts_one", extract_facts_one)
    builder.add_node("profile_site", profile_site)
    builder.add_node("draft_script", draft_script)
    builder.add_node("critique", critique_node)

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
