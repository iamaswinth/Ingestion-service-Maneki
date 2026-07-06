"""The sales-script LangGraph agent.

    START --(Send fan-out, one per page)--> extract_facts_one x N (Haiku)
        --(fan-in via `facts` reducer)--> derive_icp (Sonnet)
        --> draft_script (Sonnet) <-------------------+
        --> critique (Sonnet) ---(revise, bounded)-----+
        --(finalize)--> END

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
from ..models import IcpProfile, SalesScript, SalesScriptCritique
from . import prompts

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


class SalesScriptState(TypedDict):
    tenant_id: str
    job_id: str
    site_url: str
    pages: list[dict]
    facts: Annotated[list[str], operator.add]
    icp: Optional[dict]
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
        response = await _get_client().messages.create(
            model=settings.sales_script_extract_model,
            max_tokens=1024,
            system=prompts.EXTRACT_FACTS_SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": _EXTRACT_FACTS_SCHEMA}},
            messages=[{"role": "user", "content": prompts.build_extract_facts_prompt(page)}],
        )
    return _text_of(response)


async def extract_facts_one(payload: dict) -> dict:
    text = await _call_extract_facts(payload["page"])
    facts = [f for f in json.loads(text).get("facts", []) if isinstance(f, str) and f.strip()]
    return {"facts": facts}


@traceable(name="derive_icp_llm", run_type="llm")
async def _call_derive_icp(facts: list[str]) -> str:
    response = await _get_client().messages.create(
        model=settings.sales_script_model,
        max_tokens=2048,
        system=prompts.DERIVE_ICP_SYSTEM_PROMPT,
        output_config={
            "format": {"type": "json_schema", "schema": IcpProfile.model_json_schema()}
        },
        messages=[{"role": "user", "content": prompts.build_derive_icp_prompt(facts)}],
    )
    return _text_of(response)


async def derive_icp(state: SalesScriptState) -> dict:
    text = await _call_derive_icp(state["facts"])
    return {"icp": json.loads(text)}


@traceable(name="draft_script_llm", run_type="llm")
async def _call_draft_script(prompt: str) -> str:
    response = await _get_client().messages.create(
        model=settings.sales_script_model,
        max_tokens=4096,
        system=prompts.DRAFT_SCRIPT_SYSTEM_PROMPT,
        output_config={
            "format": {"type": "json_schema", "schema": SalesScript.model_json_schema()}
        },
        messages=[{"role": "user", "content": prompt}],
    )
    return _text_of(response)


async def draft_script(state: SalesScriptState) -> dict:
    is_revision = state["critique"] is not None
    if is_revision:
        prompt = prompts.build_revise_prompt(
            state["facts"], state["icp"], state["script"], state["critique"]
        )
    else:
        prompt = prompts.build_draft_prompt(state["facts"], state["icp"])
    text = await _call_draft_script(prompt)
    return {
        "script": json.loads(text),
        "revision_count": state["revision_count"] + (1 if is_revision else 0),
    }


@traceable(name="critique_llm", run_type="llm")
async def _call_critique(facts: list[str], script: dict) -> str:
    response = await _get_client().messages.create(
        model=settings.sales_script_model,
        max_tokens=4096,
        system=prompts.CRITIQUE_SYSTEM_PROMPT,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": SalesScriptCritique.model_json_schema(),
            }
        },
        messages=[{"role": "user", "content": prompts.build_critique_prompt(facts, script)}],
    )
    return _text_of(response)


async def critique_node(state: SalesScriptState) -> dict:
    text = await _call_critique(state["facts"], state["script"])
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
    builder.add_node("derive_icp", derive_icp)
    builder.add_node("draft_script", draft_script)
    builder.add_node("critique", critique_node)

    builder.add_conditional_edges(START, _dispatch_extract, ["extract_facts_one"])
    builder.add_edge("extract_facts_one", "derive_icp")
    builder.add_edge("derive_icp", "draft_script")
    builder.add_edge("draft_script", "critique")
    builder.add_conditional_edges(
        "critique", _should_revise, {"revise": "draft_script", "finalize": END}
    )
    return builder.compile()


# Module-level singleton; also the langgraph.json entrypoint for local Studio dev.
graph = build_graph()
