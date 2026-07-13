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
import difflib
import json
import operator
import re
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

# 1-4 consecutive Title-Case words, e.g. "Odyssey", "Yuan Teoh", "Merouane
# Zouaid", "Comp AI" -- a cheap heuristic for "this looks like a proper
# noun", used by _find_name_drift below. Must include single words: a
# possessive like "Odyssey's" breaks the multi-word chain at the apostrophe
# (lowercase "s" doesn't continue the pattern), so a drifted single-word
# name would never become a candidate if 2+ words were required. Not a real
# NER model; deliberately simple.
_NAME_RE = re.compile(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,3}\b")

# Common sentence-fillers that only end up capitalized because they sit at
# the start of a mid-sentence clause (", A Google software engineer, ...")
# or a sentence -- not because they're part of a name. Stripped from the
# front of a candidate before whitelist comparison so "A Google" doesn't
# get treated as drift against the real name "Google".
_LEADING_STOPWORDS = {
    "A", "An", "The", "In", "On", "At", "For", "By", "Is", "It", "This",
    "That", "These", "Those", "We", "Our", "You", "Your", "If", "So",
    "And", "But", "Or", "With", "From", "To", "Once", "When", "While",
}


def _strip_leading_stopword(phrase: str) -> str:
    first, _, rest = phrase.partition(" ")
    return rest if rest and first in _LEADING_STOPWORDS else phrase


def _is_abbreviation(candidate: str, whitelist: set[str]) -> bool:
    """True if `candidate` is a whitespace-boundary prefix or suffix of a
    whitelisted phrase, or vice versa -- e.g. "Agnost" vs. "Agnost AI"
    (prefix), or "Voice BDRs" vs. "Our Voice BDRs" (suffix, e.g. a script
    rephrasing "our" as "your" for a prospect drops the leading word). A
    company casually dropping its own suffix mid-sentence, or a possessive
    getting rephrased, is normal variation, not the character-level drift
    this check is meant to catch; only flag a *different-spelling* near
    miss, not a *shorter/longer* correct one."""
    return any(
        w != candidate
        and (
            w.startswith(candidate + " ")
            or candidate.startswith(w + " ")
            or w.endswith(" " + candidate)
            or candidate.endswith(" " + w)
        )
        for w in whitelist
    )


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


@traceable(name="derive_icp_llm", run_type="llm")
async def _call_derive_icp(facts: list[str]) -> str:
    return await _create_structured(
        model=settings.sales_script_model,
        max_tokens=3072,
        system=prompts.DERIVE_ICP_SYSTEM_PROMPT,
        schema=IcpProfile.model_json_schema(),
        user_content=prompts.build_derive_icp_prompt(facts),
    )


async def derive_icp(state: SalesScriptState) -> dict:
    text = await _call_derive_icp(state["facts"])
    return {"icp": json.loads(text)}


@traceable(name="draft_script_llm", run_type="llm")
async def _call_draft_script(prompt: str) -> str:
    return await _create_structured(
        model=settings.sales_script_model,
        # SalesScript grew (proof_points, qualification_signals, staged
        # discovery_questions, a fixed 5+-category objection checklist) —
        # 6144 was sized for the smaller pre-expansion schema and now
        # truncates mid-JSON on a normal-sized script, not just large sites.
        max_tokens=8192,
        system=prompts.DRAFT_SCRIPT_SYSTEM_PROMPT,
        schema=SalesScript.model_json_schema(),
        user_content=prompt,
    )


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
    return await _create_structured(
        model=settings.sales_script_model,
        max_tokens=6144,
        system=prompts.CRITIQUE_SYSTEM_PROMPT,
        schema=SalesScriptCritique.model_json_schema(),
        user_content=prompts.build_critique_prompt(facts, script),
    )


def _all_strings(obj) -> list[str]:
    """Every string leaf in a nested dict/list structure, e.g. a SalesScript
    dict -- used to scan the whole script for name-shaped text regardless of
    which field it landed in."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _all_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _all_strings(v)]
    return []


def _find_name_drift(facts: list[str], script: dict) -> list[str]:
    """Deterministic, non-LLM catch for the failure mode where draft_script
    autocorrects an unusual real name to a more common-sounding word (e.g.
    "Odysser" -> "Odyssey") and critique's own LLM call doesn't reliably
    notice, even when explicitly instructed to check exact spelling --
    LLMs are unreliable at this kind of character-level string comparison.

    Builds a whitelist of name-shaped phrases actually present in the facts,
    then flags any name-shaped phrase in the script that isn't an exact
    match but *is* a close one -- that gap is exactly what drift looks like.
    A phrase that exactly matches the whitelist never reaches the fuzzy
    check, so two different real similar-sounding names (each individually
    grounded in the facts) don't false-positive against each other.
    """
    whitelist = set(_NAME_RE.findall(" ".join(facts)))
    raw_candidates = set(_NAME_RE.findall(" ".join(_all_strings(script))))
    candidates = {_strip_leading_stopword(c) for c in raw_candidates}

    # Short strings make the fuzzy check noisy -- e.g. "Isn" (from a
    # sentence-opening "Isn't...") vs. "In" (from a quoted testimonial's
    # opening word) share enough characters to clear the ratio threshold
    # despite being unrelated. Real name drift (Odysser/Odyssey) is well
    # above this length; a common short capitalized word isn't a name.
    long_whitelist = {w for w in whitelist if len(w) >= 4}

    issues = []
    for candidate in sorted(candidates - whitelist):
        if len(candidate) < 4 or _is_abbreviation(candidate, whitelist):
            continue
        match = difflib.get_close_matches(candidate, long_whitelist, n=1, cutoff=0.8)
        if match:
            issues.append(
                f"\"{candidate}\" does not exactly match the source spelling "
                f"\"{match[0]}\" -- likely name drift, not a genuine claim."
            )
    return issues


async def critique_node(state: SalesScriptState) -> dict:
    text = await _call_critique(state["facts"], state["script"])
    critique = json.loads(text)

    drift_issues = _find_name_drift(state["facts"], state["script"])
    if drift_issues:
        critique["passed"] = False
        critique["issues"] = critique.get("issues", []) + drift_issues

    return {"critique": critique}


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
