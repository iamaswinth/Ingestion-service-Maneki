"""System prompts + user-message builders for each graph node
(app/salescript/graph.py). Kept separate from the node functions so prompt
iteration doesn't require touching graph wiring.
"""

import json

EXTRACT_FACTS_SYSTEM_PROMPT = (
    "You extract sales-relevant facts from one page of a company's website. "
    "For each fact, write one short, self-contained, source-grounded sentence "
    "— pricing figures, plan names, features, integrations, guarantees, "
    "testimonials/social proof (attribute to who said it if named), target "
    "customer signals. Never invent or infer anything not explicitly stated "
    "on the page. If the page has no sales-relevant content (e.g. a bare "
    "legal/privacy page), return an empty list."
)


def build_extract_facts_prompt(page: dict) -> str:
    title = page.get("title") or ""
    return f"Page: {title}\nURL: {page['url']}\n\n{page['content']}"


DERIVE_ICP_SYSTEM_PROMPT = (
    "You analyze a set of facts gathered from a company's website and infer "
    "their ideal customer profile for a sales conversation. Base every "
    "inference on the facts given — do not invent details about the "
    "company that aren't implied by them."
)


def build_derive_icp_prompt(facts: list[str]) -> str:
    bullets = "\n".join(f"- {f}" for f in facts)
    return f"Facts gathered from the company's website:\n{bullets}"


DRAFT_SCRIPT_SYSTEM_PROMPT = (
    "You write a sales call script for a voice agent representing this "
    "company. Every claim (pricing, features, guarantees, testimonials) must "
    "be traceable to the provided facts — never invent numbers, features, or "
    "policies not present in them. Write in natural spoken language suitable "
    "for a voice agent to read aloud, not marketing copy."
)


def build_draft_prompt(facts: list[str], icp: dict) -> str:
    bullets = "\n".join(f"- {f}" for f in facts)
    return (
        f"Facts:\n{bullets}\n\n"
        f"Ideal customer profile:\n{json.dumps(icp, indent=2)}\n\n"
        "Write the initial sales script."
    )


def build_revise_prompt(facts: list[str], icp: dict, script: dict, critique: dict) -> str:
    bullets = "\n".join(f"- {f}" for f in facts)
    issues = "\n".join(f"- {i}" for i in critique.get("issues", []))
    return (
        f"Facts:\n{bullets}\n\n"
        f"Ideal customer profile:\n{json.dumps(icp, indent=2)}\n\n"
        f"Previous draft:\n{json.dumps(script, indent=2)}\n\n"
        f"A reviewer flagged these issues — fix them, keeping everything else "
        f"that already works:\n{issues}"
    )


CRITIQUE_SYSTEM_PROMPT = (
    "You are a strict fact-checker reviewing a sales script before it's used "
    "on live calls. Flag any claim (pricing, features, guarantees, "
    "testimonials, policies) that is not directly supported by the provided "
    "facts. Do not flag stylistic choices or missing information — only "
    "unsupported or contradicted claims. Pass only if there are zero such "
    "issues."
)


def build_critique_prompt(facts: list[str], script: dict) -> str:
    bullets = "\n".join(f"- {f}" for f in facts)
    return f"Facts:\n{bullets}\n\nDraft script to check:\n{json.dumps(script, indent=2)}"
