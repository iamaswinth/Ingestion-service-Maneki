"""System prompts + user-message builders for each graph node
(app/salescript/graph.py). Kept separate from the node functions so prompt
iteration doesn't require touching graph wiring.

Not every tenant is a B2B SaaS company selling to a buying committee — a
tenant might be a portfolio, a local service, a nonprofit, or something that
fits no standard commercial pattern at all. `profile_site` classifies which
kind of site this is (SiteArchetype, app/models.py) and draft_script/critique
build their system prompts from the matching app/salescript/playbooks.py
entry, so the *shape* of the reasoning here (ground everything in facts,
never invent) stays fixed while the *content* (what concerns to check for,
what the CTA should invite) adapts to the site.
"""

import json

from .playbooks import Playbook

EXTRACT_FACTS_SYSTEM_PROMPT = (
    "You extract facts from one page of a website that a visitor might "
    "plausibly ask about in conversation. For each fact, write one short, "
    "self-contained, source-grounded sentence — pricing or rates, plans, "
    "products, services offered, features, integrations, guarantees, "
    "credentials or experience, process, hours or location, availability, "
    "contact routes, mission or impact, publications or work samples, "
    "testimonials/social proof (attribute to who said it if named), and "
    "target audience signals. Never invent or infer anything not explicitly "
    "stated on the page. If the page has genuinely nothing a visitor would "
    "ask about (e.g. a bare legal/privacy page), return an empty list."
)


def build_extract_facts_prompt(page: dict) -> str:
    title = page.get("title") or ""
    return f"Page: {title}\nURL: {page['url']}\n\n{page['content']}"


PROFILE_SITE_SYSTEM_PROMPT = (
    "You analyze facts gathered from a website and classify what kind of "
    "site it is, so a downstream writer can tailor a conversational script "
    "to it instead of assuming it's always a B2B software company. Base "
    "every inference on the facts given — do not invent details that aren't "
    "implied by them.\n\n"
    "Classify `archetype` as exactly one of:\n"
    "- saas_product: software or an app sold by subscription or licence\n"
    "- ecommerce: sells physical or digital goods, with a cart/checkout\n"
    "- local_service: a location-bound business — restaurant, clinic, "
    "salon, gym, trade/repair service\n"
    "- professional_services: an agency, studio, or consultancy selling "
    "engagements/projects/retainers rather than a packaged product\n"
    "- portfolio: an individual showcasing their work to get hired or "
    "commissioned (designer, photographer, developer, writer, artist, ...)\n"
    "- creator_media: a blog, publication, documentation/open-source "
    "project, newsletter, or personal brand — visitors consume content, "
    "they don't buy a product\n"
    "- nonprofit: an organization seeking donations, volunteers, or support "
    "for a cause, not selling anything\n"
    "- other: none of the above fits — do not force a bad fit onto a site "
    "that is genuinely something else\n\n"
    "Then infer: `audience` (who visits this site), `visitor_goals` (what "
    "they came to do — not \"pain points\", this site may not be selling "
    "against a problem at all), `conversion_triggers` (what would make a "
    "visitor act), `primary_action` (the one thing this site most wants a "
    "visitor to do), `tone` (how the site talks about itself), and "
    "`publishes_pricing` (true only if the facts include an actual price, "
    "rate, or cost figure — not just a \"contact us\")."
)


def build_profile_site_prompt(facts: list[str]) -> str:
    bullets = "\n".join(f"- {f}" for f in facts)
    return f"Facts gathered from the website:\n{bullets}"


_SHARED_DRAFT_PREAMBLE = (
    "You write a script for a voice agent embedded on this website — it "
    "talks with visitors in-page, not over a phone line. Every claim "
    "(pricing, features, guarantees, testimonials) must be traceable to the "
    "provided facts — never invent numbers, features, or policies not "
    "present in them. Write in natural spoken language suitable for a voice "
    "agent to read aloud, not marketing copy.\n\n"
    "Pacing: one idea per sentence. If a sentence has more than one 'and' "
    "or an em-dash aside packed into it, split it into two shorter "
    "sentences — a TTS engine has no natural pause points inside a dense, "
    "multi-clause sentence, and it will read as a wall of sound. This "
    "applies throughout — opening_hook, value_props, objection_handling "
    "responses, everywhere — not just as a general style note.\n\n"
    "differentiators: describe only what makes this site (or its owner) "
    "different in its own terms, grounded in its own stated features/facts. "
    "Never compare to a named or implied competitor — no fact set here "
    "includes competitor data, so any such comparison would be invented.\n\n"
    "proof_points: pair each testimonial or stat from the facts with the "
    "specific concern or value prop it reinforces (set `reinforces` to "
    "something like \"concern: switching cost\" or \"value prop: faster "
    "onboarding\"). Only use testimonials/stats that appear in the facts — "
    "never invent a customer name or number."
)


def _build_objection_guidance(playbook: Playbook) -> str:
    if playbook.concern_checklist:
        checklist = ", ".join(playbook.concern_checklist)
        return (
            "objection_handling: always include an entry for each of these "
            f"fixed categories — {checklist} — plus any additional concerns "
            "implied by the facts. For a category the facts support, write "
            "a grounded response and set covered=true. For a category with "
            "no supporting facts, set covered=false and write a response "
            "that honestly says this needs input from the site owner "
            "rather than inventing specifics."
        )
    return (
        "objection_handling: derive entries entirely from concerns implied "
        "by the facts — there is no fixed checklist for this kind of site, "
        "so do not invent generic commercial objections (price, security, "
        "ROI, ...) it gives no evidence for. For a concern the facts don't "
        "fully address, set covered=false and say so honestly rather than "
        "inventing specifics."
    )


def build_draft_system_prompt(playbook: Playbook, profile: dict) -> str:
    parts = [
        _SHARED_DRAFT_PREAMBLE,
        playbook.mission,
        f"discovery_questions: {playbook.discovery_guidance}",
        _build_objection_guidance(playbook),
        f"pricing_talk_track: {playbook.commercial_guidance}",
        f"closing_cta: {playbook.cta_guidance}",
        f"qualification_signals: {playbook.signals_guidance}",
        (
            f"This site's audience: {profile.get('audience', '')}. Its "
            f"primary goal for a visitor: {profile.get('primary_action', '')}. "
            f"Write in a tone consistent with: {profile.get('tone', '')}."
        ),
    ]
    return "\n\n".join(p for p in parts if p)


def build_draft_prompt(facts: list[str], profile: dict) -> str:
    bullets = "\n".join(f"- {f}" for f in facts)
    return (
        f"Facts:\n{bullets}\n\n"
        f"Site profile:\n{json.dumps(profile, indent=2)}\n\n"
        "Write the initial script."
    )


def build_revise_prompt(facts: list[str], profile: dict, script: dict, critique: dict) -> str:
    bullets = "\n".join(f"- {f}" for f in facts)
    issues = "\n".join(f"- {i}" for i in critique.get("issues", []))
    return (
        f"Facts:\n{bullets}\n\n"
        f"Site profile:\n{json.dumps(profile, indent=2)}\n\n"
        f"Previous draft:\n{json.dumps(script, indent=2)}\n\n"
        f"A reviewer flagged these issues — fix them, keeping everything else "
        f"that already works:\n{issues}"
    )


_SHARED_CRITIQUE_PROMPT = (
    "You are a strict fact-checker reviewing a script before it's used by a "
    "voice agent embedded on the website it was written from. Flag any "
    "claim (pricing, features, guarantees, testimonials, policies) that is "
    "not directly supported by the provided facts. Do not flag stylistic "
    "choices or missing information — only unsupported or contradicted "
    "claims. Pass only if there are zero such issues. Keep each issue to "
    "one short sentence naming the claim and why it's unsupported — no "
    "lengthy justification or quoting large passages back.\n\n"
    "This scope includes the newer fields, not just opening_hook/value_props/"
    "pricing_talk_track/closing_cta: flag a differentiators claim that "
    "implies or names a competitor comparison (no fact set includes "
    "competitor data, so this is always unsupported); flag a proof_point "
    "whose claim — a testimonial quote, named customer, or stat — does not "
    "appear in the facts (fabricated attribution). Do NOT flag an "
    "objection_handling entry with covered=false as an issue — that field is "
    "an intentional, disclosed gap (no supporting facts exist), not a "
    "fabricated claim, so it is correct as long as it doesn't assert facts "
    "it doesn't have.\n\n"
    "Also check every named person or company against the facts' *exact* "
    "spelling — flag a claim that names \"Acme\" when the facts say "
    "\"Acmee\", for example, with the same severity as an invented name. "
    "An unusual or unfamiliar name is exactly the kind the model is most "
    "likely to silently autocorrect to a more familiar-sounding word, so "
    "check names carefully rather than assuming a close match is correct."
)


def build_critique_system_prompt(playbook: Playbook) -> str:
    stages = ", ".join(playbook.discovery_stages)
    stage_clause = (
        f"flag a discovery_question tagged with a stage outside "
        f"({stages}) or mistagged relative to its content"
    )
    return f"{_SHARED_CRITIQUE_PROMPT} Also {stage_clause}."


def build_critique_prompt(facts: list[str], script: dict) -> str:
    bullets = "\n".join(f"- {f}" for f in facts)
    return f"Facts:\n{bullets}\n\nDraft script to check:\n{json.dumps(script, indent=2)}"
