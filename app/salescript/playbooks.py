"""Per-archetype guidance that shapes the draft/critique prompts and the
chunk section titles (app/salescript/chunker.py).

`profile_site` (app/salescript/graph.py) infers a SiteArchetype from the
crawl; everything else in the generation graph is the same regardless of
archetype — only the *content* of the guidance handed to draft_script and
critique changes. This is what lets a portfolio site get a script about
availability and rates instead of vendor lock-in and ROI.

Each playbook's guidance strings are written to be dropped directly into
build_draft_system_prompt / build_critique_system_prompt (app/salescript/
prompts.py) as-is — keep them in that voice (imperative, addressed to the
model) rather than descriptive prose.
"""

from dataclasses import dataclass, field

from ..models import SiteArchetype

# The eight section keys app/salescript/chunker.py emits, in the order it
# emits them. Every playbook's `section_labels` must cover all eight —
# enforced by tests/test_playbooks.py — so the chunker never falls back to a
# missing-key error mid-approval.
SECTION_KEYS = (
    "opening_hook",
    "discovery_questions",
    "value_prop",
    "objection",
    "proof_point",
    "pricing_talk_track",
    "differentiators",
    "closing_cta",
)


@dataclass(frozen=True)
class Playbook:
    # One or two sentences replacing "you write a sales script for a company"
    # — sets what kind of conversation this is and who the visitor is to it.
    mission: str
    # The fixed list of concerns objection_handling must always cover for
    # this archetype (mirrors the old hard-coded price/security/effort/
    # lock-in/ROI checklist, but per archetype). Empty for "other" — that
    # archetype derives its checklist from the facts alone.
    concern_checklist: tuple[str, ...]
    # How discovery_questions should be staged for this archetype.
    discovery_guidance: str
    # The stage labels draft_script may use for this archetype's
    # discovery_questions, and critique should treat as valid.
    discovery_stages: tuple[str, ...]
    # What pricing_talk_track should actually contain here.
    commercial_guidance: str
    # What closing_cta should actually invite here.
    cta_guidance: str
    # What qualification_signals should name here.
    signals_guidance: str
    # Chunk/section display titles, keyed by SECTION_KEYS. value_prop and
    # objection are titled per-item by the chunker (e.g. "Value Prop: {pain
    # point}"); the label here is the prefix/noun used in that title.
    section_labels: dict[str, str] = field(default_factory=dict)


PLAYBOOKS: dict[SiteArchetype, Playbook] = {
    "saas_product": Playbook(
        mission=(
            "This is a software product sold by subscription or licence. The "
            "visitor is evaluating whether to buy or sign up, likely on "
            "behalf of a team."
        ),
        concern_checklist=(
            "price",
            "security/compliance",
            "implementation effort",
            "vendor lock-in/switching cost",
            "ROI proof",
        ),
        discovery_guidance=(
            "Order using SPIN — situation, then problem, then implication, "
            "then need_payoff — and tag each question with its stage. This "
            "is the order the agent should ask them in a live conversation, "
            "not an unordered list."
        ),
        discovery_stages=("situation", "problem", "implication", "need_payoff"),
        commercial_guidance=(
            "State plan pricing if the facts include figures. If the facts "
            "include no pricing figures (e.g. the site only offers \"contact "
            "sales\"), do not write a vague deflection — write a confident, "
            "specific bridge to a qualifying question or CTA, e.g. \"Pricing "
            "depends on your team size — let's get you on a quick call with "
            "someone who can quote you accurately.\""
        ),
        cta_guidance=(
            "Invite a clear next step toward buying or signing up — start a "
            "trial, book a demo, or talk to sales."
        ),
        signals_guidance=(
            "Name what to listen for during discovery that signals buying "
            "readiness — e.g. team size, current tool being replaced, budget "
            "authority, urgency/timeline."
        ),
        section_labels={
            "opening_hook": "Opening Hook",
            "discovery_questions": "Discovery Questions",
            "value_prop": "Value Prop",
            "objection": "Objection",
            "proof_point": "Proof Point",
            "pricing_talk_track": "Pricing",
            "differentiators": "Differentiators",
            "closing_cta": "Closing",
        },
    ),
    "ecommerce": Playbook(
        mission=(
            "This site sells physical or digital goods for purchase. The "
            "visitor is deciding whether to buy, often comparing a specific "
            "product against alternatives."
        ),
        concern_checklist=(
            "shipping & delivery time",
            "returns & exchanges",
            "sizing/fit or compatibility",
            "payment security",
            "product authenticity/quality",
        ),
        discovery_guidance=(
            "Order questions from general intent to specific need — what "
            "they're shopping for, then what matters most to them (fit, "
            "budget, use case), then what would make them decide today. Tag "
            "each with a stage: context, goal, then fit."
        ),
        discovery_stages=("context", "goal", "fit"),
        commercial_guidance=(
            "Cover shipping cost/timeline, returns policy, and payment "
            "options if the facts state them. If the facts include no "
            "pricing at all, say so plainly rather than inventing a figure, "
            "and point to where the visitor can see current prices."
        ),
        cta_guidance=(
            "Invite the visitor to view the product, add it to cart, or "
            "check out — the concrete next click, not a vague \"let us know\"."
        ),
        signals_guidance=(
            "Name what to listen for that signals purchase intent — e.g. "
            "specific product interest, budget range, timing (gift, "
            "replacement, restock), size/variant needed."
        ),
        section_labels={
            "opening_hook": "Opening Hook",
            "discovery_questions": "Discovery Questions",
            "value_prop": "Why It's Worth Buying",
            "objection": "Shopper Concern",
            "proof_point": "Proof Point",
            "pricing_talk_track": "Shipping, Returns & Payment",
            "differentiators": "Differentiators",
            "closing_cta": "Closing",
        },
    ),
    "local_service": Playbook(
        mission=(
            "This is a location-bound business (restaurant, clinic, salon, "
            "gym, trade/repair service, etc). The visitor is deciding "
            "whether to visit, book, or call, and whether it's convenient "
            "and trustworthy."
        ),
        concern_checklist=(
            "hours & location",
            "how to book or reserve",
            "price range",
            "accessibility/parking",
            "credentials/experience",
        ),
        discovery_guidance=(
            "Order questions from what the visitor needs, to when/where "
            "they need it, to what would make them book. Tag each with a "
            "stage: context, goal, then fit."
        ),
        discovery_stages=("context", "goal", "fit"),
        commercial_guidance=(
            "State price range if the facts include it. If pricing isn't "
            "published, say so plainly and give a confident way to get an "
            "exact quote (call, visit, or booking link) rather than "
            "deflecting."
        ),
        cta_guidance=(
            "Invite the visitor to book, call, or visit — whichever action "
            "the facts show this business actually wants."
        ),
        signals_guidance=(
            "Name what to listen for that signals readiness to book — e.g. "
            "preferred date/time, location convenience, specific service "
            "needed, urgency."
        ),
        section_labels={
            "opening_hook": "Opening Hook",
            "discovery_questions": "Discovery Questions",
            "value_prop": "Why Choose Us",
            "objection": "Visitor Concern",
            "proof_point": "Proof Point",
            "pricing_talk_track": "Pricing & Booking",
            "differentiators": "Differentiators",
            "closing_cta": "Closing",
        },
    ),
    "professional_services": Playbook(
        mission=(
            "This is an agency, studio, or consultancy that sells "
            "engagements (projects, retainers, contracts) rather than a "
            "packaged product. The visitor is assessing fit before "
            "reaching out."
        ),
        concern_checklist=(
            "price/budget fit",
            "process & timeline",
            "relevant experience/track record",
            "scope (what is and isn't included)",
            "availability/capacity",
        ),
        discovery_guidance=(
            "Order using SPIN — situation, then problem, then implication, "
            "then need_payoff — and tag each question with its stage, the "
            "same as a consultative sales conversation."
        ),
        discovery_stages=("situation", "problem", "implication", "need_payoff"),
        commercial_guidance=(
            "If the facts include pricing/rate figures, state them. If the "
            "site only offers \"get in touch\" or \"request a quote\", don't "
            "deflect — write a confident bridge, e.g. \"Every engagement is "
            "scoped to the project, so the best next step is a quick call to "
            "get you an accurate quote.\""
        ),
        cta_guidance=(
            "Invite the visitor to start a project enquiry or book an intro "
            "call — the concrete next step toward being hired."
        ),
        signals_guidance=(
            "Name what to listen for during discovery — e.g. project scope, "
            "timeline, budget range, decision-maker status."
        ),
        section_labels={
            "opening_hook": "Opening Hook",
            "discovery_questions": "Discovery Questions",
            "value_prop": "Why Work With Us",
            "objection": "Client Concern",
            "proof_point": "Proof Point",
            "pricing_talk_track": "Rates & Engagement",
            "differentiators": "Differentiators",
            "closing_cta": "Closing",
        },
    ),
    "portfolio": Playbook(
        mission=(
            "This is an individual showcasing their work to get hired or "
            "commissioned. The visitor is usually a prospective client, "
            "recruiter, or collaborator deciding whether this person fits "
            "their project. Help them understand the work and make it easy "
            "to get in touch — this is not a sale to close."
        ),
        concern_checklist=(
            "availability & timeline",
            "rates/budget fit",
            "scope (what they do and don't take on)",
            "relevant experience",
            "how to get in touch",
        ),
        discovery_guidance=(
            "Order questions from what kind of work the visitor is looking "
            "for, to their timeline and budget, to what would make them "
            "reach out. Tag each with a stage: context, goal, then fit."
        ),
        discovery_stages=("context", "goal", "fit"),
        commercial_guidance=(
            "State rates only if the facts give a figure. Most portfolios "
            "don't publish rates — in that case, don't deflect vaguely; "
            "write a confident, specific bridge, e.g. \"Rates depend on "
            "project scope — the fastest way to get a number is to reach "
            "out with a bit about what you need.\""
        ),
        cta_guidance=(
            "Invite the visitor to get in touch about a project — email, a "
            "contact form, or whatever contact route the facts show. This "
            "is the whole point of the site, so make it concrete and easy."
        ),
        signals_guidance=(
            "Name what to listen for that signals a real opportunity — e.g. "
            "project type, timeline, budget range, whether they've seen "
            "specific work."
        ),
        section_labels={
            "opening_hook": "Opening Hook",
            "discovery_questions": "Discovery Questions",
            "value_prop": "Why Hire Them",
            "objection": "Visitor Concern",
            "proof_point": "Proof Point",
            "pricing_talk_track": "Rates & Availability",
            "differentiators": "What Makes Them Different",
            "closing_cta": "Get In Touch",
        },
    ),
    "creator_media": Playbook(
        mission=(
            "This is a blog, publication, documentation/open-source project, "
            "newsletter, or personal brand site. The visitor is deciding "
            "whether the content is worth their time and attention, not "
            "whether to buy something."
        ),
        concern_checklist=(),
        discovery_guidance=(
            "Order questions from what the visitor is looking for, to what "
            "would make it worth following/subscribing/using. Tag each with "
            "a stage: context, goal, then fit. Omit any question that would "
            "only make sense in a commercial pitch."
        ),
        discovery_stages=("context", "goal", "fit"),
        commercial_guidance=(
            "Most sites like this have no pricing at all — if the facts "
            "confirm that (e.g. free, open-source, ad-supported), say so "
            "plainly and positively rather than writing a sales bridge. If "
            "the facts do include a price (e.g. a paid newsletter tier), "
            "state it."
        ),
        cta_guidance=(
            "Invite the visitor toward the site's real goal — subscribe, "
            "read more, follow, star/contribute — not a sales close."
        ),
        signals_guidance=(
            "Name what to listen for that signals genuine interest — e.g. "
            "topic they care about, how they found the site, whether they "
            "want to contribute or just read."
        ),
        section_labels={
            "opening_hook": "Opening Hook",
            "discovery_questions": "Discovery Questions",
            "value_prop": "Why It's Worth Your Time",
            "objection": "Visitor Hesitation",
            "proof_point": "Proof Point",
            "pricing_talk_track": "Cost / Access",
            "differentiators": "What Makes This Different",
            "closing_cta": "Next Step",
        },
    ),
    "nonprofit": Playbook(
        mission=(
            "This is a nonprofit or cause-driven organization. The visitor "
            "is deciding whether and how to support the mission — not "
            "buying a product."
        ),
        concern_checklist=(
            "where donations/support actually go",
            "how to get involved (donate, volunteer, both)",
            "credibility/transparency",
            "impact so far",
        ),
        discovery_guidance=(
            "Order questions from what the visitor cares about, to how they "
            "want to help, to what would make them commit. Tag each with a "
            "stage: context, goal, then fit."
        ),
        discovery_stages=("context", "goal", "fit"),
        commercial_guidance=(
            "This is a donation/support ask, not pricing. State suggested "
            "donation amounts or membership tiers if the facts give them; "
            "otherwise describe how to give without inventing a figure."
        ),
        cta_guidance=(
            "Invite the visitor to donate, volunteer, or otherwise support "
            "the mission — whichever the facts show this organization asks "
            "for."
        ),
        signals_guidance=(
            "Name what to listen for that signals willingness to support — "
            "e.g. cause alignment, capacity to give time vs. money, prior "
            "involvement."
        ),
        section_labels={
            "opening_hook": "Opening Hook",
            "discovery_questions": "Discovery Questions",
            "value_prop": "Why This Matters",
            "objection": "Visitor Concern",
            "proof_point": "Proof Point",
            "pricing_talk_track": "How To Give",
            "differentiators": "What Makes This Different",
            "closing_cta": "Get Involved",
        },
    ),
    "other": Playbook(
        mission=(
            "This site doesn't fit a standard commercial or organizational "
            "pattern. Write the script grounded only in what the facts "
            "actually show the site is for and what a visitor would want to "
            "know — don't force a sales or fundraising framing onto it."
        ),
        concern_checklist=(),
        discovery_guidance=(
            "Derive the questions entirely from the facts — what a visitor "
            "would plausibly want to know about this specific site. Tag "
            "each with a stage: context, goal, then fit. Do not invent "
            "commercial concerns (pricing, ROI, security) that the facts "
            "don't support."
        ),
        discovery_stages=("context", "goal", "fit"),
        commercial_guidance=(
            "Only discuss cost/pricing if the facts actually mention it. If "
            "they don't, say plainly that there's nothing to report rather "
            "than writing a sales bridge — this may not be a commercial "
            "site at all."
        ),
        cta_guidance=(
            "Invite whatever next step the facts actually support (contact, "
            "explore further, follow up) — do not invent a CTA the site "
            "doesn't have."
        ),
        signals_guidance=(
            "Name whatever signals from the facts would help a future "
            "conversation with this visitor — keep it grounded in what the "
            "site actually offers."
        ),
        section_labels={
            "opening_hook": "Opening Hook",
            "discovery_questions": "Discovery Questions",
            "value_prop": "Highlight",
            "objection": "Visitor Question",
            "proof_point": "Proof Point",
            "pricing_talk_track": "Cost",
            "differentiators": "What Makes This Different",
            "closing_cta": "Next Step",
        },
    ),
}


def get(archetype: str | None) -> Playbook:
    """Resolve a playbook by archetype string, falling back to "other" for
    None/unknown values — covers sales_scripts rows written before
    site_profile existed, and guards against a future archetype value this
    deployment's PLAYBOOKS dict doesn't yet know about."""
    return PLAYBOOKS.get(archetype, PLAYBOOKS["other"])  # type: ignore[arg-type]
