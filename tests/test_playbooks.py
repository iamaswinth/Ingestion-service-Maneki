"""Structural tests for app/salescript/playbooks.py — every archetype has a
complete playbook, and playbooks.get() degrades gracefully for unknown/None
input (the shape a pre-existing sales_scripts row with site_profile IS NULL
takes — see app/salescript/service.py::approve_and_index).
"""

from app.models import SiteArchetype
from app.salescript import playbooks
from app.salescript.playbooks import SECTION_KEYS

# typing.get_args needs the Literal itself, not a string alias of it.
_ALL_ARCHETYPES = SiteArchetype.__args__


def test_every_archetype_has_a_playbook():
    assert set(playbooks.PLAYBOOKS.keys()) == set(_ALL_ARCHETYPES)


def test_every_playbook_covers_all_section_keys():
    for archetype, playbook in playbooks.PLAYBOOKS.items():
        missing = set(SECTION_KEYS) - set(playbook.section_labels.keys())
        assert not missing, f"{archetype} playbook missing labels for {missing}"


def test_every_playbook_has_non_empty_guidance_strings():
    for archetype, playbook in playbooks.PLAYBOOKS.items():
        assert playbook.mission.strip(), archetype
        assert playbook.discovery_guidance.strip(), archetype
        assert playbook.discovery_stages, archetype
        assert playbook.commercial_guidance.strip(), archetype
        assert playbook.cta_guidance.strip(), archetype
        assert playbook.signals_guidance.strip(), archetype


def test_get_falls_back_to_other_for_none():
    assert playbooks.get(None) is playbooks.PLAYBOOKS["other"]


def test_get_falls_back_to_other_for_unknown_value():
    # Guards against a future deployment writing an archetype value this
    # version's PLAYBOOKS dict doesn't recognize yet.
    assert playbooks.get("some_future_archetype") is playbooks.PLAYBOOKS["other"]


def test_get_resolves_known_archetype():
    assert playbooks.get("portfolio") is playbooks.PLAYBOOKS["portfolio"]


def test_other_playbook_has_no_fixed_concern_checklist():
    # "other" is the fully-generic fallback -- it must not silently assume a
    # commercial checklist the way every named archetype does.
    assert playbooks.PLAYBOOKS["other"].concern_checklist == ()


def test_saas_product_keeps_spin_discovery_stages():
    # Locks in that the pre-existing SPIN-staged flow (situation/problem/
    # implication/need_payoff) is preserved for the archetype it always fit.
    assert playbooks.PLAYBOOKS["saas_product"].discovery_stages == (
        "situation",
        "problem",
        "implication",
        "need_payoff",
    )
