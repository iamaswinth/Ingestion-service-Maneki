"""The score scale returned by /query must not depend on which retrieval path ran.

`_search_vector_only` returns cosine similarity (0..1). `_search_hybrid` fuses
via RRF, whose raw values are bounded by `(w_vec + w_lex) / (k + 1)` — about
0.033 at the default k=60. Consumers compare `score` against an absolute
threshold, so shipping both scales through one field made
voice_runtime's `retrieval_agentic_score_threshold` (0.35) unreachable under
hybrid search: every hit read as "thin" and the agent answered "I don't have
any information about that" no matter how good the match was.
"""

import pytest

from app import config as config_module
from app.ingestion.store import _normalize_rrf


@pytest.fixture(autouse=True)
def default_weights(monkeypatch):
    monkeypatch.setattr(config_module.settings, "hybrid_rrf_k", 60)
    monkeypatch.setattr(config_module.settings, "hybrid_vector_weight", 1.0)
    monkeypatch.setattr(config_module.settings, "hybrid_lexical_weight", 1.0)


def _raw(vrank=None, lrank=None, k=60, wv=1.0, wl=1.0) -> float:
    """The RRF expression the SQL computes."""
    score = 0.0
    if vrank is not None:
        score += wv / (k + vrank)
    if lrank is not None:
        score += wl / (k + lrank)
    return score


def test_top_of_both_legs_normalizes_to_one():
    assert _normalize_rrf(_raw(vrank=1, lrank=1)) == pytest.approx(1.0)


def test_top_of_one_leg_only_normalizes_to_a_half():
    assert _normalize_rrf(_raw(vrank=1)) == pytest.approx(0.5)


def test_a_strong_hit_clears_the_consumer_threshold():
    # The regression that mattered: raw ≈ 0.033 could never beat 0.35.
    strong = _normalize_rrf(_raw(vrank=1, lrank=2))
    assert strong > 0.35


def test_a_weak_single_leg_hit_is_still_judged_thin():
    # Normalising must not make everything look good — deep results should
    # still fall below the threshold so the agentic follow-up can fire.
    weak = _normalize_rrf(_raw(lrank=50))
    assert weak < 0.35


def test_ordering_is_preserved():
    better = _normalize_rrf(_raw(vrank=1, lrank=1))
    worse = _normalize_rrf(_raw(vrank=8, lrank=12))
    assert better > worse


def test_never_exceeds_one():
    assert _normalize_rrf(999.0) == 1.0


def test_zero_stays_zero():
    assert _normalize_rrf(0.0) == 0.0


def test_scale_is_independent_of_k(monkeypatch):
    # k is a tuning knob for fusion behaviour; changing it must not silently
    # move every consumer's threshold, which is exactly what raw scores did.
    monkeypatch.setattr(config_module.settings, "hybrid_rrf_k", 10)
    assert _normalize_rrf(_raw(vrank=1, lrank=1, k=10)) == pytest.approx(1.0)

    monkeypatch.setattr(config_module.settings, "hybrid_rrf_k", 200)
    assert _normalize_rrf(_raw(vrank=1, lrank=1, k=200)) == pytest.approx(1.0)


def test_respects_asymmetric_weights(monkeypatch):
    monkeypatch.setattr(config_module.settings, "hybrid_vector_weight", 3.0)
    monkeypatch.setattr(config_module.settings, "hybrid_lexical_weight", 1.0)
    assert _normalize_rrf(_raw(vrank=1, lrank=1, wv=3.0, wl=1.0)) == pytest.approx(1.0)
    # Vector-only match now carries three quarters of the available weight.
    assert _normalize_rrf(_raw(vrank=1, wv=3.0)) == pytest.approx(0.75)


def test_zero_weights_do_not_divide_by_zero(monkeypatch):
    monkeypatch.setattr(config_module.settings, "hybrid_vector_weight", 0.0)
    monkeypatch.setattr(config_module.settings, "hybrid_lexical_weight", 0.0)
    assert _normalize_rrf(0.0) == 0.0
