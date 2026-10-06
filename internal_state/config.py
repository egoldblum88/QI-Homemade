"""
Every setting of the internal state, in one place (docs/07_INTERNAL_STATE.md, section 4.4).

These are engineering settings: chosen by hand, then measured (python -m internal_state.evaluate). Tests and the
benchmark pass their own values with make_config({...}); a change here should be made only with a recorded reason.
"""

import copy

DEFAULTS = {
    # how fast each decaying number returns to its baseline (seconds until half of the difference is gone)
    "half_life_seconds": {"valence": 1200, "activation": 180, "surprise": 300, "novelty": 600},
    # how strongly one episode moves each number. Valence is a moving average of each episode's signal
    # (mean signed operation error + information * this weight): the first design added the signal instead, and the
    # first measurement showed valence pinned at +1.0 after 8 messages (a new system beats its 0.5 priors), D-074
    "gains": {"valence": 0.5, "information": 0.6, "activation": 0.6, "surprise": 0.5, "novelty": 0.5},
    # the expectations: small learned predictors of the system's own outcomes (Rescorla-Wagner + Pearce-Hall)
    "expectations": {"prior": 0.5, "gain_start": 0.3, "gain_min": 0.05, "gain_max": 0.6, "gamma": 0.3,
                     "recent_half_life_seconds": 3600},
    # information change: a question that was unknown becomes answerable (+), a contradiction is flagged (-)
    "information": {"gap_resolved": 1.0, "contradiction": -0.5, "operations_only": False},
    # information need: unknown questions not yet resolved (fading), conflicting facts, facts you were unsure of
    "information_need": {"gap_weight": 0.4, "gap_half_life_seconds": 86400, "conflict_weight": 0.4,
                         "unsure_weight": 0.2},
    # slowly shifting baselines (experimental "development"): on, capped, resettable
    "baselines": {"adapt": True, "rate": 0.02, "valence": [-0.25, 0.25], "activation": [0.0, 0.2]},
    # memory couplings (M1 encoding importance, M2 recall order, M3 recall -> state). M1 and M2 change what the brain
    # is shown, so by your rule they are on by default only if the memory benchmark stays within 1 point of state-off.
    # Measured 2026-10-02 (python -m internal_state.evaluate, 40 users, D-074): it did NOT. With M1, "What do I
    # like/love?" found the right memory 28.7% instead of 100%: the first memory of each kind is the most novel, so
    # its boost pushes later, equally relevant ones out of the 3 recalled. M2 changed nothing further. Both are
    # built, tested and switchable (python -m chat.server --state-memory-effects), and OFF by default. Tags (the
    # record of significance at encoding) and M3 (recall -> state, which never changes what the brain sees) stay on.
    "memory": {"tag_threshold": 0.4, "importance_boost": 0.2, "tag_half_life_seconds": 7 * 86400,
               "importance": False,
               "recall_order": False,
               "recall_weight": 0.1, "recall_cap": 0.1, "congruence": 0.5,
               "reinstate": True, "reinstate_rate": 0.2, "reinstate_activation": 0.2, "refractory_seconds": 1800},
}


def make_config(overrides=None):
    """The defaults, with any overrides merged in (nested dictionaries are merged, not replaced)."""
    cfg = copy.deepcopy(DEFAULTS)

    def merge(into, new):
        for key, value in (new or {}).items():
            if key not in into:
                raise KeyError(f"unknown internal-state setting {key!r}")
            if isinstance(value, dict):
                merge(into[key], value)
            else:
                into[key] = value
    merge(cfg, overrides)
    return cfg
