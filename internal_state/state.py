"""
The state's dynamics: pure functions, so they can be tested exactly (docs/07_INTERNAL_STATE.md, sections 4-5).

The decaying numbers and their ranges:
    valence     -1 .. 1   recent outcomes of the system's own verified checks relative to what was expected,
                          plus information gained, minus contradictions added
    activation   0 .. 1   how much is changing: the strongest of surprise, novelty and conflict, saturating
    surprise     0 .. 1   moving average of unsigned prediction errors
    novelty      0 .. 1   moving average of how new recent input was

Between episodes each one returns toward its baseline with its own half-life:
    x <- baseline + (x - baseline) * 2 ** (-elapsed / half_life)

One episode (gains in config.py):
    signal     =  clip(mean(signed operation errors) + w_i * information, -1, 1)
    valence    <- valence + k_v * (signal - valence)          (a moving average: it stays within its inputs' range)
    surprise   <- surprise + l_s * (largest |error| - surprise)                  (if the episode had any check)
    novelty    <- novelty + l_n * (novelty of the input - novelty)                (if it had new input)
    activation <- activation + k_a * max(|error|, novelty, conflict) * (1 - activation)     (saturating)
and, if on, the baselines slowly follow the state:
    baseline   <- clip(baseline + rate * (x - baseline))          (valence and activation only; capped)
"""

DECAYING = ("valence", "activation", "surprise", "novelty")
RANGES = {"valence": (-1.0, 1.0), "activation": (0.0, 1.0), "surprise": (0.0, 1.0), "novelty": (0.0, 1.0)}
ADAPTING = ("valence", "activation")


def clip(x, low, high):
    return min(high, max(low, x))


def neutral():
    """The state before any experience: every decaying number and every baseline at 0."""
    return {d: 0.0 for d in DECAYING}, {d: 0.0 for d in DECAYING}


def decay(values, baselines, elapsed, cfg):
    """Each number moves back toward its baseline: half of the difference per half-life."""
    out = {}
    for d in DECAYING:
        half = cfg["half_life_seconds"][d]
        factor = 2 ** (-max(0.0, elapsed) / half) if half > 0 else 0.0
        out[d] = baselines[d] + (values[d] - baselines[d]) * factor
    return out


def valence_signal(appraisal, cfg):
    """mean(signed operation errors) + w_i * information, clipped; None when the episode has neither."""
    deltas = appraisal["operation_errors"]
    info = 0.0 if cfg["information"]["operations_only"] else appraisal["information"]
    if not deltas and not info:
        return None
    return clip((sum(deltas) / len(deltas) if deltas else 0.0) + cfg["gains"]["information"] * info, *RANGES["valence"])


def update(values, appraisal, cfg):
    """One appraised episode. appraisal: the numbers from appraisal.py (operation errors, largest |error|,
    novelty or None, information change, conflict)."""
    g = cfg["gains"]
    out = dict(values)
    signal = valence_signal(appraisal, cfg)
    if signal is not None:
        out["valence"] = clip(values["valence"] + g["valence"] * (signal - values["valence"]), *RANGES["valence"])
    if appraisal["has_checks"]:
        out["surprise"] = clip(values["surprise"] + g["surprise"] * (appraisal["largest_error"] - values["surprise"]),
                               *RANGES["surprise"])
    if appraisal["novelty"] is not None:
        out["novelty"] = clip(values["novelty"] + g["novelty"] * (appraisal["novelty"] - values["novelty"]),
                              *RANGES["novelty"])
    drive = max(appraisal["largest_error"], appraisal["novelty"] or 0.0, appraisal["conflict"])
    out["activation"] = clip(values["activation"] + g["activation"] * drive * (1 - values["activation"]),
                             *RANGES["activation"])
    return out


def reinstate(values, tag, strength, cfg):
    """A significant memory was recalled into the prompt: the state moves part of the way toward the state at
    encoding (never past it), in proportion to the tag's current strength."""
    m = cfg["memory"]
    out = dict(values)
    share = m["reinstate_rate"] * strength
    out["valence"] = clip(values["valence"] + share * (tag["valence"] - values["valence"]), *RANGES["valence"])
    lift = m["reinstate_activation"] * strength * tag["activation"]
    out["activation"] = clip(values["activation"] + lift * (1 - values["activation"]), *RANGES["activation"])
    return out


def adapt(baselines, values, cfg):
    """The slow part (experimental, capped): baselines follow the state a little after every episode."""
    b = cfg["baselines"]
    if not b["adapt"]:
        return dict(baselines)
    out = dict(baselines)
    for d in ADAPTING:
        out[d] = clip(baselines[d] + b["rate"] * (values[d] - baselines[d]), *b[d])
    return out
