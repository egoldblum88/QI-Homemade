"""
Expectations: what "expected" means for the internal state.

Small learned predictors of binary outcomes, one per context (docs/07 section 4.2). Each keeps p (the predicted
probability that the outcome happens, 0.5 before any experience), n (outcomes seen) and its learning gain:

    delta = outcome - p                 prediction error (signed); |delta| is the surprise
    p     <- p + gain * delta           Rescorla-Wagner (1972)
    gain  <- clip((1 - gamma) * gain + gamma * |delta|, min, max)        Pearce-Hall (1980) "associability"

so a surprising outcome makes the next update larger (more learning attention), and predictable outcomes make it
smaller. Two kinds of outcome:
    operation     the system's own components working as built, checked by an existing measurement:
                  interpret:question, express:<status>, reply:complete          -> signed, into valence
    information   how the world turned out, with no better or worse side:
                  answerable:<question type>, consistent:<knowledge category>   -> surprise and uncertainty only
"""

import math
import re

KEY = re.compile(r"^(?:interpret:question|reply:complete|express:[a-z]+(?:, yes/no)?|answerable:[a-z0-9 /-]+|"
                 r"consistent:[a-z_]+)$")
KINDS = {"operation", "information"}


def entropy(p):
    """Bernoulli entropy in bits: 1 when p = 0.5 (no idea), 0 when the outcome is certain."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


class Expectations:

    def __init__(self, store, cfg):
        self.store, self.cfg = store, cfg["expectations"]
        self.rows = {r["key"]: r for r in store.expectations()}

    def predict(self, key):
        row = self.rows.get(key)
        return (row["p"], row["gain"]) if row else (self.cfg["prior"], self.cfg["gain_start"])

    def learn(self, key, kind, outcome, t):
        """One outcome: returns what was expected, the error and the gain before and after."""
        if not KEY.match(key) or kind not in KINDS:
            raise ValueError(f"not an expectation the internal state keeps: {key!r} ({kind})")
        p, gain = self.predict(key)
        delta = float(outcome) - p
        new_p = min(1.0, max(0.0, p + gain * delta))
        c = self.cfg
        new_gain = min(c["gain_max"], max(c["gain_min"], (1 - c["gamma"]) * gain + c["gamma"] * abs(delta)))
        row = {"key": key, "kind": kind, "p": new_p, "n": (self.rows[key]["n"] if key in self.rows else 0) + 1,
               "gain": new_gain, "last_used": t}
        self.rows[key] = row
        self.store.save_expectation(row)
        return {"key": key, "kind": kind, "outcome": int(outcome), "expected": round(p, 6), "delta": round(delta, 6),
                "gain_before": round(gain, 6), "gain_after": round(new_gain, 6)}

    def weights(self, now):
        half = self.cfg["recent_half_life_seconds"]
        return {k: 2 ** (-max(0.0, now - r["last_used"]) / half) for k, r in self.rows.items()}

    def uncertainty(self, now):
        """Recency-weighted mean entropy of the expectations: 1.0 before any experience."""
        if not self.rows:
            return 1.0
        w = self.weights(now)
        total = sum(w.values())
        if total < 1e-12:                          # all very old: plain mean
            return sum(entropy(r["p"]) for r in self.rows.values()) / len(self.rows)
        return sum(w[k] * entropy(r["p"]) for k, r in self.rows.items()) / total

    def mean_gain(self, now):
        """Recency-weighted mean learning gain (the start value before any experience)."""
        if not self.rows:
            return self.cfg["gain_start"]
        w = self.weights(now)
        total = sum(w.values())
        if total < 1e-12:
            return sum(r["gain"] for r in self.rows.values()) / len(self.rows)
        return sum(w[k] * r["gain"] for k, r in self.rows.items()) / total
