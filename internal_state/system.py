"""
The internal state in the chat: one call per message, AFTER the reply (and after the self-model has recorded the
episode in the event log), so it can never change the reply that was just written.

    after_episode(message event)   1 decay every number to the message's time
                                   2 M3: significant memories recalled into the prompt move the state toward
                                     their state at encoding (bounded, refractory)
                                   3 appraise the episode (appraisal.py) and let the expectations learn
                                   4 update the state (state.py); the baselines follow slowly (if on)
                                   5 M1: memories formed in a significant episode are tagged, and their
                                     importance is raised (logged in memory's own log, with the reason)
                                   6 record one history row: before, after, the items and every event id
    recall_bias()                  M2: before the NEXT message's recall, a small, capped bonus for tagged
                                   memories (only ordering memories that are already relevant)

Nothing here writes, chooses or edits text for the brain, and the brain never sees the state.
"""

import math
import time

from internal_state import state as dynamics
from internal_state.appraisal import Appraiser
from internal_state.config import make_config
from internal_state.expectations import Expectations, entropy
from internal_state.store import DEFAULT_DB, StateStore

DEFINITIONS = {
    "valence": "recent outcomes of the system's own verified checks relative to what was expected, plus information "
               "gained, minus contradictions added",
    "activation": "how much is changing: the strongest of surprise, novelty and conflict (saturating)",
    "surprise": "moving average of unsigned prediction errors",
    "novelty": "moving average of how new recent input was (count-based)",
    "uncertainty": "recency-weighted mean entropy of the expectations (1 = no experience yet)",
    "information_need": "pressure of open gaps: unknown questions not yet resolved (fading), conflicting facts and "
                        "facts you were unsure of",
    "learning_gain": "how fast the expectations learn (Pearce-Hall): rises after surprising outcomes",
}
RANGES = {**{d: list(r) for d, r in dynamics.RANGES.items()}, "uncertainty": [0, 1], "information_need": [0, 1],
          "learning_gain": [0.05, 0.6]}
DIMENSIONS = tuple(DEFINITIONS)
CHECK_LABELS = {"interpret": "question covered by the rules", "answerable": "question answerable",
                "express": "reply said the reasoning result", "reply": "reply complete (ended by itself, first try)",
                "consistent": "what you said matched what was stored"}


def rounded(d):
    return {k: round(v, 6) for k, v in d.items()}


def check_label(key):
    head, _, detail = key.partition(":")
    label = CHECK_LABELS.get(head, head)
    return label if head in ("interpret", "reply") else f"{label} ({detail.replace('_', ' ')})"


def causes(appraisal, effects=None):
    """The causes of one change, in plain words (no first person, no feelings): what the evidence says."""
    out = []
    if appraisal.get("valence_signal") is not None:
        out.append(f"this episode's valence signal: {appraisal['valence_signal']:+.2f} (valence moves halfway toward it)")
    for c in appraisal.get("checks", []):
        out.append(f"{check_label(c['key'])}: {'yes' if c['outcome'] else 'no'} (expected {c['expected']:.2f}, "
                   f"error {c['delta']:+.2f})")
    if appraisal.get("novelty") is not None:
        out.append(f"novelty {appraisal['novelty']:.2f} ({len(appraisal.get('novelty_items', []))} items)")
    for item in appraisal.get("information_items", []):
        out.append("an earlier unknown question became answerable" if item["what"] == "gap resolved"
                   else f"a contradiction was flagged (fact {item['fact']})")
    for item in appraisal.get("conflict_items", []):
        if item["what"] == "reasoning found conflicting values":
            out.append("reasoning found conflicting values")
    if appraisal.get("gaps_opened"):
        out.append(f"{len(appraisal['gaps_opened'])} question(s) answered \"unknown\" (a gap)")
    for r in (effects or {}).get("reinstated", []):
        out.append(f"recalled memory {r['memory']} (tag strength {r['strength']:.2f}) moved the state toward its "
                   "state at encoding")
    return out


class InternalState:

    def __init__(self, path=DEFAULT_DB, persist=True, config=None, clock=time.time):
        self.cfg = make_config(config)
        self.store = StateStore(path if persist else ":memory:")
        self.clock = clock
        self.expectations = Expectations(self.store, self.cfg)
        self.texts = {}                 # this run only: message event id -> your question (never stored)
        self.memory = self.learning = self.world = self.log = None
        self.style = "statements"

    @property
    def persistent(self):
        return self.store.persistent

    @property
    def path(self):
        return self.store.path

    def close(self):
        self.store.close()

    def attach(self, engine):
        """Read-only access to the parts it appraises (and the event log, its evidence)."""
        self.memory, self.learning, self.world = engine.memory, engine.learning, engine.world
        self.log = engine.self_model.log
        self.style = engine.answer_format
        engine.self_model.internal = self

    # ------------------------------------------------------------------ the state now
    def stored(self):
        values, baselines = dynamics.neutral()
        rows, t = self.store.current(), None
        for d in dynamics.DECAYING:
            if d in rows:
                values[d], baselines[d], t = rows[d]["value"], rows[d]["baseline"], rows[d]["t"]
        return values, baselines, t

    def values(self, now=None):
        """Every number at time `now` (decayed since the last episode), and the baselines."""
        now = self.clock() if now is None else now
        values, baselines, t = self.stored()
        if t is not None:
            values = dynamics.decay(values, baselines, now - t, self.cfg)
        return {**values, **self.computed(now)}, baselines

    def computed(self, now, new_facts=()):
        return {"uncertainty": self.expectations.uncertainty(now),
                "information_need": self.information_need(now, new_facts),
                "learning_gain": self.expectations.mean_gain(now)}

    def open_gaps(self, now):
        """Unknown questions not yet resolved: the same question asked again counts once (the latest asking)."""
        latest = {}
        for gap in self.store.gaps(open_only=True):
            e = self.log.get(gap["event_id"]) if self.log is not None else None
            key = (e["data"].get("path"), e["data"].get("missing")) if e else gap["event_id"]
            if key not in latest or gap["opened"] > latest[key]["opened"]:
                latest[key] = gap
        return list(latest.values())

    def information_need(self, now, new_facts=()):
        """new_facts: facts this episode created. Learning stored them before the appraisal, so the value BEFORE the
        episode is computed without them (and with the facts they replaced, which were current until then)."""
        c = self.cfg["information_need"]
        total = sum(c["gap_weight"] * 2 ** (-max(0.0, now - g["opened"]) / c["gap_half_life_seconds"])
                    for g in self.open_gaps(now))
        if self.learning is not None:
            from world.model import KNOWN_CONFIDENCE
            store = self.learning.store
            facts = store.current()
            if new_facts:
                new = set(new_facts)
                with store.lock:
                    replaced = [dict(r) for r in store.db.execute(
                        f"SELECT * FROM facts WHERE replaced_by IN ({','.join('?' * len(new))})", tuple(new)).fetchall()]
                facts = [f for f in facts if f["id"] not in new] + replaced
            for f in facts:
                if "conflicts with" in (f["note"] or ""):
                    total += c["conflict_weight"]
                elif f["confidence"] < KNOWN_CONFIDENCE:
                    total += c["unsure_weight"]
        return 1.0 - math.exp(-total)

    # ------------------------------------------------------------------ M2: recall order
    def strength(self, tag, now):
        """A tag fades (7-day half-life) unless you state the thing again; being recalled does not renew it."""
        return tag["salience"] * 2 ** (-max(0.0, now - tag["renewed"]) / self.cfg["memory"]["tag_half_life_seconds"])

    def recall_bias(self, now=None):
        """{memory id: b}: retrieval multiplies a tagged memory's score by (1 + b). It only ORDERS memories that
        already passed the relevance thresholds (memory/retrieval.py), so it can never bring in an unrelated one."""
        m = self.cfg["memory"]
        if not m["recall_order"]:
            return {}
        now = self.clock() if now is None else now
        valence = self.values(now)[0]["valence"]
        out = {}
        for memory_id, tag in self.store.tags().items():
            congruence = 1 - abs(valence - tag["valence"]) / 2
            b = min(m["recall_cap"], m["recall_weight"] * self.strength(tag, now) * (1 + m["congruence"] * congruence))
            if b > 1e-6:
                out[memory_id] = round(b, 6)
        return out

    # ------------------------------------------------------------------ one episode
    def after_episode(self, message_id, text=None):
        if self.log is None:
            return None
        appraiser = Appraiser(self.log, self.store, self.cfg, self.memory, self.learning, self.world, self.style,
                              self.texts)
        ep = appraiser.episode(message_id)
        now = ep["message"]["t"]
        if text is not None and ep["message"]["data"].get("question"):
            self.texts[message_id] = text
        items = appraiser.appraise(ep)
        values, baselines, t = self.stored()
        if t is not None:
            values = dynamics.decay(values, baselines, now - t, self.cfg)
        new_facts = [i for k in ("learned", "corrected", "changed") for i in
                     ((ep["knowledge"] or {}).get("refs", {}).get(k, []))]
        before = {**values, **self.computed(now, new_facts)}

        reinstated = []                                  # M3: recall -> state (refractory; never renews the tag)
        if self.cfg["memory"]["reinstate"] and items["recalled"]:
            tags = self.store.tags()
            for memory_id in items["recalled"]:
                tag = tags.get(memory_id)
                if tag is None:
                    continue
                if tag["last_reinstated"] is not None and now - tag["last_reinstated"] < \
                        self.cfg["memory"]["refractory_seconds"]:
                    continue
                strength = self.strength(tag, now)
                if strength < 0.01:
                    continue
                values = dynamics.reinstate(values, tag, strength, self.cfg)
                tag["last_reinstated"] = now
                self.store.save_tag(tag)
                reinstated.append({"memory": memory_id, "strength": round(strength, 6),
                                   "valence_at_encoding": tag["valence"], "event": ep["recall"]["id"]})

        checks = []                                      # each check learns its expectation, in order
        for c in items["checks"]:
            learned = self.expectations.learn(c["key"], c["kind"], c["outcome"], now)
            checks.append({**learned, "event": c["event"], **({"fact": c["fact"]} if "fact" in c else {})})
        novelty = (sum(i["novelty"] for i in items["novelty_items"]) / len(items["novelty_items"])
                   if items["novelty_items"] else None)
        information = dynamics.clip(sum(i["value"] for i in items["information_items"]), -1.0, 1.0)
        conflict = 1.0 if items["conflict_items"] else 0.0
        largest = max((abs(c["delta"]) for c in checks), default=0.0)
        numbers = {"operation_errors": [c["delta"] for c in checks if c["kind"] == "operation"],
                   "largest_error": largest, "has_checks": bool(checks), "novelty": novelty,
                   "information": information, "conflict": conflict}
        signal = dynamics.valence_signal(numbers, self.cfg)
        values = dynamics.update(values, numbers, self.cfg)
        new_baselines = dynamics.adapt(baselines, values, self.cfg)
        for gap in items["gaps_opened"]:
            self.store.open_gap(gap, now)
        for item in items["information_items"]:
            if item["what"] == "gap resolved":
                self.store.resolve_gap(item["gap"], item["event"], now)
        salience = max(largest, novelty or 0.0, conflict, abs(information))
        effects = self.encode(items["formed"], ep["formed"], salience, values, now)
        effects["reinstated"] = reinstated
        self.store.save_current(values, new_baselines, now)
        after = {**values, **self.computed(now)}
        appraisal = {"checks": checks, "novelty": None if novelty is None else round(novelty, 6),
                     "novelty_items": items["novelty_items"], "information": round(information, 6),
                     "information_items": items["information_items"], "conflict": conflict,
                     "conflict_items": items["conflict_items"], "gaps_opened": items["gaps_opened"],
                     "largest_error": round(largest, 6), "salience": round(salience, 6),
                     "valence_signal": None if signal is None else round(signal, 6)}
        row = self.store.add_history("episode", now, episode=message_id, events=items["events"], appraisal=appraisal,
                                     before=rounded(before), after=rounded(after),
                                     baselines={"before": rounded(baselines), "after": rounded(new_baselines)},
                                     effects=effects)
        return {"enabled": True, "history_row": row, "before": rounded(before), "after": rounded(after),
                "change": {d: round(after[d] - before[d], 6) for d in DIMENSIONS},
                "baselines": rounded(new_baselines), "appraisal": appraisal, "effects": effects,
                "causes": causes(appraisal, effects), "evidence": items["events"], "shown_to_brain": False}

    def encode(self, formed, formed_event, salience, values, now):
        """M1: memories formed during a significant episode (salience >= the threshold) are tagged, and their
        importance is raised by boost * salience (memory's own log records why). Below the threshold: nothing,
        so most memories carry no tag. Saying the same thing again renews an existing tag (new evidence)."""
        m = self.cfg["memory"]
        out = {"tagged": [], "renewed": [], "salience": round(salience, 6)}
        if self.memory is None or not formed:
            return out
        tags = self.store.tags()
        for memory_id in formed:
            if self.memory.store.get(memory_id) is None:
                continue
            tag = tags.get(memory_id)
            if tag is None:
                if salience < m["tag_threshold"]:
                    continue
                boost = m["importance_boost"] * salience if m["importance"] else 0.0
                if boost:
                    self.memory.store.raise_importance(memory_id, boost, f"significant at encoding: salience "
                                                       f"{salience:.2f} (internal state, event {formed_event['id']})")
                tag = {"memory_id": memory_id, "salience": salience, "valence": values["valence"],
                       "activation": values["activation"], "boost": boost, "event_id": formed_event["id"], "t": now,
                       "renewed": now, "last_reinstated": None}
                self.store.save_tag(tag)
                out["tagged"].append({"memory": memory_id, "salience": round(salience, 6),
                                      "importance_added": round(boost, 6), "event": formed_event["id"]})
            else:
                extra = m["importance_boost"] * (salience - tag["salience"]) if m["importance"] and \
                    salience > tag["salience"] else 0.0
                if extra > 0:
                    self.memory.store.raise_importance(memory_id, extra, f"significant again: salience {salience:.2f} "
                                                       f"(internal state, event {formed_event['id']})")
                tag.update({"renewed": now, "boost": tag["boost"] + extra})
                if salience > tag["salience"]:
                    tag.update({"salience": salience, "valence": values["valence"], "activation": values["activation"],
                                "event_id": formed_event["id"]})
                self.store.save_tag(tag)
                out["renewed"].append({"memory": memory_id, "importance_added": round(extra, 6),
                                       "event": formed_event["id"]})
        return out

    # ------------------------------------------------------------------ your controls
    def reset(self, what="baselines"):
        """"baselines": the slow part back to 0 (the current values are kept). "everything": back to the start."""
        now = self.clock()
        values, baselines = self.values(now)
        if what == "baselines":
            decaying = {d: values[d] for d in dynamics.DECAYING}
            self.store.save_current(decaying, dynamics.neutral()[1], now)
            self.store.add_history("reset baselines", now, baselines={"before": rounded(baselines),
                                                                       "after": rounded(dynamics.neutral()[1])})
        elif what == "everything":
            self.store.clear()
            self.expectations = Expectations(self.store, self.cfg)
            self.store.add_history("reset everything", now)
        else:
            raise ValueError("reset what? 'baselines' or 'everything'")
        return self.values(now)

    # ------------------------------------------------------------------ views
    def view(self, limit=30):
        now = self.clock()
        values, baselines = self.values(now)
        history = self.store.history(limit=limit)
        chart = self.store.history(limit=80)[::-1]
        tags = []
        for memory_id, tag in sorted(self.store.tags().items()):
            memory = self.memory.store.get(memory_id) if self.memory is not None else None
            if memory is None:
                continue
            tags.append({"memory": memory_id, "summary": memory.summary, "salience": round(tag["salience"], 3),
                         "strength_now": round(self.strength(tag, now), 3), "valence_at_encoding": round(tag["valence"], 3),
                         "importance_added": round(tag["boost"], 3), "event": tag["event_id"]})
        gaps = []
        for gap in self.open_gaps(now):
            e = self.log.get(gap["event_id"]) if self.log is not None else None
            gaps.append({"event": gap["event_id"], "missing": (e["data"].get("missing") or e["data"].get("path"))
                         if e else "?", "opened": gap["opened"]})
        return {"enabled": True, "now": rounded(values), "baselines": rounded(baselines), "definitions": DEFINITIONS,
                "ranges": RANGES, "half_life_seconds": self.cfg["half_life_seconds"],
                "switches": {"baseline_adaptation": self.cfg["baselines"]["adapt"],
                             "importance_at_encoding": self.cfg["memory"]["importance"],
                             "recall_order": self.cfg["memory"]["recall_order"],
                             "recall_moves_state": self.cfg["memory"]["reinstate"]},
                "history": [{"id": h["id"], "t": h["t"], "kind": h["kind"], "episode": h["episode"],
                             "events": h["events"], "causes": causes(h["appraisal"], h["effects"]),
                             "change": {d: round(h["after"][d] - h["before"][d], 6) for d in DIMENSIONS
                                        if d in h["after"] and d in h["before"]},
                             "baselines": h["baselines"], "effects": h["effects"]} for h in history],
                "chart": [{"t": h["t"], **{d: h["after"].get(d) for d in DIMENSIONS}} for h in chart if h["after"]],
                "expectations": [{**r, "uncertainty": round(entropy(r["p"]), 6), "label": check_label(r["key"])}
                                 for r in self.store.expectations()],
                "tags": tags, "gaps": gaps,
                "record": {"database": self.path.name if self.persistent else "RAM only (memory is off)",
                           "history_rows": len(self.store.history())}}

    def self_report(self):
        """The internal state as the self-model shows it: numbers with their evidence, never feelings."""
        from self_model.model import clock, item
        now = self.clock()
        values, baselines = self.values(now)
        history = self.store.history(limit=8)
        last = next((h for h in history if h["kind"] == "episode"), None)
        anchor = [{"store": "internal_state"}] + ([{"event": last["episode"]}] if last else [])
        out = {"current": [item(f"{d.replace('_', ' ')}: {values[d]:+.2f} ({DEFINITIONS[d]})", *anchor, dimension=d,
                                value=round(values[d], 6)) for d in DIMENSIONS],
               "changes": [], "baselines": [], "expectations": []}
        for h in history:
            if h["kind"] != "episode":
                out["changes"].append(item(f"{clock(h['t'])}: {h['kind']} (by you)", {"store": "internal_state"}))
                continue
            change = ", ".join(f"{d.replace('_', ' ')} {h['after'][d] - h['before'][d]:+.2f}" for d in DIMENSIONS
                               if abs(h["after"][d] - h["before"][d]) >= 0.005) or "no change above 0.005"
            out["changes"].append(item(f"{clock(h['t'])}: {change}; caused by: " + "; ".join(
                causes(h["appraisal"], h["effects"])[:4]), *[{"event": e} for e in h["events"][:4]]))
        for d in dynamics.ADAPTING:
            out["baselines"].append(item(f"baseline of {d}: {baselines[d]:+.2f} (starts at 0; follows the state slowly, "
                                         f"capped at {self.cfg['baselines'][d]})", {"store": "internal_state"}))
        for r in self.store.expectations():
            out["expectations"].append(item(f"{check_label(r['key'])}: expected {r['p']:.2f} after {r['n']} outcomes, "
                                            f"learning gain {r['gain']:.2f}, uncertainty {entropy(r['p']):.2f}",
                                            {"store": "internal_state"}))
        return out
