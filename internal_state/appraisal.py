"""
Appraisal: one episode (your message and everything it caused, from the event log) -> the content-free items
that change the state (docs/07_INTERNAL_STATE.md, section 5). Every item carries the id of the event it came from.

Only these event kinds are ever read (ELIGIBLE). The event log holds no words of any reply, so generated text
cannot reach the state; only the checks the system itself measured about a reply are used.

    checks       outcomes the expectations predict (expectations.py):
                   interpret:question          your question was covered by the rules (reasoning or recorded-state)
                   answerable:<question type>  reasoning found an answer (vs "unknown")         information
                   express:<status>            the reply said the reasoning result (the expression check)
                   reply:complete              the reply ended by itself on the first attempt
                   consistent:<category>       what you said matched what was stored (confirmed) or not
                                               (corrected, or a flagged contradiction)          information
    novelty      how new the input was: about 1 / sqrt(1 + times seen) for each thing, relation and question
                 form involved (count-based novelty, as in Bellemare et al. 2016)
    information  +1 for each earlier "unknown" question that has become answerable, -0.5 for each contradiction
                 flagged by learning (clipped to -1 .. 1)
    conflict     1 when reasoning found conflicting values or learning flagged a contradiction
    recalled     memories put in the brain's prompt (for the recall -> state effect)
    formed       memories formed (for the encoding effect)

NOT appraised: forgetting, your panel controls, measurements, the program starting or stopping. Treating
deletion as bad would be a built-in self-preservation value.
"""

import math
import re

ELIGIBLE = {"message", "self question", "reasoning", "recall", "reply", "memory formed", "knowledge changed"}
CHILD_KEYS = {"self question": "self", "reasoning": "reasoning", "recall": "recall", "reply": "reply",
              "memory formed": "formed", "knowledge changed": "knowledge"}


def seen_novelty(count):
    return 1.0 / math.sqrt(1.0 + count)


def question_type(data):
    """The reasoning's question type without details: "lookup (temperature)" -> "lookup"."""
    return re.sub(r"\s*\(.*?\)", "", data.get("type", "")).strip().lower()[:60] or "question"


class Appraiser:

    def __init__(self, log, store, cfg, memory=None, learning=None, world=None, style="statements", texts=None):
        self.log, self.store, self.cfg = log, store, cfg
        self.memory, self.learning, self.world = memory, learning, world
        self.style = style
        self.texts = texts if texts is not None else {}

    def episode(self, message_id):
        """The message event and the eligible events it caused."""
        msg = self.log.get(message_id)
        if msg is None or msg["kind"] != "message":
            raise ValueError(f"event {message_id} is not a message")
        ep = {"message": msg, **{k: None for k in CHILD_KEYS.values()}}
        for e in self.log.events(cause=message_id):
            if e["kind"] in CHILD_KEYS:
                ep[CHILD_KEYS[e["kind"]]] = e
        return ep

    def appraise(self, ep):
        """The items of one episode. Expectations are NOT consulted here: the system does that, in order."""
        msg, reasoning, self_q, reply, knowledge = ep["message"], ep["reasoning"], ep["self"], ep["reply"], ep["knowledge"]
        out = {"checks": [], "novelty_items": [], "information_items": [], "conflict_items": [], "gaps_opened": [],
               "recalled": [], "formed": [], "events": [msg["id"]] + [ep[k]["id"] for k in CHILD_KEYS.values() if ep[k]]}
        s = self_q["data"] if self_q else {}
        if msg["data"].get("question") and not s.get("outside") and (reasoning is not None or self_q is not None):
            covered = bool(self_q) or bool(reasoning and reasoning["data"].get("understood"))
            out["checks"].append({"key": "interpret:question", "kind": "operation", "outcome": int(covered),
                                  "event": (self_q or reasoning)["id"]})
        if reasoning is not None and reasoning["data"].get("understood"):
            d = reasoning["data"]
            out["checks"].append({"key": "answerable:" + question_type(d), "kind": "information",
                                  "outcome": int(d.get("status") != "unknown"), "event": reasoning["id"]})
            if d.get("status") == "unknown":
                out["gaps_opened"].append(reasoning["id"])
            if d.get("status") == "conflicting":
                out["conflict_items"].append({"what": "reasoning found conflicting values", "event": reasoning["id"]})
            if d.get("form"):
                earlier = sum(1 for e in self.log.events(kinds=["reasoning"])
                              if e["id"] < reasoning["id"] and e["data"].get("form") == d["form"])
                out["novelty_items"].append({"of": "question form", "ref": {"form": d["form"]}, "seen": earlier,
                                             "novelty": round(seen_novelty(earlier), 6), "event": reasoning["id"]})
        if reply is not None:
            d = reply["data"]
            if d.get("said") is not None:
                key = "express:" + d.get("status", "known") + (", yes/no" if d.get("yes_no") else "")
                out["checks"].append({"key": key, "kind": "operation", "outcome": int(bool(d["said"])),
                                      "event": reply["id"]})
            complete = d.get("ended_by_itself", True) and d.get("attempts", 1) == 1
            out["checks"].append({"key": "reply:complete", "kind": "operation", "outcome": int(complete),
                                  "event": reply["id"]})
        if knowledge is not None and self.learning is not None:
            self.knowledge_items(knowledge, out)
        elif ep["formed"] is not None:
            self.memory_novelty(ep["formed"], out)
        if ep["recall"] is not None:
            out["recalled"] = list(ep["recall"]["refs"].get("memories", []))
        if ep["formed"] is not None:
            out["formed"] = list(ep["formed"]["refs"].get("memories", []))
        return out

    # ------------------------------------------------------------------ learning
    def knowledge_items(self, knowledge, out):
        store = self.learning.store
        refs = knowledge["refs"]
        changed = set(refs.get("changed", []))
        mine = set(refs.get("learned", [])) | set(refs.get("corrected", [])) | changed
        for kind, outcome in (("confirmed", 1), ("corrected", 0), ("changed", 0)):
            for fact_id in refs.get(kind, []):
                f = store.fact(fact_id)
                if f is None:
                    continue
                out["checks"].append({"key": "consistent:" + f["category"], "kind": "information", "outcome": outcome,
                                      "event": knowledge["id"], "fact": fact_id})
        for fact_id in sorted(changed):
            out["conflict_items"].append({"what": "a contradiction was flagged", "fact": fact_id, "event": knowledge["id"]})
            out["information_items"].append({"what": "contradiction", "fact": fact_id, "event": knowledge["id"],
                                             "value": self.cfg["information"]["contradiction"]})
        placeholders = ",".join("?" * len(mine)) or "NULL"
        for fact_id in refs.get("learned", []):
            f = store.fact(fact_id)
            if f is None:
                continue
            with store.lock:
                items = [("thing", {"entity": f["subject_id"]}, store.db.execute(
                    f"SELECT COUNT(*) FROM facts WHERE (subject_id = ? OR value_id = ?) AND id NOT IN ({placeholders})",
                    (f["subject_id"], f["subject_id"], *mine)).fetchone()[0]),
                         ("relation", {"fact": fact_id}, store.db.execute(
                    f"SELECT COUNT(*) FROM facts WHERE relation = ? AND id NOT IN ({placeholders})",
                    (f["relation"], *mine)).fetchone()[0])]
                if f["value_id"]:
                    items.append(("thing", {"entity": f["value_id"]}, store.db.execute(
                        f"SELECT COUNT(*) FROM facts WHERE (subject_id = ? OR value_id = ?) AND id NOT IN ({placeholders})",
                        (f["value_id"], f["value_id"], *mine)).fetchone()[0]))
            for what, ref, count in items:
                out["novelty_items"].append({"of": what, "ref": ref, "seen": count,
                                             "novelty": round(seen_novelty(count), 6), "event": knowledge["id"]})
        if mine and self.world is not None:
            self.resolved_gaps(knowledge, out)

    def resolved_gaps(self, knowledge, out):
        """Earlier "unknown" questions that the new facts now answer (asked again: reasoning only, nothing kept)."""
        for gap in self.store.gaps(open_only=True):
            reasoning = self.log.get(gap["event_id"])
            question = self.question_text(reasoning["cause"]) if reasoning else None
            if question is None:
                continue
            thought = self.world.think(question, self.style, keep=False)
            if thought.answered and thought.frame.status != "unknown":
                out["information_items"].append({"what": "gap resolved", "gap": gap["event_id"], "event": knowledge["id"],
                                                 "value": self.cfg["information"]["gap_resolved"]})

    def question_text(self, message_id):
        """The words of a question: from memory.db (where they live), or this run's own record when memory is off."""
        if message_id in self.texts:
            return self.texts[message_id]
        msg = self.log.get(message_id)
        turn = msg["refs"].get("turn") if msg else None
        if turn and self.memory is not None:
            row = self.memory.store.turn(turn)
            return row["text"] if row else None
        return None

    # ------------------------------------------------------------------ memory (when learning is off)
    def memory_novelty(self, formed, out):
        earlier = {m for e in self.log.events(kinds=["memory formed"]) if e["id"] < formed["id"]
                   for m in e["refs"].get("memories", [])}
        for memory_id in formed["refs"].get("memories", []):
            m = self.memory.store.get(memory_id) if self.memory is not None else None
            if m is None:
                continue
            if memory_id in earlier:                  # the same memory again (you said it again): nothing new
                count, novelty = 1, 0.0
            else:
                count = sum(1 for x in self.memory.store.active() if x.type == m.type and x.id < memory_id)
                novelty = seen_novelty(count)
            out["novelty_items"].append({"of": "memory type", "ref": {"memory": memory_id}, "seen": count,
                                         "novelty": round(novelty, 6), "event": formed["id"]})
