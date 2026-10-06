"""
The learning system in the chat: the same pattern as memory/system.py.

    before the reply   (D-079) a statement is read on a throwaway copy (preview): what it would teach is shown to
                       the brain as the status line; nothing is stored then.
                       "forget ..." removes matching knowledge; then the current facts relevant to your
                       message are recalled (never one the message itself contradicts: "I don't have a dog"). Each is shown to the brain as an EARLIER TURN: the sentence you
                       taught it with, followed by the reply the brain gave then. Nothing else is added.
    after the reply    YOUR message (never the reply) goes through the learning pipeline (learner.py).

The knowledge view on the chat page lists what was learned (grouped, with sources, confidence, dates,
confirmations, earlier values and connections) and the learning experience log.
"""

import re

from learning.knowledge import CATEGORIES, DEFAULT_DB, KnowledgeStore
from learning.learner import Learner


class LearningSystem:

    def __init__(self, path=DEFAULT_DB):
        self.store = KnowledgeStore(path)
        self.learner = Learner(self.store)

    @property
    def path(self):
        return self.store.path

    def close(self):
        self.store.close()

    # -------------------------------------------------------------- with the chat
    def before_reply(self, message):
        from learning.extract import read
        reading = read(message)
        forgotten = []
        if reading.forget:
            matches = self.learner.recall(reading.forget, k=10)
            best = matches[0].relevance if matches else 0.0
            for r in matches:
                if r.relevance >= 0.5 and r.relevance >= 0.8 * best:
                    forgotten.append(self.learner.public(r.fact))
                    self.store.remove(r.fact["id"], why="you asked to forget it")
        contradicted = set()                 # "I don't have a dog": the facts it contradicts are not shown (D-075)
        for o in reading.observations:
            if o.negated:
                contradicted |= {f["id"] for f in self.learner.contradicted_by(o, self.learner.known_subject(o))}
        recalled = [r for r in self.learner.recall(message) if r.fact["id"] not in contradicted]
        return {"recalled": recalled, "forgotten": forgotten}

    def preview(self, message):
        """What learning WOULD do with your message, worked out on a throwaway in-memory copy of the knowledge store
        (D-079). Returns (what would happen, the copy); the caller closes the copy. Nothing is stored: after_reply
        stores it, after the reply, exactly as before."""
        copy = self.store.copy_in_memory()
        return Learner(copy).learn(message), copy

    def after_reply(self, message, reply, turn_id=None):
        """Learn from YOUR message. The reply is stored only as the turn in between, never as knowledge."""
        return self.learner.learn(message, turn_id=turn_id, reply_after=reply)

    # ------------------------------------------------------------------ your view
    def view(self):
        """Everything learned, grouped by category and entity, with connections and the experience log."""
        store = self.store
        groups = []
        for category, label in CATEGORIES.items():
            facts = store.current(category=category)
            by_entity = {}
            for f in facts:
                by_entity.setdefault(f["subject_id"], []).append(self.learner.public(f))
            entities = []
            for entity_id, items in by_entity.items():
                names = store.aliases(entity_id)
                connections = sorted({store.entity_label(f["value_id"]) for f in facts
                                      if f["subject_id"] == entity_id and f["value_id"]} |
                                     {store.entity_label(f["subject_id"]) for f in store.current()
                                      if f["value_id"] == entity_id})
                entities.append({"id": entity_id, "label": store.entity_label(entity_id),
                                 "names": [a["alias"] for a in names if a["current"]],
                                 "former_names": [a["alias"] for a in names if not a["current"]],
                                 "inferred_names": [a["alias"] for a in names if a["how"] == "inferred"],
                                 "facts": items, "connections": connections})
            groups.append({"category": category, "label": label, "entities": entities})
        experience = [{**e, "fact": store.describe(e["fact_id"]) if e["fact_id"] and store.fact(e["fact_id"]) else ""}
                      for e in store.experience(30)]
        return {"groups": groups, "experience": experience, "counts": store.counts(), "database": self.path.name}

    def why(self, fact_id):
        f = self.store.fact(fact_id)
        if f is None:
            return None
        return {**self.learner.public(f), "history": [self.learner.public(h) for h in self.store.history(fact_id)],
                "reply_at_the_time": f["reply_after"]}

    def remove(self, fact_id):
        return self.store.remove(fact_id)

    def clear(self):
        return self.store.clear()

    # ----------------------------------------------------- approved general knowledge
    def import_file(self, path):
        """General knowledge from a text file YOU approved: one statement per line (# lines are notes).
        Stored in the 'general' category, source = the file. Nothing is ever imported automatically."""
        from learning.extract import read
        from pathlib import Path
        added, skipped = 0, []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            observations = [o for o in read("Learn that " + re.sub(r"^learn that\s+", "", line, flags=re.I)).observations
                            if o.category in ("taught", "skill")]
            if not observations:
                skipped.append(line)
            for o in observations:
                subject_id = self.learner.entity_for(o, [])
                existing = [f for f in self.store.current(subject_id, o.relation) if f["value"].lower() == o.value.lower()]
                if existing:
                    self.store.reinforce(existing[0]["id"])
                    continue
                category = "skill" if o.category == "skill" else "general"
                self.store.add_fact(category, subject_id, o.relation, o.value, o.text, "approved_source", 0.85,
                                    f"from the approved file {Path(path).name}", source_detail=Path(path).name)
                added += 1
        return added, skipped
