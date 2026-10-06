"""
The learning pipeline, and recall.

LEARN, for each thing found in one of YOUR messages (learning/extract.py):
    CONNECT     who or what is it about? "my dog" -> your dog;  "Max" -> whatever is called Max (your dog,
                if you said so). A link the system makes itself is marked "inferred".
    NOVELTY     already known, same value     -> REINFORCE: confirmations + 1, last confirmed now
                known, different value         -> one value at a time (a name, a capital): the new one becomes
                                                  current and the old one is kept as history.
                                                  "Correction: ..." -> corrected; a world fact changed without a
                                                  correction is flagged as a conflict and gets lower confidence
                                                  "I like X" after "I don't like X" (or the reverse) -> replaced
                new                            -> STORE
    IMPORTANCE  only what the reading rules accept is learned at all (taught > stated; passing remarks never)
    NEGATION    "My dog's name is not Max", "I don't have a dog" (D-075): stored as a NEGATED fact (you said it
                is not so). Current facts it contradicts are superseded and kept as history: the same value
                ("not Max" when the name was Max), or, when you say you do not have a <role>, everything known
                about your <role>. A later positive statement ("My dog's name is Rex") supersedes a negation it
                contradicts in the same way. You are the authority about yourself (corrected); a world fact you
                taught is flagged as a conflict instead (as for any changed world fact).

RECALL, for a question:
    entities named in the question (and everything connected to them), plus word overlap; CURRENT facts only
    (measured: when old and new values were both in its prompt, the brain picked the old one). At most 3.
    A question about the system ("what is your name?") recalls nothing about you.
"""

import math
import re
import time
from dataclasses import dataclass

from learning import extract
from learning.knowledge import KnowledgeStore
from memory.retrieval import FAVORITE_ASKED, favorite_category, subject_of, terms
from memory.writer import RELATIONS

PET_ROLES = {"dog", "cat", "puppy", "kitten", "bird", "parrot", "fish", "hamster", "rabbit", "horse", "turtle", "snake",
             "lizard", "pony", "goldfish", "mouse", "pet"}

MAX_RECALLED = 3
MIN_RELEVANCE = 0.35
RELATIVE_CUTOFF = 0.6
CONFLICT_CONFIDENCE = 0.6           # a world fact changed without a correction
# questions asking for EVERYTHING about something ("what is the capital of X" asks for one thing only)
ABOUT = re.compile(r"\b(?:what do you know about|what else do you know about|tell me about|tell me more about|"
                   r"who is|describe)\s+(?:my\s+)?(?P<x>[\w' -]+?)\s*\??$", re.I)


@dataclass
class Recalled:
    fact: dict
    relevance: float
    score: float
    parts: dict
    label: str

    @property
    def text(self):
        return self.fact["text"]


POSSESSION = {"has", "have", "owns", "own"}
DETERMINERS = re.compile(r"^(?:a|an|any|one|the|some)\s+", re.I)


def possessed_role(value):
    """ "a dog" -> "dog", "any dogs" -> "dog": the role a possession is about (None when it is not a role)."""
    noun = DETERMINERS.sub("", (value or "").strip().lower())
    if noun in extract.ROLE_WORDS:
        return noun
    if noun.endswith("s") and noun[:-1] in extract.ROLE_WORDS:
        return noun[:-1]
    return None


def same_relation(a, b):
    from world import words
    return words.key(a) == words.key(b)


def norm(value):
    return re.sub(r"[^a-z0-9 ]", "", value.lower()).strip()


class Learner:

    def __init__(self, store: KnowledgeStore):
        self.store = store

    # ------------------------------------------------------------------- connect
    def entity_for(self, obs, events):
        """The entity an observation is about (created on first mention)."""
        if obs.subject_kind == "you":
            return self.store.you()
        if obs.subject_kind == "role":
            return self.store.role_entity(obs.role)
        if obs.subject_kind == "path":                # "my uncle's pony": whatever your uncle's pony is
            owner = self.store.role_entity(obs.role)
            for f in self.store.current(owner):
                if f["value_id"] and f["relation"] == obs.path:
                    events.append(("connected", f"\"{obs.subject}\" is {self.store.entity_label(f['value_id'])}"))
                    return f["value_id"]
            label = f"your {obs.role}'s {obs.path}"
            entity_id = self.store.add_entity(label, "concept", alias=f"my {obs.role}'s {obs.path}")
            self.store.add_fact("about_you", owner, obs.path, label, obs.text, obs.source, obs.confidence,
                                f"implied: you mentioned your {obs.role}'s {obs.path}", value_id=entity_id)
            return entity_id
        found = self.store.find(obs.subject)
        if found:
            entity_id = found[0]["entity_id"]
            entity = self.store.entity(entity_id)
            if entity["kind"] == "role":
                events.append(("connected", f"\"{obs.subject}\" is {entity['name']} (the name you gave it)"))
            return entity_id
        return self.store.add_entity(obs.subject, obs.subject_kind, alias=obs.subject)

    def name_given(self, entity_id, name, events):
        """A role entity got a name ("my dog's name is Max"): the name now refers to it. If something called Max
        was already known on its own, the two are the same: merge them (an inferred connection)."""
        for other in self.store.find(name):
            if other["entity_id"] != entity_id and self.store.entity(other["entity_id"])["kind"] != "role":
                self.store.merge(entity_id, other["entity_id"],
                                 f"\"{name}\" and {self.store.entity(entity_id)['name']} are the same (inferred)")
                events.append(("connected", f"what was known about \"{name}\" now belongs to "
                                            f"{self.store.entity(entity_id)['name']} (inferred)"))
        self.store.add_alias(name, entity_id, how="stated", current=True)

    # --------------------------------------------------------------------- learn
    def learn(self, message, turn_id=None, reply_after=""):
        """Learn from one of YOUR messages. Returns what happened, for the chat's details."""
        reading = extract.read(message)
        out = {"learned": [], "confirmed": [], "corrected": [], "changed": [], "connected": [], "note": reading.note}
        if reading.forget:
            return out
        for obs in reading.observations:
            events = []
            if obs.negated:
                self.learn_negation(obs, out, turn_id, reply_after)
                continue
            subject_id = self.entity_for(obs, events)
            if obs.category == "taught" and self.store.entity(subject_id)["kind"] in ("you", "role"):
                obs.category = "about_you"          # "Max is a golden retriever": about YOUR dog, so user-specific
            value_id = None
            if obs.value_named and obs.relation != "name":
                found = self.store.find(obs.value, include_former=False)      # never a FORMER name (D-070)
                value_id = found[0]["entity_id"] if found else self.store.add_entity(obs.value, "named", alias=obs.value)
            existing = self.store.current(subject_id, obs.relation)
            same = [f for f in existing if norm(f["value"]) == norm(obs.value) and not f.get("negated")]
            if same:                                                  # said again: reinforce, no duplicate
                self.store.reinforce(same[0]["id"])
                out["confirmed"].append(self.public(self.store.fact(same[0]["id"])))
                continue
            opposite = {"likes": "dislikes", "dislikes": "likes"}.get(obs.relation)
            replaced = [f for f in existing if (obs.single or obs.correction) and not f.get("negated")]
            denied = self.denials_of(obs, subject_id)            # "I don't have a dog", then "my dog's name is Rex"
            if opposite:
                replaced += [f for f in self.store.current(subject_id, opposite) if norm(f["value"]) == norm(obs.value)]
            conflict = bool(replaced) and obs.category == "taught" and not obs.correction
            note = (f"conflicts with what you said before ({', '.join(f['value'] for f in replaced)}): kept as history, "
                    "and this is less certain") if conflict else ""
            connected = [detail for kind, detail in events if kind == "connected"]
            if connected:
                note = "; ".join(([note] if note else []) + [f"connected by the system: {connected[0]} (inferred)"])
            confidence = min(obs.confidence, CONFLICT_CONFIDENCE) if conflict else obs.confidence
            fact_id = self.store.add_fact(obs.category, subject_id, obs.relation, obs.value, obs.text, obs.source,
                                          confidence, obs.reason, value_id=value_id, turn_id=turn_id,
                                          reply_after=reply_after, note=note)
            for old in replaced:
                kind = "changed" if conflict else "corrected"
                self.store.supersede(old["id"], fact_id, kind=kind)
                out[kind].append({"from": old["value"], "to": obs.value, "fact": self.store.describe(fact_id), "id": fact_id})
                if obs.relation == "name":
                    self.store.retire_alias(old["value"], subject_id)     # a former name: still findable
            for old in denied:                                            # a negation this statement contradicts
                self.store.supersede(old["id"], fact_id, kind="corrected")
                out["corrected"].append({"from": "not " + old["value"], "to": obs.value,
                                         "fact": self.store.describe(fact_id), "id": fact_id})
            if obs.relation == "name" and obs.subject_kind == "role":
                self.name_given(subject_id, obs.value, events)
            out["learned"].append(self.public(self.store.fact(fact_id)))
            out["connected"] += [detail for kind, detail in events if kind == "connected"]
        return out

    # -------------------------------------------------------------------- negation (D-075)
    def known_subject(self, obs):
        """The entity an observation is about, WITHOUT creating anything (None when it is not known yet)."""
        if obs.subject_kind == "you":
            row = self.store.db.execute("SELECT id FROM entities WHERE kind = 'you'").fetchone()
            return row["id"] if row else None
        if obs.subject_kind == "role":
            return self.store.role_entity(obs.role, create=False)
        if obs.subject_kind == "path":
            return None
        found = self.store.find(obs.subject)
        return found[0]["entity_id"] if found else None

    def contradicted_by(self, obs, subject_id):
        """The current (positive) facts a negated observation contradicts: the same value for the same relation, or,
        for "I don't have a <role>", everything known about your <role> (and a stored "I have a <role>")."""
        if subject_id is None:
            return []
        out = [f for f in self.store.current(subject_id) if not f.get("negated") and
               same_relation(f["relation"], obs.relation) and norm(f["value"]) == norm(obs.value)]
        role = possessed_role(obs.value) if obs.subject_kind == "you" and obs.relation in POSSESSION else None
        if role:
            role_id = self.store.role_entity(role, create=False)
            if role_id is not None:
                out += [f for f in self.store.current(role_id) if not f.get("negated")]
            out += [f for f in self.store.current(subject_id) if not f.get("negated") and f["relation"] in POSSESSION
                    and possessed_role(f["value"]) == role and f not in out]
        return out

    def denials_of(self, obs, subject_id):
        """The current NEGATED facts a positive observation contradicts: "not Max" when you now say Max, or "no dog"
        when you now say something about your dog (or that you have one)."""
        out = [f for f in self.store.current(subject_id) if f.get("negated") and
               same_relation(f["relation"], obs.relation) and norm(f["value"]) == norm(obs.value)]
        row = self.store.db.execute("SELECT id FROM entities WHERE kind = 'you'").fetchone()
        you = row["id"] if row else None
        role = obs.role if obs.subject_kind in ("role", "path") else (
            possessed_role(obs.value) if obs.subject_kind == "you" and obs.relation in POSSESSION else None)
        if role and you is not None:
            out += [f for f in self.store.current(you) if f.get("negated") and f["relation"] in POSSESSION and
                    possessed_role(f["value"]) == role and f not in out]
        return out

    def learn_negation(self, obs, out, turn_id=None, reply_after=""):
        """Store what you said is NOT so, and supersede the current facts it contradicts (history kept)."""
        events = []
        subject_id = self.entity_for(obs, events) if obs.subject_kind != "path" else self.known_subject(obs)
        if subject_id is None:
            return
        if self.store.entity(subject_id)["kind"] in ("you", "role") and obs.category == "taught":
            obs.category = "about_you"
        same = [f for f in self.store.current(subject_id) if f.get("negated") and
                same_relation(f["relation"], obs.relation) and norm(f["value"]) == norm(obs.value)]
        if same:                                          # said again: confirmed
            self.store.reinforce(same[0]["id"])
            out["confirmed"].append(self.public(self.store.fact(same[0]["id"])))
            return
        contradicted = self.contradicted_by(obs, subject_id)
        conflict = obs.category == "taught" and any(f["category"] in ("taught", "general") for f in contradicted)
        note = (f"conflicts with what you said before ({', '.join(f['value'] for f in contradicted)}): kept as history, "
                "and this is less certain") if conflict else ""
        confidence = min(obs.confidence, CONFLICT_CONFIDENCE) if conflict else obs.confidence
        fact_id = self.store.add_fact(obs.category, subject_id, obs.relation, obs.value, obs.text, obs.source,
                                      confidence, obs.reason, turn_id=turn_id, reply_after=reply_after, note=note,
                                      negated=True)
        for old in contradicted:
            kind = "changed" if conflict else "corrected"
            self.store.supersede(old["id"], fact_id, kind=kind)
            out[kind].append({"from": old["value"], "to": "not " + obs.value, "fact": self.store.describe(fact_id),
                              "id": fact_id})
            if old["relation"] == "name":                 # no longer its name: a former name (still findable)
                self.store.retire_alias(old["value"], old["subject_id"])
        out["learned"].append(self.public(self.store.fact(fact_id)))

    # -------------------------------------------------------------------- recall
    def mentioned(self, question):
        """Entities named in a question: by any of their names ("Max", "France", "my dog", "water")."""
        q = " " + re.sub(r"[^a-z0-9' ]", " ", question.lower()) + " "
        found = {}
        for a in self.store.all_aliases():
            if f" {a['alias']} " in q or f" {a['alias']}'s " in q:
                found[a["entity_id"]] = max(found.get(a["entity_id"], 0), len(a["alias"]))
        about = ABOUT.search(question)
        return found, about

    def recall(self, question, k=MAX_RECALLED, now=None):
        """The current facts relevant to a question, best first."""
        facts = self.store.current()
        if not facts:
            return []
        now = time.time() if now is None else now
        mentioned, about = self.mentioned(question)
        asks_about_system = subject_of(question) == "system"
        you = self.store.you()
        query = set(terms(question, expand=False))
        fact_terms = [set(terms(f"{f['text']} {f['relation']} {f['value']}")) for f in facts]
        df = {}
        for ts in fact_terms:
            for t in ts:
                df[t] = df.get(t, 0) + 1
        idf = {t: 1.0 + math.log((len(facts) + 1) / (df.get(t, 0) + 1)) for t in query}
        weight = sum(idf.values())
        # a question about ONE favorite or ONE person or pet ("my cat") is only about that one (D-063)
        asked = FAVORITE_ASKED.search(question)
        favorite = favorite_category(asked.group(1)) if asked else None
        roles = set(re.findall(r"\bmy\s+((?:best\s+)?[a-z]+?)(?:'s|s)?\b", question.lower())) & (RELATIONS | {"pet"})
        mine = re.findall(r"\bmy\s+(?!favou?rite\b|fav\b|fave\b)([a-z]+)", question.lower())
        about_you = (bool(re.search(r"\b(?:who am i|what am i called|how old am i)\b", question, re.I)) or
                     any(w not in RELATIONS and w != "pet" for w in mine)) and not roles     # "my name", "my age"
        found = []
        for f, ft in zip(facts, fact_terms):
            if asks_about_system and (f["subject_id"] == you or f["category"] == "about_you"):
                continue                                              # "what is your name?" is not about you
            if favorite and (f["relation"] in ("likes", "dislikes") or (f["relation"].startswith("favorite ") and
                                                                         favorite_category(f["relation"][9:]) != favorite)):
                continue
            if about_you and f["subject_id"] != you and not (f["subject_id"] in mentioned or f["value_id"] in mentioned):
                continue
            if roles and not (f["subject_id"] in mentioned or f["value_id"] in mentioned):
                subject = self.store.entity(f["subject_id"])
                if subject["kind"] != "role" or not (subject["role"] in roles or ("pet" in roles and subject["role"] in PET_ROLES)):
                    continue
            words = sum(idf[t] for t in query & ft) / weight if weight else 0.0
            entity = 1.0 if (f["subject_id"] in mentioned or f["value_id"] in mentioned) else 0.0
            relation_terms = set(terms(f["relation"], expand=False))
            relation = 1.0 if relation_terms and relation_terms <= query else 0.0     # ALL of it: "favorite food"
            whole = 1.0 if (about and entity) else 0.0                # "what do you know about Max": all of Max
            relevance = min(1.0, 0.45 * entity + 0.45 * words + 0.25 * relation + 0.3 * whole)
            recency = math.exp(-max(0.0, now - f["last_confirmed"]) / (30 * 86400))
            score = relevance * (0.8 + 0.1 * f["confidence"] + 0.1 * recency)
            found.append(Recalled(f, relevance, score, {"entity": entity, "words": words, "relation": relation,
                                                         "about": whole}, self.store.describe(f["id"])))
        best = max((r.relevance for r in found), default=0.0)
        keep = [r for r in found if r.relevance >= MIN_RELEVANCE and r.relevance >= RELATIVE_CUTOFF * best]
        keep.sort(key=lambda r: -r.score)
        return keep[:k]

    # ------------------------------------------------------------------- helpers
    def public(self, f):
        """A fact as the chat page shows it."""
        from learning.knowledge import CATEGORIES, SOURCES
        return {"id": f["id"], "category": f["category"], "category_label": CATEGORIES[f["category"]],
                "about": self.store.entity_label(f["subject_id"]), "relation": f["relation"], "value": f["value"],
                "text": f["text"], "source": f["source"], "source_label": SOURCES[f["source"]],
                "confidence": round(f["confidence"], 2), "status": f["status"], "first_learned": f["first_learned"],
                "last_confirmed": f["last_confirmed"], "confirmations": f["confirmations"], "reason": f["reason"],
                "note": f["note"], "previous": [h["value"] for h in self.store.history(f["id"])],
                "negated": bool(f.get("negated"))}
