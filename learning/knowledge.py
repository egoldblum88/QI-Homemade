"""
The knowledge store: a small knowledge graph in one SQLite file on this Mac (state/knowledge.db).

    entities    things knowledge is about: "you" (the person talking), "your dog", "Max", "France", "water"
    aliases     the names used for an entity: "my dog", "max" (and former names, after a correction)
    facts       subject --relation--> value, e.g.  your dog --name--> Max,  France --capital--> Paris
                each with: category, source, confidence, first learned, last confirmed, confirmations,
                current or superseded (the history is kept), and the sentence that taught it (evidence)
    experience  what happened while learning: learned, confirmed, corrected, changed, connected, removed

Categories are kept apart:
    about_you   your preferences and personal facts (user-specific)
    taught      world facts you stated or taught: NOT assumed true (confidence depends on the source)
    general     facts from files you approved (data/knowledge/); starts empty: no worldview is preloaded
    skill       procedures you taught ("To make tea, ...")

The one rule enforced here, whatever calls it: text the model generated can never become knowledge.
"""

import sqlite3
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "state" / "knowledge.db"

CATEGORIES = {"about_you": "about you", "taught": "taught by you", "general": "general knowledge (approved source)",
              "skill": "skill (a procedure)"}
SOURCES = {"user_stated": "you said it", "user_taught": "you taught it (\"learn that ...\")",
           "approved_source": "a file you approved", "system_inferred": "inferred by the system"}
REFUSED_SOURCES = {"model_generated", "generated", "model"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,                  -- how it is shown: "you", "your dog", "Max", "France"
    kind    TEXT NOT NULL,                  -- you | role | named | concept
    role    TEXT,                           -- for role entities: "dog", "sister" (of the person talking)
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS aliases (
    alias     TEXT NOT NULL,                -- lowercase: "my dog", "max", "france"
    entity_id INTEGER NOT NULL,
    how       TEXT NOT NULL,                -- stated (you said it) | inferred (the system connected it)
    current   INTEGER NOT NULL DEFAULT 1,   -- 0 = a former name, kept after a correction
    created   REAL NOT NULL,
    PRIMARY KEY (alias, entity_id)
);
CREATE TABLE IF NOT EXISTS facts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    category       TEXT NOT NULL,           -- about_you | taught | general | skill
    subject_id     INTEGER NOT NULL,
    relation       TEXT NOT NULL,           -- name, is a, capital, freezes at, likes, how to, ...
    value          TEXT NOT NULL,           -- as it was written
    value_id       INTEGER,                 -- the value as an entity, when it is one (a connection)
    text           TEXT NOT NULL,           -- the sentence that taught it: shown to the brain
    source         TEXT NOT NULL,           -- user_stated | user_taught | approved_source | system_inferred
    source_detail  TEXT NOT NULL DEFAULT '',
    confidence     REAL NOT NULL,
    status         TEXT NOT NULL DEFAULT 'current',     -- current | superseded
    replaced_by    INTEGER,
    first_learned  REAL NOT NULL,
    last_confirmed REAL NOT NULL,
    confirmations  INTEGER NOT NULL DEFAULT 1,
    turn_id        INTEGER,                 -- evidence: the conversation turn (in state/memory.db)
    reply_after    TEXT NOT NULL DEFAULT '',            -- the MODEL's reply then: generated, never knowledge
    reason         TEXT NOT NULL,
    note           TEXT NOT NULL DEFAULT '',
    negated        INTEGER NOT NULL DEFAULT 0   -- 1: you said it is NOT so ("my dog's name is not Max"), D-075
);
CREATE TABLE IF NOT EXISTS experience (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    t         REAL NOT NULL,
    kind      TEXT NOT NULL,                -- learned | confirmed | corrected | changed | connected | removed | imported
    fact_id   INTEGER,
    entity_id INTEGER,
    detail    TEXT NOT NULL
);
"""


def shown(fact):
    """A fact's value as the logs show it: "not Max" for a negated fact."""
    return ("not " if fact.get("negated") else "") + fact["value"]


class KnowledgeStore:

    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.lock, self.db:
            self.db.executescript(SCHEMA)
            columns = {r[1] for r in self.db.execute("PRAGMA table_info(facts)").fetchall()}
            if "negated" not in columns:              # a database from before D-075: add the column (all facts stay 0)
                self.db.execute("ALTER TABLE facts ADD COLUMN negated INTEGER NOT NULL DEFAULT 0")

    def close(self):
        with self.lock:
            self.db.close()

    def copy_in_memory(self):
        """An in-memory copy of everything (D-079: a statement is read before the reply on this copy). Nothing done to
        the copy is ever written back; close it when done."""
        copy = KnowledgeStore(":memory:")
        with self.lock:
            self.db.backup(copy.db)
        return copy

    # --------------------------------------------------------------------- entities
    def you(self):
        """The person talking (always the same entity). There is no entity for the system itself."""
        row = self.db.execute("SELECT id FROM entities WHERE kind = 'you'").fetchone()
        if row:
            return row["id"]
        return self.add_entity("you", "you")

    def add_entity(self, name, kind, role=None, alias=None, how="stated"):
        with self.lock, self.db:
            entity_id = self.db.execute("INSERT INTO entities (name, kind, role, created) VALUES (?,?,?,?)",
                                        (name, kind, role, time.time())).lastrowid
        if alias:
            self.add_alias(alias, entity_id, how)
        return entity_id

    def entity(self, entity_id):
        row = self.db.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        return dict(row) if row else None

    def add_alias(self, alias, entity_id, how="stated", current=True):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO aliases (alias, entity_id, how, current, created) VALUES (?,?,?,?,?)",
                            (alias.lower().strip(), entity_id, how, int(current), time.time()))

    def retire_alias(self, alias, entity_id):
        with self.lock, self.db:
            self.db.execute("UPDATE aliases SET current = 0 WHERE alias = ? AND entity_id = ?",
                            (alias.lower().strip(), entity_id))

    def find(self, alias, include_former=True):
        """Entities known by this name, current names first."""
        rows = self.db.execute("SELECT entity_id, how, current FROM aliases WHERE alias = ? ORDER BY current DESC",
                               (alias.lower().strip(),)).fetchall()
        return [dict(r) for r in rows if include_former or r["current"]]

    def aliases(self, entity_id):
        rows = self.db.execute("SELECT * FROM aliases WHERE entity_id = ? ORDER BY current DESC, created",
                               (entity_id,)).fetchall()
        return [dict(r) for r in rows]

    def all_aliases(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM aliases").fetchall()]

    def role_entity(self, role, create=True):
        """The person's <role> ("dog", "sister"): one entity per role, created on first mention."""
        row = self.db.execute("SELECT id FROM entities WHERE kind = 'role' AND role = ?", (role,)).fetchone()
        if row:
            return row["id"]
        if not create:
            return None
        return self.add_entity(f"your {role}", "role", role=role, alias=f"my {role}")

    def merge(self, keep_id, other_id, why):
        """Two entities turned out to be the same (e.g. "Max" and your dog): move everything onto one."""
        note = f"connected by the system: {why}"
        with self.lock, self.db:
            self.db.execute("UPDATE facts SET note = CASE WHEN note = '' THEN ? ELSE note || '; ' || ? END "
                            "WHERE subject_id = ? OR value_id = ?", (note, note, other_id, other_id))
            if self.entity(keep_id)["kind"] in ("you", "role"):        # now known to be about YOU (your dog): user-specific
                self.db.execute("UPDATE facts SET category = 'about_you' WHERE subject_id = ? AND category = 'taught'",
                                (other_id,))
            self.db.execute("UPDATE facts SET subject_id = ? WHERE subject_id = ?", (keep_id, other_id))
            self.db.execute("UPDATE facts SET value_id = ? WHERE value_id = ?", (keep_id, other_id))
            for a in self.db.execute("SELECT * FROM aliases WHERE entity_id = ?", (other_id,)).fetchall():
                self.db.execute("INSERT OR REPLACE INTO aliases (alias, entity_id, how, current, created) "
                                "VALUES (?,?,?,?,?)", (a["alias"], keep_id, "inferred", a["current"], a["created"]))
            self.db.execute("DELETE FROM aliases WHERE entity_id = ?", (other_id,))
            self.db.execute("DELETE FROM entities WHERE id = ?", (other_id,))
            self._log("connected", why, entity_id=keep_id)

    # ------------------------------------------------------------------------ facts
    def add_fact(self, category, subject_id, relation, value, text, source, confidence, reason, value_id=None,
                 source_detail="", turn_id=None, reply_after="", note="", negated=False):
        if source in REFUSED_SOURCES or source not in SOURCES:
            raise ValueError(f"refused: knowledge cannot come from {source!r}. Model-generated text never becomes "
                             "knowledge.")
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        now = time.time()
        with self.lock, self.db:
            fact_id = self.db.execute(
                "INSERT INTO facts (category, subject_id, relation, value, value_id, text, source, source_detail, "
                "confidence, first_learned, last_confirmed, turn_id, reply_after, reason, note, negated) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (category, subject_id, relation, value, value_id, text, source, source_detail, float(confidence), now,
                 now, turn_id, reply_after[:200], reason, note, int(bool(negated)))).lastrowid
            self._log("learned", f"{self.describe(fact_id)}", fact_id=fact_id, entity_id=subject_id)
        return fact_id

    def fact(self, fact_id):
        row = self.db.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return dict(row) if row else None

    def current(self, subject_id=None, relation=None, category=None):
        query, args = "SELECT * FROM facts WHERE status = 'current'", []
        for column, value in (("subject_id", subject_id), ("relation", relation), ("category", category)):
            if value is not None:
                query += f" AND {column} = ?"
                args.append(value)
        return [dict(r) for r in self.db.execute(query + " ORDER BY first_learned, id", args).fetchall()]

    def about(self, entity_id):
        """Every current fact connected to an entity: as the subject, or as the value."""
        rows = self.db.execute("SELECT * FROM facts WHERE status = 'current' AND (subject_id = ? OR value_id = ?) "
                               "ORDER BY first_learned, id", (entity_id, entity_id)).fetchall()
        return [dict(r) for r in rows]

    def reinforce(self, fact_id, turn_id=None):
        """The same information again: no new fact; it is confirmed (and becomes a little more certain)."""
        with self.lock, self.db:
            self.db.execute("UPDATE facts SET confirmations = confirmations + 1, last_confirmed = ?, "
                            "confidence = MIN(0.97, confidence + 0.02) WHERE id = ?", (time.time(), fact_id))
            self._log("confirmed", self.describe(fact_id), fact_id=fact_id)

    def supersede(self, old_id, new_id, kind="corrected"):
        """A correction: the old fact stays as history (status superseded), the new one is current."""
        with self.lock, self.db:
            self.db.execute("UPDATE facts SET status = 'superseded', replaced_by = ? WHERE id = ?", (new_id, old_id))
            self._log(kind, f"{shown(self.fact(old_id))} -> {shown(self.fact(new_id))} "
                            f"({self.describe(new_id, with_value=False)})", fact_id=new_id)

    def history(self, fact_id):
        """The earlier values this fact replaced, newest first."""
        rows = self.db.execute("SELECT * FROM facts WHERE replaced_by = ?", (fact_id,)).fetchall()
        out = []
        for r in rows:
            out.append(dict(r))
            out += self.history(r["id"])
        return out

    def remove(self, fact_id, why="removed by you"):
        """Delete a fact (and the history it replaced) completely."""
        removed = [fact_id] + [h["id"] for h in self.history(fact_id)]
        with self.lock, self.db:
            n = 0
            for i in removed:
                n += self.db.execute("DELETE FROM facts WHERE id = ?", (i,)).rowcount
            if n:
                self._log("removed", why, fact_id=fact_id)
        return n > 0

    def clear(self):
        with self.lock, self.db:
            n = self.db.execute("DELETE FROM facts").rowcount
            self.db.execute("DELETE FROM aliases")
            self.db.execute("DELETE FROM entities")
            self._log("removed", f"everything cleared ({n} facts)")
        return n

    def experience(self, limit=40):
        rows = self.db.execute("SELECT * FROM experience ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def counts(self):
        out = {c: 0 for c in CATEGORIES}
        for row in self.db.execute("SELECT category, COUNT(*) AS n FROM facts WHERE status = 'current' GROUP BY category"):
            out[row["category"]] = row["n"]
        out["superseded"] = self.db.execute("SELECT COUNT(*) FROM facts WHERE status = 'superseded'").fetchone()[0]
        out["entities"] = self.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        return out

    # ---------------------------------------------------------------------- helpers
    def entity_label(self, entity_id):
        e = self.entity(entity_id)
        if e is None:
            return "?"
        if e["kind"] == "role":
            names = [a["alias"] for a in self.aliases(entity_id) if a["current"] and not a["alias"].startswith("my ")]
            return e["name"] + (f" ({names[0].capitalize()})" if names else "")
        return e["name"]

    def describe(self, fact_id, with_value=True):
        f = self.fact(fact_id)
        if f is None:
            return "?"
        return f"{self.entity_label(f['subject_id'])} {f['relation']}" + (f" {shown(f)}" if with_value else "")

    def _log(self, kind, detail, fact_id=None, entity_id=None):
        with self.lock:
            self.db.execute("INSERT INTO experience (t, kind, fact_id, entity_id, detail) VALUES (?,?,?,?,?)",
                            (time.time(), kind, fact_id, entity_id, detail))
