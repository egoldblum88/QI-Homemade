"""
The memory database: one SQLite file on this Mac (state/memory.db by default).

SQLite is built into Python. It keeps everything in one local file, survives closing the chat,
restarting the server and restarting the computer, and writes each change completely or not at
all, so a crash can never leave half a memory. No server, no cloud, no network.

Three tables:
    memories     long-term memory: what you told it that is worth keeping
    turns        conversation history: every message and reply, per session (separate from memory)
    memory_log   an audit trail: created, recalled, replaced, deleted (deleted content is NOT kept)

The one rule enforced here, whatever calls it: text the model generated can NEVER become a
memory. Only what you said (or asked it to remember) is stored as information.
"""

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "state" / "memory.db"

TYPES = ("fact", "episodic", "preference", "relationship")
# Where a memory's information came from. Only the first two are created in this stage.
SOURCES = {
    "user_stated": "you said it",
    "user_requested": "you asked for it to be remembered",
    "system_inferred": "inferred by the system (never automatically a fact)",
}
REFUSED_SOURCES = {"model_generated", "generated", "model"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    type            TEXT NOT NULL,              -- fact | episodic | preference | relationship
    content         TEXT NOT NULL,              -- your own words (command words like "remember that" removed)
    summary         TEXT NOT NULL,              -- the line shown in the memory panel
    key             TEXT,                       -- what it is about, e.g. "user.name" (a newer one replaces an older one)
    subject         TEXT NOT NULL,              -- who it is about: "user" or an entity
    entities        TEXT NOT NULL DEFAULT '[]', -- JSON list of names and things mentioned
    source          TEXT NOT NULL,              -- user_stated | user_requested | system_inferred
    confidence      REAL NOT NULL,
    importance      REAL NOT NULL,
    pinned          INTEGER NOT NULL DEFAULT 0, -- you marked it important
    created         REAL NOT NULL,              -- when (seconds since 1970)
    last_retrieved  REAL,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'active',   -- active | replaced
    replaced_by     INTEGER,
    turn_id         INTEGER,                    -- the message it came from (conversation history)
    reply_after     TEXT NOT NULL DEFAULT '',   -- the MODEL's reply at the time: generated text, never information
    reason          TEXT NOT NULL               -- why it was remembered
);
CREATE TABLE IF NOT EXISTS sessions (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    started REAL NOT NULL,
    brain   TEXT
);
CREATE TABLE IF NOT EXISTS turns (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session INTEGER NOT NULL,
    role    TEXT NOT NULL,                      -- user | ai
    text    TEXT NOT NULL,
    source  TEXT NOT NULL,                      -- user | model_generated
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    t         REAL NOT NULL,
    memory_id INTEGER,
    action    TEXT NOT NULL,                    -- created | recalled | replaced | deleted | cleared | pinned |
                                                -- importance raised (by the internal state, with the reason) |
                                                -- contradicted (you said the opposite: set aside, D-075)
    detail    TEXT NOT NULL DEFAULT ''
);
"""


@dataclass
class Memory:
    id: int
    type: str
    content: str
    summary: str
    key: str
    subject: str
    entities: list
    source: str
    confidence: float
    importance: float
    pinned: bool
    created: float
    last_retrieved: float
    retrieval_count: int
    status: str
    replaced_by: int
    turn_id: int
    reply_after: str
    reason: str
    extra: dict = field(default_factory=dict)

    def to_public(self):
        """The fields the chat page shows."""
        return {"id": self.id, "type": self.type, "content": self.content, "summary": self.summary,
                "key": self.key, "subject": self.subject, "entities": self.entities,
                "source": self.source, "source_label": SOURCES.get(self.source, self.source),
                "confidence": round(self.confidence, 2), "importance": round(self.importance, 2),
                "pinned": self.pinned, "created": self.created, "last_retrieved": self.last_retrieved,
                "retrieval_count": self.retrieval_count, "status": self.status, "reason": self.reason}


class MemoryStore:
    """All reading and writing of the memory database goes through here."""

    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()                     # the chat server answers from several threads
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.lock, self.db:
            self.db.executescript(SCHEMA)

    def close(self):
        with self.lock:
            self.db.close()

    # ------------------------------------------------------------------- memories
    def add(self, type, content, summary, source, confidence, importance, reason, key=None, subject="user",
            entities=(), turn_id=None, reply_after="", now=None):
        """Store one memory and return it. A newer memory with the same key replaces the older one; the exact
        same memory said again is not stored twice (its log records the confirmation)."""
        if source in REFUSED_SOURCES or source not in SOURCES:
            raise ValueError(f"refused: a memory cannot come from {source!r}. Only what the user said is stored "
                             "as information; model-generated text never becomes a memory.")
        if type not in TYPES:
            raise ValueError(f"unknown memory type {type!r}")
        if not content.strip():
            raise ValueError("empty memory")
        now = time.time() if now is None else now
        same = self.db.execute("SELECT id FROM memories WHERE status = 'active' AND type = ? AND lower(content) = ? "
                               "AND (key IS ? OR key = ?)", (type, content.strip().lower(), key, key)).fetchone()
        if same:                                     # said again: no duplicate, the existing memory is confirmed
            with self.lock, self.db:
                self._log(same["id"], "confirmed", "said again", now)
            return self.get(same["id"])
        with self.lock, self.db:
            cur = self.db.execute(
                "INSERT INTO memories (type, content, summary, key, subject, entities, source, confidence, importance,"
                " created, turn_id, reply_after, reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (type, content.strip(), summary, key, subject, json.dumps(sorted(set(entities))), source,
                 float(confidence), float(importance), now, turn_id, reply_after[:200], reason))
            new_id = cur.lastrowid
            self._log(new_id, "created", reason, now)
            if key:                                            # a correction replaces the older memory
                for row in self.db.execute("SELECT id FROM memories WHERE key = ? AND status = 'active' AND id != ?",
                                           (key, new_id)).fetchall():
                    self.db.execute("UPDATE memories SET status = 'replaced', replaced_by = ? WHERE id = ?",
                                    (new_id, row["id"]))
                    self._log(row["id"], "replaced", f"by memory {new_id}", now)
        return self.get(new_id)

    def get(self, memory_id):
        row = self.db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return _to_memory(row) if row else None

    def active(self):
        rows = self.db.execute("SELECT * FROM memories WHERE status = 'active' ORDER BY created, id").fetchall()
        return [_to_memory(r) for r in rows]

    def replaced(self):
        rows = self.db.execute("SELECT * FROM memories WHERE status = 'replaced' ORDER BY created, id").fetchall()
        return [_to_memory(r) for r in rows]

    def mark_retrieved(self, memory_ids, now=None):
        now = time.time() if now is None else now
        with self.lock, self.db:
            for memory_id in memory_ids:
                self.db.execute("UPDATE memories SET last_retrieved = ?, retrieval_count = retrieval_count + 1 "
                                "WHERE id = ?", (now, memory_id))
                self._log(memory_id, "recalled", "", now)

    def raise_importance(self, memory_id, amount, why):
        """Raise a memory's importance (at most 1.0), with the reason in the memory log. Used by the internal
        state for memories formed during a significant episode (internal_state/system.py, M1)."""
        with self.lock, self.db:
            found = self.db.execute("UPDATE memories SET importance = MIN(1.0, importance + ?) WHERE id = ?",
                                    (float(amount), memory_id))
            if found.rowcount:
                self._log(memory_id, "importance raised", f"+{amount:.3f}: {why}", time.time())
        return found.rowcount == 1

    def retire(self, memory_id, why):
        """You said the opposite ("I don't have a dog"): the memory is set aside (status 'replaced', kept as history
        in the log, never recalled again). Nothing new is stored from a negative statement."""
        with self.lock, self.db:
            found = self.db.execute("UPDATE memories SET status = 'replaced' WHERE id = ? AND status = 'active'",
                                    (memory_id,))
            if found.rowcount:
                self._log(memory_id, "contradicted", why, time.time())
        return found.rowcount == 1

    def set_pinned(self, memory_id, pinned):
        with self.lock, self.db:
            found = self.db.execute("UPDATE memories SET pinned = ? WHERE id = ?", (int(bool(pinned)), memory_id))
            self._log(memory_id, "pinned" if pinned else "unpinned", "", time.time())
        return found.rowcount == 1

    def delete(self, memory_id, why="deleted by you"):
        """Remove a memory completely (its content is not kept anywhere)."""
        with self.lock, self.db:
            found = self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self.db.execute("UPDATE memories SET replaced_by = NULL WHERE replaced_by = ?", (memory_id,))
            if found.rowcount:
                self._log(memory_id, "deleted", why, time.time())
        return found.rowcount == 1

    def clear(self):
        """Delete every memory (conversation history is separate: see clear_history)."""
        with self.lock, self.db:
            n = self.db.execute("DELETE FROM memories").rowcount
            self._log(None, "cleared", f"{n} memories deleted", time.time())
        return n

    def log(self, memory_id=None, limit=50):
        query = "SELECT * FROM memory_log" + (" WHERE memory_id = ?" if memory_id is not None else "")
        rows = self.db.execute(query + " ORDER BY id DESC LIMIT ?",
                               ((memory_id, limit) if memory_id is not None else (limit,))).fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------- conversation history
    def start_session(self, brain=None):
        with self.lock, self.db:
            return self.db.execute("INSERT INTO sessions (started, brain) VALUES (?, ?)",
                                   (time.time(), brain)).lastrowid

    def add_turn(self, session, role, text):
        source = "user" if role == "user" else "model_generated"
        with self.lock, self.db:
            return self.db.execute("INSERT INTO turns (session, role, text, source, created) VALUES (?,?,?,?,?)",
                                   (session, role, text, source, time.time())).lastrowid

    def turn(self, turn_id):
        row = self.db.execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()
        return dict(row) if row else None

    def history_counts(self):
        turns = self.db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        sessions = self.db.execute("SELECT COUNT(DISTINCT session) FROM turns").fetchone()[0]
        return {"turns": turns, "sessions": sessions}

    def clear_history(self):
        """Delete the stored conversation history (memories stay; they keep their own copy of your words)."""
        with self.lock, self.db:
            n = self.db.execute("DELETE FROM turns").rowcount
            self.db.execute("UPDATE memories SET turn_id = NULL")
            self._log(None, "history cleared", f"{n} turns deleted", time.time())
        return n

    # ---------------------------------------------------------------------- helpers
    def _log(self, memory_id, action, detail, t):
        self.db.execute("INSERT INTO memory_log (t, memory_id, action, detail) VALUES (?,?,?,?)",
                        (t, memory_id, action, detail))


def _to_memory(row):
    return Memory(id=row["id"], type=row["type"], content=row["content"], summary=row["summary"], key=row["key"],
                  subject=row["subject"], entities=json.loads(row["entities"]), source=row["source"],
                  confidence=row["confidence"], importance=row["importance"], pinned=bool(row["pinned"]),
                  created=row["created"], last_retrieved=row["last_retrieved"],
                  retrieval_count=row["retrieval_count"], status=row["status"], replaced_by=row["replaced_by"],
                  turn_id=row["turn_id"], reply_after=row["reply_after"], reason=row["reason"])
