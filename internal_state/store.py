"""
The internal state's own database: one SQLite file on this Mac (state/internal_state.db), separate from memory,
knowledge and the event log. With long-term memory off it is kept in RAM only.

    current       the CURRENT state: one row per decaying number (value, baseline, when it was last updated)
    expectations  the learned predictors (context, p, outcomes seen, learning gain, last used)
    tags          memories stored during a significant episode: salience, the state at encoding, the importance
                  added, when the tag was renewed (new evidence) and last reinstated (recall)
    gaps          questions answered "unknown": the reasoning event that opened each, and the event that resolved it
    history       the LONG-TERM STATE HISTORY: one row per appraised episode (or reset), with the state before and
                  after, the appraisal items, the effects and the event ids that caused it

Nothing here copies an event: history rows REFER to the event log (the evidence). Only computed numbers are kept.
"""

import json
import sqlite3
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "state" / "internal_state.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS current (
    dim      TEXT PRIMARY KEY,
    value    REAL NOT NULL,
    baseline REAL NOT NULL,
    t        REAL
);
CREATE TABLE IF NOT EXISTS expectations (
    key       TEXT PRIMARY KEY,
    kind      TEXT NOT NULL,
    p         REAL NOT NULL,
    n         INTEGER NOT NULL,
    gain      REAL NOT NULL,
    last_used REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tags (
    memory_id       INTEGER PRIMARY KEY,
    salience        REAL NOT NULL,
    valence         REAL NOT NULL,
    activation      REAL NOT NULL,
    boost           REAL NOT NULL,
    event_id        INTEGER NOT NULL,
    t               REAL NOT NULL,
    renewed         REAL NOT NULL,
    last_reinstated REAL
);
CREATE TABLE IF NOT EXISTS gaps (
    event_id    INTEGER PRIMARY KEY,
    opened      REAL NOT NULL,
    resolved_by INTEGER,
    resolved_t  REAL
);
CREATE TABLE IF NOT EXISTS history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    t         REAL NOT NULL,
    kind      TEXT NOT NULL,
    episode   INTEGER,
    events    TEXT NOT NULL DEFAULT '[]',
    appraisal TEXT NOT NULL DEFAULT '{}',
    before    TEXT NOT NULL DEFAULT '{}',
    after     TEXT NOT NULL DEFAULT '{}',
    baselines TEXT NOT NULL DEFAULT '{}',
    effects   TEXT NOT NULL DEFAULT '{}'
);
"""
HISTORY_KINDS = {"episode", "reset baselines", "reset everything"}


class StateStore:

    def __init__(self, path=DEFAULT_DB):
        self.persistent = str(path) != ":memory:"
        self.path = Path(path) if self.persistent else None
        if self.persistent:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.lock, self.db:
            self.db.executescript(SCHEMA)

    def close(self):
        with self.lock:
            self.db.close()

    # --------------------------------------------------------------------- current
    def current(self):
        with self.lock:
            return {r["dim"]: dict(r) for r in self.db.execute("SELECT * FROM current").fetchall()}

    def save_current(self, values, baselines, t):
        with self.lock, self.db:
            for dim, value in values.items():
                self.db.execute("INSERT OR REPLACE INTO current (dim, value, baseline, t) VALUES (?,?,?,?)",
                                (dim, float(value), float(baselines[dim]), t))

    # ---------------------------------------------------------------- expectations
    def expectations(self):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM expectations ORDER BY key").fetchall()]

    def save_expectation(self, row):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO expectations (key, kind, p, n, gain, last_used) VALUES (?,?,?,?,?,?)",
                            (row["key"], row["kind"], row["p"], row["n"], row["gain"], row["last_used"]))

    # ------------------------------------------------------------------------ tags
    def tags(self):
        with self.lock:
            return {r["memory_id"]: dict(r) for r in self.db.execute("SELECT * FROM tags").fetchall()}

    def save_tag(self, tag):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO tags (memory_id, salience, valence, activation, boost, event_id, t, "
                            "renewed, last_reinstated) VALUES (?,?,?,?,?,?,?,?,?)",
                            (tag["memory_id"], tag["salience"], tag["valence"], tag["activation"], tag["boost"],
                             tag["event_id"], tag["t"], tag["renewed"], tag.get("last_reinstated")))

    def drop_tags(self, memory_ids):
        with self.lock, self.db:
            for memory_id in memory_ids:
                self.db.execute("DELETE FROM tags WHERE memory_id = ?", (memory_id,))

    # ------------------------------------------------------------------------ gaps
    def gaps(self, open_only=True):
        query = "SELECT * FROM gaps" + (" WHERE resolved_by IS NULL" if open_only else "") + " ORDER BY event_id"
        with self.lock:
            return [dict(r) for r in self.db.execute(query).fetchall()]

    def open_gap(self, event_id, t):
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO gaps (event_id, opened) VALUES (?, ?)", (event_id, t))

    def resolve_gap(self, event_id, by, t):
        with self.lock, self.db:
            self.db.execute("UPDATE gaps SET resolved_by = ?, resolved_t = ? WHERE event_id = ?", (by, t, event_id))

    # --------------------------------------------------------------------- history
    def add_history(self, kind, t, episode=None, events=(), appraisal=None, before=None, after=None, baselines=None,
                    effects=None):
        if kind not in HISTORY_KINDS:
            raise ValueError(f"unknown history kind {kind!r}")
        with self.lock, self.db:
            return self.db.execute(
                "INSERT INTO history (t, kind, episode, events, appraisal, before, after, baselines, effects) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (t, kind, episode, json.dumps(list(events)), json.dumps(appraisal or {}, sort_keys=True),
                 json.dumps(before or {}, sort_keys=True), json.dumps(after or {}, sort_keys=True),
                 json.dumps(baselines or {}, sort_keys=True), json.dumps(effects or {}, sort_keys=True))).lastrowid

    def history(self, limit=None, newest_first=True):
        query = "SELECT * FROM history ORDER BY id" + (" DESC" if newest_first else "") + (" LIMIT ?" if limit else "")
        with self.lock:
            rows = self.db.execute(query, (int(limit),) if limit else ()).fetchall()
        out = []
        for r in rows:
            row = dict(r)
            for key in ("events", "appraisal", "before", "after", "baselines", "effects"):
                row[key] = json.loads(row[key])
            out.append(row)
        return out

    # ----------------------------------------------------------------------- reset
    def clear(self):
        """Everything back to the start: no state, no expectations, no tags, no gaps, no history."""
        with self.lock, self.db:
            for table in ("current", "expectations", "tags", "gaps", "history"):
                self.db.execute(f"DELETE FROM {table}")

    def dump(self):
        """Every stored row (for the blank-slate scan and the tests)."""
        with self.lock:
            return {table: [dict(r) for r in self.db.execute(f"SELECT * FROM {table}").fetchall()]
                    for table in ("current", "expectations", "tags", "gaps", "history")}
