"""
The blank-slate scan for the internal state: the code and the stored state must hold no predefined emotions as traits,
personality, opinions, values, goals, identity, favorite things or consciousness claims. Only computed numbers
with their evidence are allowed.

    scan_code()          every identifier and every string the package can output (docstrings, which explain what
                         is NOT included, are skipped), in every internal_state/*.py file that runs in the program.
                         Not scanned: this file (it has to hold the list of words it looks for) and the measurement
                         tools evaluate.py and simulate.py, which never run in the chat and must contain test inputs
                         such as an emotional reply (like the tests/ folder)
    scan_state(store)    every stored row: only the allowed numbers, keys and labels, no trait word, and none of
                         your values (the state refers to evidence; it never copies it)

Used by tests/test_internal_state.py and python -m internal_state.evaluate. "surprise" is allowed: here it is the
standard name for an unsigned prediction error (Itti & Baldi 2009), not an emotion label.
"""

import ast
import json
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
NOT_SCANNED = {"blank_slate.py", "evaluate.py", "simulate.py"}
TRAIT_WORDS = {
    "happy", "happiness", "sad", "sadness", "angry", "anger", "fear", "fearful", "afraid", "scared", "joy", "joyful",
    "disgust", "lonely", "loneliness", "love", "hate", "anxious", "anxiety", "calm", "excited", "bored", "boredom",
    "proud", "pride", "shame", "guilt", "jealous", "envy", "hope", "despair", "frustrated", "frustration", "mood",
    "emotion", "emotions", "emotional", "feel", "feeling", "feelings", "personality", "persona", "temperament", "trait",
    "traits", "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", "opinion", "opinions",
    "belief", "beliefs", "goal", "goals", "identity", "favorite", "favourite", "conscious", "consciousness", "sentient",
    "alive", "worldview", "likes", "dislikes", "wants", "desire", "desires",
}
FIRST_PERSON = re.compile(r"\bI (?:feel|am|like|love|want|believe|hate)\b|\bI'm\b|\bmy (?:mood|feelings?)\b", re.I)
ALLOWED_LABELS = {"thing", "relation", "question form", "memory type", "gap resolved", "contradiction", "operation",
                  "information", "reasoning found conflicting values", "a contradiction was flagged"}


def words_of(text):
    return {w.lower() for w in re.split(r"[^A-Za-z]+|(?<=[a-z])(?=[A-Z])", str(text)) if w}


def scan_code(folder=PACKAGE):
    """Findings: (file, what) for every trait word in an identifier or output string, or first-person phrase."""
    findings = []
    for path in sorted(Path(folder).glob("*.py")):
        if Path(folder).resolve() == PACKAGE and path.name in NOT_SCANNED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                      and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.arg):
                names.append(node.arg)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                names.append(node.value)
                if FIRST_PERSON.search(node.value):
                    findings.append((path.name, "first person: " + node.value[:60]))
            for name in names:
                bad = words_of(name) & TRAIT_WORDS
                if bad:
                    findings.append((path.name, f"{name[:60]!r} ({', '.join(sorted(bad))})"))
    return findings


def scan_state(store, forbidden_values=()):
    """Findings for the stored state: unexpected rows, keys or labels, trait words, or copied values."""
    from internal_state.expectations import KEY, KINDS
    from internal_state.state import DECAYING
    from world.interpret import FIRST_FORMS, FORMS
    findings = []
    dump = store.dump()
    for row in dump["current"]:
        if row["dim"] not in DECAYING:
            findings.append(("current", row["dim"]))
    for row in dump["expectations"]:
        if not KEY.match(row["key"]) or row["kind"] not in KINDS:
            findings.append(("expectations", row["key"]))
    allowed = ALLOWED_LABELS | {f[0] for f in FIRST_FORMS + FORMS}

    def strings(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield "key", k
                yield from strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from strings(v)
        elif isinstance(x, str):
            yield "value", x
    for row in dump["history"]:
        for column in ("appraisal", "before", "after", "baselines", "effects"):
            for kind, text in strings(json.loads(row[column])):
                if words_of(text) & TRAIT_WORDS:
                    findings.append(("history", text))
                if kind == "value" and text not in allowed and not KEY.match(text):
                    findings.append(("history", f"unexpected text {text!r}"))
    flat = json.dumps(dump)
    findings += [("copied value", v) for v in forbidden_values if v and v in flat]
    return findings
