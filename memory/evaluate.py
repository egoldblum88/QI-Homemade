"""
The memory benchmark: long-term memory and the neural brain measured SEPARATELY.

    python -m memory.evaluate                           # Stage 1.5b2 vs 1.5b3 (whichever exist), ~1 minute
    python -m memory.evaluate --brain models/stage1_5b2.npz --users 10

  FORMATION    does the writer store lasting information (as the right type) and skip small talk?
               (a fixed list of example messages, labeled by hand)
  RETRIEVAL    simulated users each tell it 9 things. Later, in a NEW session (the database closed and
               reopened), they ask about them. Is the needed memory given to the brain (recall)? Is it the
               only one (precision)? Do unrelated messages recall nothing? When something was never told,
               is no wrong memory of that kind given?
  GENERATION   when the needed memory WAS given to the brain, does OUR brain's reply contain the
               remembered answer? When it was never told, does the reply say "You have not told me"?
               Per kind of question. Greedy decoding, so it is repeatable.
  CONTROL      the same questions with memory switched off: what the brain gets right without memory

All values are realistic ones that were NEVER used as answers in training (green, pizza, tennis, hiking,
Georgia, ...: brain/tasks.py RESERVED_VALUES), and names are real names from our books that the brain
never saw as names. Every simulated user gets a fresh temporary database; state/memory.db is never touched.
Writes models/memory_evaluation.md.
"""

import argparse
import random
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

from brain.generate import (STAGE15B2_MODEL, STAGE15B3_MODEL, STAGE15B4B_MODEL, STAGE15B4C_MODEL, STAGE15B4_MODEL,
                            load_brain)
from brain.tasks import RESERVED_VALUES, split_names, word_lists
from chat.chat_engine import ChatEngine
from memory.system import MemorySystem
from memory.writer import analyze

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "models" / "memory_evaluation.md"
NOT_TOLD = "not told me"

# (message, should it be stored?, expected type). Test examples only: never training data.
FORMATION_EXAMPLES = [
    ("My name is Etai.", True, "fact"), ("I have a dog named Max.", True, "relationship"),
    ("I like basketball.", True, "preference"), ("My favorite subject is history.", True, "preference"),
    ("I live in Georgia.", True, "fact"), ("Remember that my locker number is 42.", True, "fact"),
    ("I'm 15 years old.", True, "fact"), ("My sister's name is Ana.", True, "relationship"),
    ("I don't like spiders.", True, "preference"), ("I have a test tomorrow.", True, "episodic"),
    ("I won my chess tournament yesterday!", True, "episodic"), ("Max is my cat.", True, "relationship"),
    ("I'm from Brazil.", True, "fact"), ("My birthday is on March 3.", True, "fact"),
    ("I love painting.", True, "preference"), ("Pizza is my favorite food.", True, "preference"),
    ("I work at a bakery.", True, "fact"), ("Call me Sam.", True, "fact"),
    ("My best friend is called Leo.", True, "relationship"), ("I moved to Denver last year.", True, "episodic"),
    ("my favoite color is green", True, "preference"), ("my fav food is pizza", True, "preference"),
    ("My fave animal is a dog", True, "preference"), ("i like hiking", True, "preference"),
    ("my hobby is painting", True, "fact"), ("i live in georgia", True, "fact"),
    ("hello", False, None), ("what are you doing?", False, None), ("tell me a story", False, None),
    ("what is 5+5?", False, None), ("hi there!", False, None), ("thanks a lot", False, None),
    ("I'm tired", False, None), ("ok", False, None), ("What is my name?", False, None),
    ("Say something funny.", False, None), ("The weather is nice.", False, None), ("lol", False, None),
    ("Do you like music?", False, None), ("You are weird.", False, None), ("I like you", False, None),
    ("Can you count to ten?", False, None), ("my name is not Bob", False, None),
    ("That is interesting.", False, None), ("Good morning!", False, None), ("I am happy today", False, None),
    ("my phone is broken", False, None), ("my dog is sick", False, None), ("the sky is blue", False, None),
]
CATEGORIES = ["color", "food", "animal", "sport"]
PETS = ["dog", "cat", "parrot", "hamster", "rabbit"]
FAMILY = ["sister", "brother", "mother", "friend", "cousin", "uncle"]
UNRELATED = ["tell me a story", "hello", "what is 7+8?", "what is your name?", "Good morning!"]
KINDS = ["your name", "favorites", "likes", "pets", "family", "places", "missing information"]


def user_script(rng, names):
    """What one simulated user says, and the later questions: (question, memory key or None, answer, kind)."""
    name, pet_name, rel_name = rng.sample(names, 3)
    told, untold = rng.sample(CATEGORIES, 2), None
    untold = rng.choice([c for c in CATEGORIES if c not in told])
    fav = {c: rng.choice(RESERVED_VALUES[c]) for c in told}
    like, love = rng.sample(RESERVED_VALUES["like"], 2)
    pet, other_pet = rng.sample(PETS, 2)
    rel, other_rel = rng.sample(FAMILY, 2)
    place = rng.choice(RESERVED_VALUES["place"])
    said = [f"My name is {name}.", f"My favorite {told[0]} is {fav[told[0]]}.", f"My favorite {told[1]} is {fav[told[1]]}.",
            f"I like {like}.", f"I love {love}.", f"I have a {pet} named {pet_name}.", f"My {rel} is called {rel_name}.",
            f"I live in {place}.", "I won my game yesterday."]
    questions = [
        ("What is my name?", "user.name", name, "your name"),
        ("Who am I?", "user.name", name, "your name"),
        (f"What is my favorite {told[0]}?", f"pref:favorite:{told[0]}", fav[told[0]], "favorites"),
        (f"what is my favorite {told[1]}", f"pref:favorite:{told[1]}", fav[told[1]], "favorites"),
        ("What do I like?", f"pref:{like}", like, "likes"),
        ("What do I love?", f"pref:{love}", love, "likes"),
        (f"What is my {pet}'s name?", f"rel:{pet}", pet_name, "pets"),
        (f"What is my {rel} called?", f"rel:{rel}", rel_name, "family"),
        ("Where do I live?", "user.location", place, "places"),
        # never told: the right answer is that it was not given
        (f"What is my favorite {untold}?", f"pref:favorite:{untold}", NOT_TOLD, "missing information"),
        (f"What is my {other_pet}'s name?", f"rel:{other_pet}", NOT_TOLD, "missing information"),
        (f"What is my {other_rel} called?", f"rel:{other_rel}", NOT_TOLD, "missing information"),
        ("Where was I born?", "user.born", NOT_TOLD, "missing information"),
    ]
    return said, questions


def formation():
    right, right_type, should_stored, false_stores, wrong = 0, 0, 0, 0, []
    should = sum(1 for _, s, _ in FORMATION_EXAMPLES if s)
    for message, store, type in FORMATION_EXAMPLES:
        found = analyze(message).candidates
        right += bool(found) == store
        if store and found:
            should_stored += 1
            right_type += found[0].type == type
        false_stores += bool(not store and found)
        if bool(found) != store or (store and found and found[0].type != type):
            wrong.append((message, [c.summary for c in found]))
    return {"accuracy": right / len(FORMATION_EXAMPLES), "recall": should_stored / should,
            "false_stores": false_stores / (len(FORMATION_EXAMPLES) - should),
            "type_accuracy": right_type / max(1, should_stored), "n": len(FORMATION_EXAMPLES), "wrong": wrong}


def with_state(folder, condition):
    """The event log and the internal state for one simulated user (internal_state/evaluate.py compares conditions):
    "default" = the internal state as shipped; "importance" = memories formed in significant episodes get more
    importance (M1); "importance+order" = also the small recall-order bonus (M2). Returns {} for the plain benchmark
    (condition None)."""
    if condition is None:
        return {}
    from internal_state.system import InternalState
    from self_model.system import SelfSystem
    config = None if condition == "default" else \
        {"memory": {"importance": True, "recall_order": condition == "importance+order"}}
    return {"self_model": SelfSystem(Path(folder) / "events.db"),
            "internal_state": InternalState(Path(folder) / "internal_state.db", config=config)}


def close_state(parts):
    for part in parts.values():
        part.close()


def evaluate(brain_path, users=40, seed=0, log=print, state=None):
    """Retrieval and generation for one brain. The simulated users are the same for every brain (same seed).
    state: None (memory alone, as always), or an internal-state condition (see with_state)."""
    model, tokenizer, record = load_brain(brain_path)
    _, names = word_lists((ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8"))
    _, never_trained = split_names(names)
    rng = random.Random(seed)
    r = {"asked": 0, "found": 0, "retrieved": 0, "correct_retrieved": 0, "unrelated": 0, "unrelated_clean": 0,
         "missing": 0, "missing_clean": 0}
    by_kind = {k: {"asked": 0, "found": 0, "generated": 0, "generated_all": 0, "control": 0} for k in KINDS}
    examples = []
    started = time.perf_counter()
    for u in range(users):
        said, questions = user_script(rng, never_trained)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            system = MemorySystem(path)
            parts = with_state(tmp, state)
            engine = ChatEngine(model, tokenizer, record, temperature=0, memory=system, **parts)
            for message in said:
                engine.reply(message)
            system.close()
            close_state(parts)
            system = MemorySystem(path)                               # restart: a new program, the same file
            parts = with_state(tmp, state)
            engine = ChatEngine(model, tokenizer, record, temperature=0, memory=system, **parts)
            plain = ChatEngine(model, tokenizer, record, temperature=0)     # the control: memory off
            for question, key, answer, kind in questions:
                out = engine.reply(question)
                engine.clear()                                        # every question on a fresh screen
                given = [m for m in out["details"]["memory"]["retrieved"] if m["in_prompt"]]
                k = by_kind[kind]
                k["asked"] += 1
                if answer == NOT_TOLD:                               # retrieval must not give a wrong memory of it
                    clean = not any(m["key"] == key for m in given)
                    r["missing"] += 1
                    r["missing_clean"] += clean
                    k["found"] += clean
                    ok = NOT_TOLD in out["reply"].lower()
                else:
                    hit = any(m["key"] == key for m in given)
                    r["asked"] += 1
                    r["found"] += hit
                    r["retrieved"] += len(given)
                    r["correct_retrieved"] += sum(m["key"] == key for m in given)
                    k["found"] += hit
                    clean = hit
                    ok = answer.lower() in out["reply"].lower()
                k["generated"] += clean and ok
                k["generated_all"] += ok
                control = plain.reply(question)["reply"]
                plain.clear()
                k["control"] += (NOT_TOLD in control.lower()) if answer == NOT_TOLD else (answer.lower() in control.lower())
                if u < 2:
                    examples.append((question, answer, [m["summary"] for m in given], out["reply"], control))
            for message in UNRELATED:
                out = engine.reply(message)
                engine.clear()
                r["unrelated"] += 1
                r["unrelated_clean"] += not out["details"]["memory"]["retrieved"]
            system.close()
            close_state(parts)
        if (u + 1) % 10 == 0:
            log(f"  {Path(brain_path).name}: {u + 1}/{users} simulated users ({time.perf_counter() - started:.0f} s)")
    return {"retrieval": r, "by_kind": by_kind, "examples": examples, "stage": record.get("stage")}


def summary(result):
    """The numbers the success criteria use."""
    r, by_kind = result["retrieval"], result["by_kind"]
    out = {"retrieval_recall": r["found"] / r["asked"], "retrieval_precision": r["correct_retrieved"] / max(1, r["retrieved"]),
           "unrelated_clean": r["unrelated_clean"] / r["unrelated"], "missing_clean": r["missing_clean"] / r["missing"]}
    for kind, k in by_kind.items():
        out[f"generation:{kind}"] = k["generated"] / max(1, k["found"])       # given the right memory (or none)
        out[f"end_to_end:{kind}"] = k["generated_all"] / k["asked"]
        out[f"found:{kind}"] = k["found"] / k["asked"]
        out[f"control:{kind}"] = k["control"] / k["asked"]
    return out


def render(f, results, users):
    names = list(results)
    s = {n: summary(results[n]) for n in names}
    head = "| Measurement | " + " | ".join(names) + " |"
    rule = "|---|" + "---|" * len(names)
    row = lambda label, key: f"| {label} | " + " | ".join(f"{100 * s[n][key]:.1f}%" for n in names) + " |"
    lines = ["# Memory evaluation", "",
             f"> Generated by `python -m memory.evaluate` on {date.today().isoformat()}. {users} simulated users per brain "
             "(the same users for every brain), each with a fresh database, asked in a NEW session. Values never used "
             "as answers in training; names never seen as names.", "",
             "## Memory (software), measured apart from the brain", "",
             "| Measurement | Result |", "|---|---|",
             f"| Formation: stored or skipped correctly ({f['n']} labeled messages) | {100 * f['accuracy']:.0f}% |",
             f"| Formation: lasting information stored | {100 * f['recall']:.0f}% |",
             f"| Formation: small talk stored by mistake (lower is better) | {100 * f['false_stores']:.0f}% |",
             f"| Formation: right memory type | {100 * f['type_accuracy']:.0f}% |", "",
             "Retrieval is software too, but it is measured through each brain's chat so the prompt limits apply:", "",
             head, rule,
             row("Retrieval: needed memory given to the brain (recall)", "retrieval_recall"),
             row("Retrieval: recalled memories that were the needed one (precision)", "retrieval_precision"),
             row("Unrelated messages that recalled nothing", "unrelated_clean"),
             row("Never told: no wrong memory of that kind given", "missing_clean"), "",
             "## The brain (generation): when retrieval was right, does the reply use it?", "",
             "(For missing information: does the reply say \"You have not told me\"?)", "", head, rule]
    for kind in KINDS:
        lines.append(row(kind, f"generation:{kind}"))
    lines += ["", "## End to end (formation + retrieval + generation) and the memory-off control", "",
              "| Kind | " + " | ".join(f"{n}: with memory | {n}: memory off" for n in names) + " |",
              "|---|" + "---|---|" * len(names)]
    for kind in KINDS:
        lines.append(f"| {kind} | " + " | ".join(f"{100 * s[n][f'end_to_end:{kind}']:.0f}% | "
                                                 f"{100 * s[n][f'control:{kind}']:.0f}%" for n in names) + " |")
    for n in names:
        lines += ["", f"## Examples, {n} (greedy)", "",
                  "| You | Answer | Given to the brain | Reply with memory | Reply without memory |", "|---|---|---|---|---|"]
        for q, a, given, reply, control in results[n]["examples"][:13]:
            lines.append(f"| {q} | {a} | {'; '.join(given) or '(nothing)'} | {reply or '(nothing)'} | "
                         f"{control or '(nothing)'} |")
    if f["wrong"]:
        lines += ["", "## Formation mistakes", ""] + [f"- `{m}` -> {x or 'not stored'}" for m, x in f["wrong"]]
    return "\n".join(lines) + "\n"


def default_brains():
    return [(p, n) for p, n in ((STAGE15B2_MODEL, "Stage 1.5b2"), (STAGE15B3_MODEL, "Stage 1.5b3"),
                                (STAGE15B4_MODEL, "Stage 1.5b4"), (STAGE15B4B_MODEL, "Stage 1.5b4b"),
                                (STAGE15B4C_MODEL, "Stage 1.5b4c"))
            if p.exists()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure long-term memory and the brain's use of it, separately.")
    parser.add_argument("--users", type=int, default=40)
    parser.add_argument("--brain", default=None, help="one brain file (default: Stage 1.5b2 and 1.5b3)")
    args = parser.parse_args(argv)
    brains = [(Path(args.brain), Path(args.brain).stem)] if args.brain else default_brains()
    print("Memory evaluation: formation, retrieval (software) and generation (the brain), separately\n")
    f = formation()
    results = {name: evaluate(path, args.users) for path, name in brains}
    report = render(f, results, args.users)
    REPORT.write_text(report, encoding="utf-8")
    print("\n" + report.split("## Examples")[0])
    print(f"Report: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
