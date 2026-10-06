"""
The learning benchmark: the knowledge system and the neural brain measured SEPARATELY.

    python -m learning.evaluate                 # the newest brain, 30 simulated users (~1 minute)
    python -m learning.evaluate --users 10

Each simulated user, in session 1, teaches and states things (made-up names, so no answer can come from
the brain's book knowledge):
    "The capital of Zorbia is Kalomi."            a taught fact
    "Learn that teloma freezes at 37 degrees."    an explicitly taught fact
    "I have a dog named Max." + "Max is a rifa."  two facts to CONNECT
    "My sister is called A." + "Correction: my sister is called B."     a CORRECTION
    "My favorite song is v." x3                   a DUPLICATE (must become 1 fact, confirmed 3 times)
    "To polish the X, use Y and water."           a SKILL (procedure)
    "hello", "what is 5+5?"                       small talk: nothing may be learned
Then the program is closed and opened again, and in session 2 the user asks about each.

KNOWLEDGE SYSTEM (software)
    learned correctly     the expected current facts exist, with the right category and source
    no duplicates         one fact, confirmations = 3;   correction: current B, history A;   connected: one entity
    delivered             the right CURRENT fact reached the brain's prompt (and an old value never did)
    unrelated             "tell me a story" recalls nothing
THE BRAIN (generation)
    given the right fact in its prompt, does the reply contain the answer? Per kind of question.
    Compared with learning OFF (memory only) and with both OFF.
Writes models/learning_evaluation.md. Every user gets fresh temporary databases; state/ is never touched.
"""

import argparse
import random
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

from brain.generate import load_brain
from brain.tasks import made_up_words
from chat.chat_engine import ChatEngine, text_key
from learning.system import LearningSystem
from memory.system import MemorySystem

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "models" / "learning_evaluation.md"
PETS = ["dog", "cat", "parrot", "hamster"]
FAMILY = ["sister", "brother", "cousin", "friend"]
CATEGORIES = ["song", "book", "game", "flower"]
KINDS = ["taught fact (capital)", "taught fact (learn that ...)", "connected facts", "pet name", "corrected value",
         "reinforced duplicate", "skill (how to)"]


def world(rng, names, words, places):
    country, city = rng.sample(places, 2)
    substance, breed, tool, thing = rng.sample(words, 4)
    pet_name, first, second = rng.sample(names, 3)
    pet, rel, cat = rng.choice(PETS), rng.choice(FAMILY), rng.choice(CATEGORIES)
    value, degrees = rng.choice(words), rng.randint(2, 98)
    teach = [f"The capital of {country} is {city}.", f"Learn that {substance} freezes at {degrees} degrees.",
             f"I have a {pet} named {pet_name}.", f"{pet_name} is a {breed}.", f"My {rel} is called {first}.",
             f"Correction: my {rel} is called {second}.", f"My favorite {cat} is {value}.", f"My favorite {cat} is {value}.",
             f"My favorite {cat} is {value}.", f"To polish the {thing}, use {tool} and water.", "hello", "what is 5+5?"]
    questions = [  # (question, kind, the fact text that must reach the brain, an old text that must not, answer)
        (f"What is the capital of {country}?", KINDS[0], [f"The capital of {country} is {city}."], None, city),
        (f"What temperature does {substance} freeze at?", KINDS[1], [f"{substance.capitalize()} freezes at {degrees} degrees."],
         None, str(degrees)),
        (f"What do you know about {pet_name}?", KINDS[2], [f"I have a {pet} named {pet_name}.", f"{pet_name} is a {breed}."],
         None, breed),
        (f"What is my {pet}'s name?", KINDS[3], [f"I have a {pet} named {pet_name}."], None, pet_name),
        (f"What is my {rel} called?", KINDS[4], [f"My {rel} is called {second}."], first, second),
        (f"What is my favorite {cat}?", KINDS[5], [f"My favorite {cat} is {value}."], None, value),
        (f"How do I polish the {thing}?", KINDS[6], [f"To polish the {thing}, use {tool} and water."], None, tool),
    ]
    expected = {("capital", city), ("freezes at", f"{degrees} degrees"), ("name", pet_name), ("is a", breed),
                ("name", second), (f"favorite {cat}", value), ("how to", f"use {tool} and water")}
    return teach, questions, expected, (rel, first, second, cat)


def run(brain_path=None, users=30, seed=0, log=print):
    model, tokenizer, record = load_brain(brain_path)
    names = made_up_words(400, seed=90, capitalize=True)
    words = made_up_words(400, seed=91)
    places = made_up_words(400, seed=92, capitalize=True)
    rng = random.Random(seed)
    sw = {"learned": 0, "expected": 0, "small_talk_clean": 0, "small_talk": 0, "duplicates_ok": 0, "corrections_ok": 0,
          "connections_ok": 0, "unrelated_clean": 0, "unrelated": 0, "users": users}
    kinds = {k: {"asked": 0, "delivered": 0, "old_shown": 0, "used": 0, "used_all": 0, "memory_only": 0, "nothing": 0}
             for k in KINDS}
    examples = []
    started = time.perf_counter()
    for u in range(users):
        teach, questions, expected, (rel, first, second, cat) = world(rng, names, words, places)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            engine = ChatEngine(model, tokenizer, record, temperature=0, memory=MemorySystem(tmp / "m.db"),
                                learning=LearningSystem(tmp / "k.db"))
            before = len(engine.learning.store.current())
            for message in teach:
                engine.reply(message)
                if message in ("hello", "what is 5+5?"):
                    sw["small_talk"] += 1
                    sw["small_talk_clean"] += len(engine.learning.store.current()) == before
                before = len(engine.learning.store.current())
            engine.memory.close()
            engine.learning.close()
            # session 2: a new program on the same files
            engine = ChatEngine(model, tokenizer, record, temperature=0, memory=MemorySystem(tmp / "m.db"),
                                learning=LearningSystem(tmp / "k.db"))
            store = engine.learning.store
            current = {(f["relation"], f["value"]) for f in store.current()}
            sw["expected"] += len(expected)
            sw["learned"] += len(expected & current)
            fav = store.current(relation=f"favorite {cat}")
            sw["duplicates_ok"] += len(fav) == 1 and fav[0]["confirmations"] == 3
            names_now = [f for f in store.current(relation="name") if f["value"] in (first, second)]
            sw["corrections_ok"] += (len(names_now) == 1 and names_now[0]["value"] == second and
                                     [h["value"] for h in store.history(names_now[0]["id"])] == [first])
            dogs = [e for e in (store.find(teach[2].split()[-1].rstrip(".")) or [])]
            sw["connections_ok"] += bool(dogs) and len({f["relation"] for f in store.about(dogs[0]["entity_id"])}) == 2
            memory_only = ChatEngine(model, tokenizer, record, temperature=0, memory=MemorySystem(tmp / "m.db"))
            nothing = ChatEngine(model, tokenizer, record, temperature=0)
            for question, kind, must, old, answer in questions:
                out = engine.reply(question)
                engine.clear()
                prompt = text_key(out["details"]["prompt"])
                delivered = all(text_key(t) in prompt for t in must)
                k = kinds[kind]
                k["asked"] += 1
                k["delivered"] += delivered
                k["old_shown"] += bool(old) and text_key(old) in prompt
                used = answer.lower() in out["reply"].lower()
                k["used"] += delivered and used
                k["used_all"] += used
                k["memory_only"] += answer.lower() in memory_only.reply(question)["reply"].lower()
                memory_only.clear()
                k["nothing"] += answer.lower() in nothing.reply(question)["reply"].lower()
                nothing.clear()
                if u < 2:
                    examples.append((question, answer, delivered, out["reply"]))
            for message in ("tell me a story", "Good morning!"):
                out = engine.reply(message)
                engine.clear()
                sw["unrelated"] += 1
                sw["unrelated_clean"] += not out["details"]["knowledge"]["recalled"]
            engine.memory.close()
            engine.learning.close()
            memory_only.memory.close()
        if (u + 1) % 10 == 0:
            log(f"  {u + 1}/{users} simulated users ({time.perf_counter() - started:.0f} s)")
    return {"software": sw, "kinds": kinds, "examples": examples, "stage": record.get("stage")}


def render(result):
    sw, kinds = result["software"], result["kinds"]
    pct = lambda a, b: f"{100 * a / max(1, b):.1f}%"
    lines = ["# Learning evaluation", "",
             f"> Generated by `python -m learning.evaluate` on {date.today().isoformat()} with Stage {result['stage']}. "
             f"{sw['users']} simulated users, made-up names, taught in one session and asked in the next.", "",
             "## The knowledge system (software)", "", "| Measurement | Result |", "|---|---|",
             f"| Expected facts learned correctly (after restarting) | {pct(sw['learned'], sw['expected'])} |",
             f"| Small talk learned nothing | {pct(sw['small_talk_clean'], sw['small_talk'])} |",
             f"| Said 3 times = 1 fact confirmed 3 times | {pct(sw['duplicates_ok'], sw['users'])} |",
             f"| Correction: current value new, old one kept as history | {pct(sw['corrections_ok'], sw['users'])} |",
             f"| Connected: both facts about the pet on one entity | {pct(sw['connections_ok'], sw['users'])} |",
             f"| Unrelated messages recalled nothing | {pct(sw['unrelated_clean'], sw['unrelated'])} |", "",
             "## Delivery and generation, per kind of question", "",
             "| Question | Right fact delivered to the brain | Old value shown | Brain used it (given delivered) | "
             "Memory only, no learning | Neither |", "|---|---|---|---|---|---|"]
    for kind, k in kinds.items():
        lines.append(f"| {kind} | {pct(k['delivered'], k['asked'])} | {pct(k['old_shown'], k['asked'])} | "
                     f"{pct(k['used'], k['delivered'])} | {pct(k['memory_only'], k['asked'])} | {pct(k['nothing'], k['asked'])} |")
    lines += ["", "Delivery is the knowledge system's job; using what was delivered is the brain's. The brain was trained to "
              "answer from earlier turns only for names, friends, favorites, likes, pets, family and places (Stage 1.5b "
              "rounds 2-3), never for capitals, freezing points, \"what do you know about\" or \"how do I\" questions.", "",
              "## Examples (greedy)", "", "| You ask | Answer | Delivered | Reply |", "|---|---|---|---|"]
    for q, a, d, reply in result["examples"]:
        lines.append(f"| {q} | {a} | {'yes' if d else 'no'} | {reply or '(nothing)'} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure learning: the knowledge system and the brain, separately.")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--brain", default=None, help="a brain file (default: the newest)")
    args = parser.parse_args(argv)
    print("Learning evaluation: knowledge system (software) and brain (generation), separately\n")
    result = run(args.brain, args.users)
    report = render(result)
    REPORT.write_text(report, encoding="utf-8")
    print("\n" + report.split("## Examples")[0])
    print(f"Report: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
