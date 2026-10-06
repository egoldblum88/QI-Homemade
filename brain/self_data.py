"""
Stage 1.5b round 4d (D-080): practice SAYING the self-model's answer to a question about recorded state.

Every example comes from the REAL memory, learning, world model, event log and self-model, driven on a made-up
conversation by internal_state/simulate.py (no brain: the outcomes "said" / "not said" are chosen at random), and
then asked a question about recorded state. The line is the system's own (self_model/say.py):

    recorded: In the last conversation, 6 messages were received. 2 facts were learned: you live in Pelora.
    nothing recorded: No failures are recorded.
    outside: Nothing is recorded about the system's opinions.

The reply to learn is that line WITHOUT its status word: only copying of recorded state, never anything invented.
Questions about what the system is, thinks or feels get "Nothing is recorded about ...", because the self-model
records and decides none of it (D-073). Nothing here gives the system a name, identity, personality, opinion,
belief, goal or feeling, and no reply speaks as a self ("I", "my"): leak checks in tools/prepare_answers.py.

Kinds (9): last, this, learned, information, uncertain, can, failures, nothing recorded, outside.
Question wordings and identity topics are split: some are used only in validation and the benchmark (never
trained), including the two questions of your real chat. Only wordings the real chat routes to the self-model are
used (some, like "What did we talk about last time?", are taken by the reasoning rules first: left out).
"""

import random
from dataclasses import dataclass

from brain.dialogue import Example

SELF_KINDS = ["last", "this", "learned", "information", "uncertain", "can", "failures", "nothing recorded", "outside"]
FORM_KIND = {"last": "last", "this": "this", "learned": "learned", "information": "information",
             "uncertain": "uncertain", "can": "can", "cannot": "failures", "failed": "failures"}
WORDINGS = {          # (trained, held): each must classify to its form (self_model/ask.py)
    "last": (["What happened in our last conversation?", "What happened in the last chat?",
              "What happened during the previous conversation?", "What happened at our earlier session?"],
             ["What happened during our last interaction?", "What happened in the previous session?",
              "What happened in our previous talk?"]),
    "this": (["What happened in this conversation?", "What have we talked about so far?",
              "What has happened in this chat?"],
             ["What has happened during this chat?", "What have we done so far?"]),
    "learned": (["What did you learn recently?", "What have you learned?", "What did you just learn?",
                 "What have you recently learned?"],
                ["What have you found out?", "What did you remember?"]),
    "information": (["What information do you have?", "What do you know?", "What facts do you have stored?",
                     "What memories do you have?"],
                    ["How much do you know?", "What data do you have?"]),
    "uncertain": (["What are you uncertain about?", "What don't you know?", "What are you unsure about?"],
                  ["What are your gaps?", "What are you not sure about?"]),
    "can": (["What can you do?", "What are you able to do?", "What are your abilities?"],
            ["What skills do you have?", "What are your capabilities?"]),
    "failures": (["What have you failed at?", "What mistakes did you make?", "What can't you do?",
                  "What are your mistakes?"],
                 ["What did you get wrong?", "What are your limitations?"]),
    "outside": (["What do you think about it?", "What is your opinion?", "What is your name?", "Do you have feelings?",
                 "How do you feel?", "Do you like {thing}?", "What is your personality?", "Are you happy?",
                 "Are you a person?", "Who are you?", "What do you believe?", "What is your favorite {fav}?",
                 "Do you want {a_thing}?", "Are you a robot?"],
                ["What do you think about that?", "Are you conscious?", "What are your goals?", "Do you have a soul?",
                 "Are you alive?", "What is your purpose?", "Do you enjoy {thing}?", "Who made you?"]),
}
FAVORITE_WORDS = ["song", "book", "game", "color", "food", "animal", "movie"]
METRICS = ["world benchmark: compare", "world benchmark: all questions", "world benchmark: relationship",
           "input tasks (exact, %)", "using an earlier turn (exact, %)", "arithmetic, never-seen problems (exact, %)"]
CHAT = ["hello", "thanks", "ok", "good morning", "that is nice"]


@dataclass
class SelfAsk:
    kind: str
    question: str
    line: str
    reply: str
    keys: list


def _a(word):
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def _script(rng, pools, n):
    """n made-up messages: statements, questions reasoning answers (or cannot), questions about recorded state."""
    name, things, places = rng.choice(pools.names), pools.things, pools.places
    role = rng.choice(["dog", "cat", "sister", "brother", "cousin", "horse", "parrot", "friend"])
    other = rng.choice(["rabbit", "aunt", "pony", "hamster", "goldfish"])     # never a role held out of round 4c
    say = [f"My name is {name}.", f"My {role}'s name is {rng.choice(pools.names)}.", f"I live in {rng.choice(places)}.",
           f"My favorite {rng.choice(FAVORITE_WORDS)} is {rng.choice(things)}.", f"I want {_a(rng.choice(things))}.",
           f"I need {_a(rng.choice(things))}.", f"I don't have {_a(other)}.",
           f"Learn that the capital of {rng.choice(places)} is {rng.choice(places)}.",
           f"I think the capital of {rng.choice(places)} is {rng.choice(places)}.",
           f"Actually, my {role}'s name is {rng.choice(pools.names)}.",
           f"What is my {role}'s name?", f"What is my {other}'s name?", f"Where do I live?",
           f"What is my {rng.choice(['teacher', 'boss', 'neighbor'])} called?", "What did you learn recently?",
           "What can you do?", rng.choice(CHAT), rng.choice(CHAT)]
    return [rng.choice(say) for _ in range(n)]


def make_state(rng, pools):
    """A simulated system with a made-up past: earlier runs, this run, sometimes a part switched off or measurements."""
    from internal_state.simulate import Sim
    parts = rng.choices([{}, {"reasoning": False}, {"learning": False, "reasoning": False}], weights=[8, 1, 1])[0]
    sim = Sim(**parts)
    for run in range(rng.choice([0, 0, 1, 1, 1, 2])):                   # earlier conversations
        for message in _script(rng, pools, rng.randint(1, 9)):
            sim.say(message, said=rng.random() < 0.8, complete=rng.random() < 0.95)
        sim.restart(rng.randint(60, 86400))
    if rng.random() < 0.3:                                               # made-up measurements of the brain
        metrics = {m: round(rng.uniform(40, 100), 1) for m in rng.sample(METRICS, rng.randint(1, 3))}
        sim.self_model.log.record("measurement", {"report": "models/evaluation.json", "report_written": sim.t,
                                                  "report_date": None, "brain_stage": "(no brain)",
                                                  "brain_checksum": "made-up", "older_than_brain_file": False,
                                                  "metrics": metrics})
    for message in _script(rng, pools, rng.choice([0, 1, 2, 4, 6])):    # this conversation so far
        sim.say(message, said=rng.random() < 0.8, complete=rng.random() < 0.95)
    return sim


def question(rng, kind, wording, pools):
    trained, held = WORDINGS[kind]
    q = rng.choice(trained if wording == "train" else held)
    return q.format(thing=rng.choice(pools.things) + "s", a_thing=_a(rng.choice(pools.things)),
                    fav=rng.choice(FAVORITE_WORDS))


def asks_for(sim, rng, pools, wording):
    """One question per kind about this state, with the system's own line and the reply to learn."""
    from self_model.say import self_line
    model = sim.self_model.model()
    out = []
    for kind in [k for k in SELF_KINDS if k != "nothing recorded"]:
        q = question(rng, kind, wording, pools)
        sl = self_line(model, q)
        if sl is None:
            continue
        got = ("outside" if sl.status == "outside" else "nothing recorded" if sl.status == "nothing recorded"
               else FORM_KIND.get(sl.form))
        if (kind == "outside") != (got == "outside"):
            continue                             # the wording was not classified as intended: left out (counted)
        out.append(SelfAsk(got, q, sl.line, sl.body, sl.keys if sl.status == "recorded" else ["record"] + sl.keys))
    return out


def self_examples(pools, per_kind, seed, filler_pool, wording="train", source_tag=None, held_numbers=frozenset()):
    """Examples until every kind has per_kind (the answer category's layout: the line, a turn in between, 0-2
    unrelated exchanges, the question). Returns (examples, checks, stats)."""
    import re
    from internal_state.blank_slate import TRAIT_WORDS
    from self_model.say import FIRST_PERSON
    rng = random.Random(seed)
    counts = {k: 0 for k in SELF_KINDS}
    out, checks = [], []
    stats = {"states": 0, "asked": 0, "held_number_skipped": 0, "value_word_skipped": 0}
    ai_turns = [a for _, a in filler_pool]
    while min(counts.values()) < per_kind and stats["states"] < per_kind * 60:
        stats["states"] += 1
        sim = make_state(rng, pools)
        try:
            asks = asks_for(sim, rng, pools, wording)
        finally:
            sim.close()
        for ask in asks:
            stats["asked"] += 1
            if counts[ask.kind] >= per_kind:
                continue
            if held_numbers and {int(n) for n in re.findall(r"\b\d+\b", ask.reply)} & held_numbers:
                stats["held_number_skipped"] += 1
                continue
            if FIRST_PERSON.search(ask.reply) or (ask.kind != "outside" and
                                                  set(re.findall(r"[a-z]+", ask.reply.lower())) & TRAIT_WORDS):
                stats["value_word_skipped"] += 1         # a made-up value that happens to be "mine", "hope", ...
                continue
            turns = [ask.line, rng.choice(ai_turns)]
            for _ in range(rng.choice([0, 0, 1, 1, 2])):
                turns += list(rng.choice(filler_pool))
            out.append(Example(turns + [ask.question], ask.reply, source_tag or f"answer:self: {ask.kind}"))
            checks.append(ask.keys)
            counts[ask.kind] += 1
    stats["per_kind"] = counts
    return out, checks, stats
