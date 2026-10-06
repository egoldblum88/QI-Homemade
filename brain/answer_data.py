"""
Training data for Stage 1.5b round 4: STRUCTURED ANSWER -> SENTENCE (the ANSWER category).

Every example is made by running the REAL pipeline on a made-up world, so the brain practises exactly what
the chat will show it:

    made-up statements -> learning (knowledge store) -> the world model
    a question         -> interpretation -> retrieval -> reasoning -> the status line the brain is shown
                          ("known: Ralo freezes at 41 degrees Celsius.", "unknown: my favorite sport.",
                           "answer Ralo: Ralo freezes at 41 degrees. Pemo freezes at 58 degrees.")
    the reply to learn  written from the same reasoning result by one fixed rule: the statements in YOUR
                        voice turned around ("my" -> "your"), "Yes." / "No." first for yes/no questions, the
                        answer of a comparison, both values of a conflict, "You think ..." when you were
                        unsure, and "You have not told me ..." when it is not known.

The reasoning is never asked of the brain: the line already says known / unknown / conflicting / uncertain /
yes / no / which one. The brain only has to say it. The reply rule is used ONLY to make training examples;
the chat never writes a reply (the Originality Guard checks the reply path).

Nothing here gives the system an identity, opinions or favorites: every fact is about made-up things, or
about YOU (the person talking), and questions about the system are never made.

Round 4c (D-075): NEGATIVE answers, from what you said is NOT so ("I don't have a cat", "My horse is not a pelan",
"The leader of Vaspo is not Rindo"): "known: I do not have a cat." -> "You do not have a cat."; "no: I do not live in
Barnu." -> "No. You do not live in Barnu."; a name you negated -> "You have not told me ..."; a role you have again;
an unknown described thing ("the tallest tower in Dovel"). negative_world() makes them, from the same rules.

Held out (validation only, never trained):
    values      different made-up words and names; the benchmark's word lists (python -m world.evaluate)
                and the test values (water, 0 degrees Celsius, Kalomi, Golden Retriever, Etai, green, ...)
                are excluded from training entirely
    numbers     0, 32, 100, 212 and every number ending in 7 are never an answer in training
    wordings    some question wordings ("What is the freezing point of X?", "Which city is the capital of
                X?", ...) appear only in validation
    relations   some relation words ("glows at", "the motto of X", "how do I sharpen ...") appear only in
                validation
"""

import random
import re
from dataclasses import dataclass
from pathlib import Path

from brain.dialogue import Example
from brain.tasks import made_up_words

FAMILY = ["sister", "brother", "mother", "father", "cousin", "friend", "aunt", "uncle", "grandmother", "grandfather",
          "neighbor", "teacher", "boss", "roommate", "son", "daughter"]
PETS = ["dog", "cat", "parrot", "hamster", "rabbit", "horse", "turtle", "lizard", "goldfish", "pony"]
FAVORITES = ["song", "book", "game", "flower", "drink", "movie", "dessert", "toy", "hobby", "band", "snack", "season"]
TEMP_VERBS = {"train": ["freeze", "melt", "boil", "burn", "soften", "harden", "crack", "shrink", "swell", "smoke"],
              "held": ["glow", "fade", "bend"]}
NOUN_RELATIONS = {"train": ["leader", "river", "mountain", "language", "currency", "symbol", "mascot", "founder",
                            "largest city", "oldest tree", "main road", "flag color"],
                  "held": ["motto", "anthem", "harbor"]}
SKILL_VERBS = {"train": ["polish", "clean", "fix", "paint", "open", "wash", "fold", "plant", "build", "seal", "dry",
                         "cool"],
               "held": ["sharpen", "carve", "mend"]}
SKILL_STEPS = ["use {a} and {b}", "mix {a} with {b}", "wash it with {a}", "rub it with {a} and {b}",
               "add {a} to the {b}", "cover it with {a}", "soak it in {a} and then dry it"]
WORDINGS = {
    "temperature": {"train": ["What temperature does {s} {v} at?", "At what temperature does {s} {v}?",
                              "what temperature does {s} {v} at"],
                    "held": ["What is the {ving} point of {s}?", "When does {s} {v}?"]},
    "capital": {"train": ["What is the capital of {c}?", "What's the capital of {c}?", "What is {c}'s capital?",
                          "what is the capital of {c}"],
                "held": ["Which city is the capital of {c}?", "Tell me the capital of {c}."]},
    "relation fact": {"train": ["What is the {r} of {c}?", "What is {c}'s {r}?"],
                      "held": ["Tell me the {r} of {c}.", "Do you remember the {r} of {c}?"]},
    "location": {"train": ["Where is {n}?", "where is {n}"], "held": ["Which place is {n} in?"]},
    "lives": {"train": ["Where does my {f} live?", "where does my {f} live"],
              "held": ["Tell me where my {f} lives.", "In which place does my {f} live?"]},
    "pet name": {"train": ["What is my {p}'s name?", "What is my {p} called?"], "held": ["Tell me my {p}'s name."]},
    "likes": {"train": ["What does my {p} like?", "what does my {p} like"], "held": ["What food does my {p} like?"]},
    "breed": {"train": ["What breed is my {p}?", "What kind of {p} is {n}?", "What breed is {n}?"],
              "held": ["What type of {p} is my {p}?", "Tell me what breed my {p} is."]},
    "age": {"train": ["How old is my {f}?"], "held": ["Tell me how old my {f} is."]},
    "chain capital": {"train": ["What is the capital of the country my {f} lives in?"],
                      "held": ["What is the capital of the place where my {f} lives?"]},
    "chain breed": {"train": ["What breed is my {f}'s {p}?"], "held": ["What kind of {p} is my {f}'s {p}?"]},
    "chain likes": {"train": ["What does my {f}'s {p} like?"], "held": ["Tell me what my {f}'s {p} likes."]},
    "family name": {"train": ["What is my {f} called?", "What is my {f}'s name?"], "held": ["Who is my {f}?"]},
    "compare temp": {"train": ["Which {vs} at a lower temperature, {a} or {b}?",
                               "Which {vs} at a higher temperature, {a} or {b}?"],
                     "held": ["Does {a} {v} at a lower temperature than {b}?"]},
    "compare age": {"train": ["Who is older, my {f} or my {g}?", "Who is younger, my {f} or my {g}?",
                              "Is my {f} older than my {g}?"],
                    "held": ["Is my {f} younger than my {g}?"]},
    "skill": {"train": ["How do I {t}?", "How can I {t}?", "how do i {t}"],
              "held": ["What is the best way to {t}?", "How should I {t}?"]},
    "favorite": {"train": ["What is my favorite {k}?", "what is my favorite {k}"], "held": ["Tell me my favorite {k}."]},
    "describe": {"train": ["What do you know about {n}?", "Tell me about {n}."], "held": ["Who is {n}?"]},
    # round 4b: facts about YOU, the person talking (round 4 had none: "What is my name?" broke, D-069)
    "my name": {"train": ["What is my name?", "What's my name?", "what is my name", "Who am I?"],
                "held": ["Tell me my name.", "Do you remember my name?"]},
    "my age": {"train": ["How old am I?", "What is my age?"], "held": ["Tell me how old I am."]},
    "my home": {"train": ["Where do I live?", "where do i live"], "held": ["Tell me where I live."]},
    "my origin": {"train": ["Where am I from?"], "held": ["Where am I from?"]},
    "my birthplace": {"train": ["Where was I born?"], "held": ["Where was I born?"]},
    "my work": {"train": ["Where do I work?"], "held": ["Tell me where I work."]},
    "my likes": {"train": ["What do I like?"], "held": ["Tell me what I like."]},
    "my dislikes": {"train": ["What don't I like?"], "held": ["What don't I like?"]},
    "my things": {"train": ["What do I own?"], "held": ["Tell me what I own."]},
    "my attribute": {"train": ["What is my {a}?", "What's my {a}?"], "held": ["Tell me my {a}."]},
    "my studies": {"train": ["What do I study?"], "held": ["Tell me what I study."]},
}
ABOUT_YOU_ATTRIBUTES = ["job", "nickname", "school", "team"]       # ("hobby" is also a favorite category)
KINDS = ["about you", "temperature", "capital and location", "relationship", "two-step chain", "three-step chain",
         "corrected",
         "conflicting", "uncertain", "yes/no", "compare", "skill", "unknown", "describe", "favorite"]
HELD_NUMBERS = {0, 32, 100, 212} | {n for n in range(-60, 500) if abs(n) % 10 == 7}
TEST_VALUES = {"water", "ice", "kalomi", "zorbia", "golden", "retriever", "etai", "green", "charlie", "max", "pelo",
               "rima", "vena", "mirra", "tolu", "tom", "pomi", "ana", "ralo", "rivo", "pim", "teloma", "france", "paris",
               "narnia", "tea", "lamp", "rif", "chess", "tennis", "zeb", "glorp", "nova"}


def benchmark_words():
    """Every word the benchmarks and tests use as a value: never used in training."""
    out = set()
    for seed, n in ((90, 400), (91, 400), (92, 400), (70, 200), (71, 200), (72, 200), (93, 400), (94, 400), (95, 400)):
        out |= {w.lower() for w in made_up_words(n, seed=seed)}
    return out | TEST_VALUES


# ------------------------------------------------------------------ the reply rule
FLIPS = [(r"\bI am\b", "you are"), (r"\bI'm\b", "you're"), (r"\bI was\b", "you were"), (r"\bI\b", "you"),
         (r"\bmy\b", "your"), (r"\bMy\b", "Your"), (r"\bme\b", "you"), (r"\bmine\b", "yours"),
         (r"\bmyself\b", "yourself")]


def you_voice(text):
    """A statement in YOUR voice, turned around: "My dog Rex is a pomi." -> "Your dog Rex is a pomi."."""
    for pattern, repl in FLIPS:
        text = re.sub(pattern, repl, text)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)   # every sentence


def training_reply(frame, line, question):
    """The reply to learn, from the reasoning result and the line the brain is shown (a fixed rule)."""
    status, _, body = line.partition(":")
    body = body.strip()
    if frame.status == "unknown":
        return f"You have not told me {you_voice(body.rstrip('.'))[:1].lower()}{you_voice(body.rstrip('.'))[1:]}."
    first = re.split(r"(?<=[.!?])\s+", body)[0]
    if frame.question_type.startswith("comparison"):
        if frame.yes_no:
            return frame.answer[0].capitalize() + "."
        winner = frame.answer[0]
        if winner == "the same":
            return "They are the same."
        cmp = re.search(r"\b(?:is|are|was)\s+(?:the\s+)?(\w+er)\b", question, re.I)
        return f"{winner[:1].upper() + winner[1:]} is {cmp.group(1).lower()}." if cmp else \
            f"{winner[:1].upper() + winner[1:]}."
    if frame.yes_no and frame.answer and frame.answer[0] in ("yes", "no"):
        return f"{frame.answer[0].capitalize()}. {you_voice(first)}"
    if frame.status == "conflicting":
        return f"You first told me {frame.alternatives[0]}, then {frame.answer[0]}."
    return you_voice(body)


def checks_for(frame):
    """What the reply must contain to count as saying the result (used by the training probe)."""
    if frame.status == "unknown":
        return ["not told"]
    if frame.detail.get("negative") and not frame.yes_no:            # "You do not have a cat."
        return ["not have", re.sub(r"^no\s+", "", frame.answer[0])]
    if frame.yes_no and frame.answer and frame.answer[0] in ("yes", "no"):
        return [frame.answer[0]]
    if frame.status == "conflicting":
        return frame.alternatives[:1] + frame.answer[:1]
    if frame.question_type.startswith("everything"):
        return [frame.answer[0]]
    return frame.answer[:1]


# ------------------------------------------------------------------ made-up worlds
@dataclass
class Ask:
    question: str
    kind: str
    status: str
    answer: list


class Pools:
    """The values one split may use (training and validation never share any)."""

    def __init__(self, names, things, places, numbers, split):
        self.names, self.things, self.places, self.numbers, self.split = names, things, places, numbers, split


def _w(rng, kind, key, split, **fill):
    options = WORDINGS[key]["train"] if split == "train" else WORDINGS[key][split]
    return rng.choice(options).format(**fill)


def _ing(verb):
    return (verb[:-1] if verb.endswith("e") and verb not in ("see",) else verb) + "ing"


def _s(verb):
    return verb + ("es" if re.search(r"(?:s|sh|ch|x|z|o)$", verb) else "s")


def make_world(rng, pools, wording="train", relations="train"):
    """One made-up world: what you say, then the questions (with the answers reasoning must reach)."""
    names = rng.sample(pools.names, 9)
    things = rng.sample(pools.things, 14)
    places = rng.sample(pools.places, 8)
    n = lambda: rng.choice(pools.numbers)                                 # noqa: E731
    f, g = rng.sample(FAMILY, 2)
    p, p2 = rng.sample(PETS, 2)
    fav, fav2 = rng.sample(FAVORITES, 2)
    v = rng.choice(TEMP_VERBS[relations])
    rel = rng.choice(NOUN_RELATIONS[relations])
    sv = rng.choice(SKILL_VERBS[relations])
    s1, s2, breed, breed2, food, food2, thing, tool, stuff, value, held_value, rel_value, likes2, owned = things
    c1, c2, c3, c4, city1, city2, city3, city4 = places
    pet, pet2, first, second, third, located, cityname, _, _ = names
    d1, d2 = rng.sample(pools.numbers, 2)
    unit = rng.choice(["degrees Celsius", "degrees", "degrees Fahrenheit"])
    a1, a2 = rng.sample([x for x in pools.numbers if 3 <= x < 90], 2)
    say = [rng.choice([f"{s1.capitalize()} {_s(v)} at {d1} {unit}.", f"Learn that {s1} {_s(v)} at {d1} {unit}."]),
           f"{s2.capitalize()} {_s(v)} at {d2} {unit}.",
           rng.choice([f"The capital of {c1} is {city1}.", f"{c1}'s capital is {city1}.",
                       f"Learn that the capital of {c1} is {city1}."]),
           f"The {rel} of {c2} is {rel_value}.", f"{located} is in {c3}.",
           f"I have a {p} named {pet}.", f"{pet} is a {breed}.", f"My {p} likes {food}.",
           f"My {f} is called {first}.", f"Correction: my {f} is called {second}.", f"My {f} lives in {c1}.",
           f"My {f} has a {p2} named {pet2}.", f"{pet2} is a {breed2}.", f"My {f}'s {p2} likes {food2}.",
           f"My {f} is {a1} years old.", f"My {g} is called {third}.", f"My {g} is {a2} years old.",
           f"The capital of {c2} is {city2}.", f"The capital of {c2} is {city3}.",
           rng.choice([f"I think the capital of {c4} is {city4}.", f"Maybe the capital of {c4} is {city4}."]),
           f"To {sv} the {thing}, {rng.choice(SKILL_STEPS).format(a=tool, b=stuff)}.",
           f"My favorite {fav} is {value}.", f"I like {likes2}."]
    rng.shuffle(say)
    say.insert(say.index(f"Correction: my {f} is called {second}."), f"My {f} is called {first}.") \
        if say.index(f"Correction: my {f} is called {second}.") < say.index(f"My {f} is called {first}.") else None
    say = list(dict.fromkeys(say))
    if say.index(f"The capital of {c2} is {city3}.") < say.index(f"The capital of {c2} is {city2}."):
        i, j = say.index(f"The capital of {c2} is {city3}."), say.index(f"The capital of {c2} is {city2}.")
        say[i], say[j] = say[j], say[i]
    if say.index(f"{pet2} is a {breed2}.") < say.index(f"My {f} has a {p2} named {pet2}."):
        i, j = say.index(f"{pet2} is a {breed2}."), say.index(f"My {f} has a {p2} named {pet2}.")
        say[i], say[j] = say[j], say[i]
    temp = f"{d1} {unit}"
    lower = s1 if d1 < d2 else s2
    higher = s2 if d1 < d2 else s1
    older, younger = (second, third) if a1 > a2 else (third, second)
    W = lambda key, **fill: _w(rng, None, key, wording, **fill)            # noqa: E731
    asks = [
        Ask(W("temperature", s=s1, v=v, ving=_ing(v)), "temperature", "known", [temp]),
        Ask(W("capital", c=c1), "capital and location", "known", [city1]),
        Ask(W("relation fact", r=rel, c=c2), "capital and location", "known", [rel_value]),
        Ask(W("location", n=located), "capital and location", "known", [c3]),
        Ask(W("lives", f=f), "relationship", "known", [c1]),
        Ask(W("pet name", p=p), "relationship", "known", [pet]),
        Ask(W("likes", p=p), "relationship", "known", [food]),
        Ask(W("breed", p=p, n=pet), "two-step chain", "known", [breed]),
        Ask(W("age", f=f), "two-step chain", "known", [f"{a1} years old"]),
        Ask(W("chain capital", f=f), "three-step chain", "known", [city1]),
        Ask(W("chain breed", f=f, p=p2), "three-step chain", "known", [breed2]),
        Ask(W("chain likes", f=f, p=p2), "three-step chain", "known", [food2]),
        Ask(W("family name", f=f), "corrected", "known", [second]),
        Ask(f"Is my {f} called {first}?", "corrected", "known", ["no"]),
        Ask(W("capital", c=c2), "conflicting", "conflicting", [city2, city3]),
        Ask(W("capital", c=c4), "uncertain", "uncertain", [city4]),
        Ask(f"Is my {f} called {second}?", "yes/no", "known", ["yes"]),
        Ask(f"Does {s1} {v} at {d1} degrees?", "yes/no", "known", ["yes"]),
        Ask(f"Does {s1} {v} at {d2 if d2 != d1 else d1 + 5} degrees?", "yes/no", "known", ["no"]),
        Ask(f"Is {pet} a {breed}?", "yes/no", "known", ["yes"]),
        Ask(f"Is the capital of {c1} {city1}?", "yes/no", "known", ["yes"]),
        Ask(f"Do I have a {p}?", "yes/no", "known", ["yes"]),
        Ask(f"Do I like {likes2}?", "yes/no", "known", ["yes"]),
        Ask(W("compare temp", vs=_s(v), v=v, a=s1, b=s2), "compare", "known", [None]),
        Ask(W("compare age", f=f, g=g), "compare", "known", [None]),
        Ask(W("skill", t=f"{sv} the {thing}"), "skill", "known", [None]),
        Ask(W("favorite", k=fav), "favorite", "known", [value]),
        Ask(W("describe", n=pet), "describe", "known", [breed]),
        Ask(W("favorite", k=fav2), "unknown", "unknown", []),
        Ask(W("capital", c=c3), "unknown", "unknown", []),
        Ask(W("temperature", s=s1, v=rng.choice([x for x in TEMP_VERBS[relations] if x != v]), ving="x"), "unknown",
            "unknown", []),
        Ask(W("family name", f=rng.choice([x for x in FAMILY if x not in (f, g)])), "unknown", "unknown", []),
        Ask(W("lives", f=g), "unknown", "unknown", []),
        Ask(W("breed", p=rng.choice([x for x in PETS if x not in (p, p2)]), n=held_value.capitalize()), "unknown",
            "unknown", []),
        Ask(f"Do I like {held_value}?", "unknown", "unknown", []),
        Ask(W("skill", t=f"{rng.choice([x for x in SKILL_VERBS[relations] if x != sv])} the {owned}"), "unknown",
            "unknown", []),
        Ask(W("describe", n=held_value.capitalize()), "unknown", "unknown", []),
    ]
    for ask in asks:                                   # the expected comparison / skill answers
        if ask.kind == "compare" and ask.answer == [None]:
            q = ask.question.lower()
            if "temperature" in q:
                if q.startswith("does"):
                    ask.answer = ["yes" if s1 == lower else "no"]
                else:
                    ask.answer = [lower if "lower" in q else higher]
            else:
                if q.startswith("is my"):
                    ask.answer = ["yes" if (("older" in q) == (a1 > a2)) else "no"]
                else:
                    ask.answer = [older if "older" in q else younger]
        if ask.kind == "skill" and ask.answer == [None]:
            ask.answer = [s.split(", ", 1)[1].rstrip(".") for s in say if s.startswith(f"To {sv} the {thing},")]
    you_say, you_asks = about_you(random.Random(rng.random()), pools, wording, exclude=set(names), liked=likes2)
    for statement in you_say:                          # said at any point of the conversation
        say.insert(rng.randrange(len(say) + 1), statement)
    return say, asks + you_asks


def about_you(rng, pools, wording="train", exclude=(), liked=None):
    """Statements about YOU (the person talking) and questions about them (round 4b)."""
    W = lambda key, **fill: _w(rng, None, key, wording, **fill)            # noqa: E731
    name = rng.choice([n for n in pools.names if n not in exclude])
    written = name.lower() if rng.random() < 0.3 else name                 # people often type "my name is etai"
    age = rng.choice([x for x in pools.numbers if 12 <= x < 90])
    home, origin, birthplace = rng.sample(pools.places, 3)
    work, own_like, disliked, owned, value, studied = rng.sample(pools.things, 6)
    liked = liked or own_like                          # one thing you like per world (the world may say it already)
    attribute = rng.choice(ABOUT_YOU_ATTRIBUTES)
    told_name_age = rng.random() < 0.75
    say = [rng.choice([f"I live in {home}.", f"i live in {home}"]), rng.choice([f"I'm from {origin}.", f"I am from {origin}."]),
           f"I was born in {birthplace}.", f"I work at the {work}.", f"I don't like {disliked}.",
           f"I own a {owned}.", f"My {attribute} is {value}.", f"I study {studied}."]
    asks = [Ask(W("my home"), "about you", "known", [home]), Ask(W("my origin"), "about you", "known", [origin]),
            Ask(W("my birthplace"), "about you", "known", [birthplace]),
            Ask(W("my work"), "about you", "known", [f"the {work}"]), Ask(W("my likes"), "about you", "known", [liked]),
            Ask(W("my dislikes"), "about you", "known", [disliked]), Ask(W("my things"), "about you", "known", [f"a {owned}"]),
            Ask(W("my attribute", a=attribute), "about you", "known", [value]),
            Ask(W("my studies"), "about you", "known", [studied]),
            Ask(f"Do I live in {home}?", "yes/no", "known", ["yes"])]
    if told_name_age:
        say += [rng.choice([f"My name is {written}.", f"my name is {written}", f"Call me {name}.", f"I'm {name}."]),
                rng.choice([f"I am {age} years old.", f"I'm {age}.", f"I am {age}."])]
        other = rng.choice([n for n in pools.names if n not in exclude and n != name])
        asks += [Ask(W("my name"), "about you", "known", [written if written in say[-2] else name]),
                 Ask(W("my name"), "about you", "known", [written if written in say[-2] else name]),
                 Ask(W("my age"), "about you", "known", [f"{age} years old"]),
                 Ask(W("my age"), "about you", "known", [f"{age} years old"]),
                 Ask(f"Is my name {name}?", "yes/no", "known", ["yes"]),
                 Ask(f"Is my name {other}?", "yes/no", "known", ["no"])]
    else:
        asks += [Ask(W("my name"), "unknown", "unknown", []), Ask(W("my age"), "unknown", "unknown", [])]
    return say, asks


# ------------------------------------------------------------------ negative answers (round 4c, D-075)
NEG_ROLES = {"train": {"pet": ["cat", "parrot", "hamster", "rabbit", "horse", "turtle", "goldfish", "pony"],
                       "family": ["sister", "brother", "cousin", "friend", "aunt", "grandmother", "neighbor", "roommate"]},
             "held": {"pet": ["dog", "lizard"], "family": ["uncle", "teacher"]}}
NEG_SAY = {
    "deny": {"train": ["I don't have {a} {r}.", "I do not have {a} {r}.", "i dont have {a} {r}", "I have no {r}."],
             "held": ["I don't have {a} {r} anymore.", "I no longer have {a} {r}."]},
    "not name": {"train": ["My {r}'s name is not {n}.", "{n} is not my {r}."], "held": ["{n} isn't my {r}."]},
    "not kind": {"train": ["My {r} is not a {k}."], "held": ["My {r} isn't a {k}."]},
    "not live": {"train": ["I don't live in {p}.", "I do not live in {p}."], "held": ["I no longer live in {p}."]},
    "not work": {"train": ["I don't work at the {t}."], "held": ["I do not work at the {t}."]},
    "world not": {"train": ["The {rel} of {c} is not {v}."], "held": ["The {rel} of {c} isn't {v}."]},
}
NEG_ASK = {
    "role name": {"train": ["What is my {r}'s name?", "What is my {r} called?"], "held": ["Tell me my {r}'s name."]},
    "role kind": {"train": ["What breed is my {r}?", "What kind of {r} do I have?"],
                  "held": ["what type of {r}", "What type of {r} do I have?"]},
    "role about": {"train": ["Tell me about my {r}."], "held": ["What do you know about my {r}?"]},
    "have": {"train": ["Do I have {a} {r}?"], "held": ["Do I have any {r}s?"]},
    "is name": {"train": ["Is my {r}'s name {n}?"], "held": ["Is my {r} called {n}?"]},
    "is kind": {"train": ["Is my {r} a {k}?"], "held": ["Is my {r} a {k}?"]},
    "live": {"train": ["Do I live in {p}?"], "held": ["Do I live in {p}?"]},
    "work": {"train": ["Do I work at the {t}?"], "held": ["Do I work at the {t}?"]},
    "world is": {"train": ["Is the {rel} of {c} {v}?"], "held": ["Is the {rel} of {c} {v}?"]},
    "description": {"train": ["What is the {sup} {noun} in {p}?", "What is the {sup} {noun} on {p}?"],
                    "held": ["Tell me the {sup} {noun} in {p}."]},
}
SUPERLATIVES = {"train": ["tallest", "longest", "oldest", "biggest", "smallest", "deepest"], "held": ["highest", "widest"]}
DESC_NOUNS = {"train": ["tower", "river", "lake", "bridge", "forest", "road", "building", "island"],
              "held": ["canal", "valley"]}
NEG_KINDS = ["you do not have", "no", "not told", "yours again", "description", "control"]


def _neg(rng, table, key, split, **fill):
    options = table[key]["train"] if split == "train" else table[key].get(split) or table[key]["train"]
    if "r" in fill:
        fill["a"] = "an" if re.match(r"[aeiou]", fill["r"]) else "a"
    return rng.choice(options).format(**fill)


def negative_world(rng, pools, wording="train", relations="train"):
    """A made-up conversation with negations, and questions whose answers come from what you said is NOT so.
    relations="held": only roles never used in training (dog, lizard, uncle, teacher)."""
    S = lambda key, **fill: _neg(rng, NEG_SAY, key, wording, **fill)          # noqa: E731
    A = lambda key, **fill: _neg(rng, NEG_ASK, key, wording, **fill)          # noqa: E731
    roles = NEG_ROLES["held" if relations == "held" else "train"]
    pets = rng.sample(roles["pet"], 2) + rng.sample([r for r in NEG_ROLES["train"]["pet"]], 1)
    p1, p2, p3 = pets[0], pets[1] if pets[1] != pets[0] else rng.choice(roles["pet"]), pets[2]
    f1, f2 = rng.sample(roles["family"], 2)
    if len({p1, p2, p3}) < 3:
        p3 = rng.choice([r for r in NEG_ROLES["train"]["pet"] if r not in (p1, p2)])
    n = rng.sample(pools.names, 9)
    home, away, far, there = rng.sample(pools.places, 4)
    work = rng.choice(pools.things)
    rel = rng.choice(NOUN_RELATIONS["held" if relations == "held" else "train"])
    country, leader = rng.choice(pools.places), rng.choice([x for x in pools.names if x not in n])
    sup = rng.choice(SUPERLATIVES["held" if relations == "held" else "train"])
    noun = rng.choice(DESC_NOUNS["held" if relations == "held" else "train"])
    blocks = [
        [f"My {p1}'s name is {n[0]}.", f"{n[0]} is a {n[1]}.", S("deny", r=p1)],      # named, then: no p1
        [S("deny", r=f1)],                                                            # never named: no f1
        [f"My {p2}'s name is {n[2]}.", S("not name", r=p2, n=n[3]), S("not kind", r=p2, k=n[4])],
        [f"My {f2}'s name is {n[5]}.", S("deny", r=f2), f"My {f2}'s name is {n[6]}."],   # no f2, then yours again
        [f"My {p3}'s name is {n[7]}.", f"My {p3}'s name is not {n[7]}."],                # its only name negated
        [S("not live", p=away), f"I live in {home}."],
        [S("not work", t=work)],
        [S("world not", rel=rel, c=country, v=leader)],
    ]
    say = []
    while any(blocks):                                   # the blocks interleaved; each block keeps its order
        block = rng.choice([b for b in blocks if b])
        say.append(block.pop(0))
    asks = [
        Ask(A("role name", r=p1), "you do not have", "known", [f"no {p1}"]),
        Ask(A("role kind", r=p1), "you do not have", "known", [f"no {p1}"]),
        Ask(A("role about", r=f1), "you do not have", "known", [f"no {f1}"]),
        Ask(A("role name", r=f1), "you do not have", "known", [f"no {f1}"]),
        Ask(A("have", r=p1), "no", "known", ["no"]),
        Ask(A("have", r=f1), "no", "known", ["no"]),
        Ask(A("is name", r=p1, n=n[0]), "no", "known", ["no"]),
        Ask(A("is name", r=p2, n=n[3]), "no", "known", ["no"]),
        Ask(A("is kind", r=p2, k=n[4]), "no", "known", ["no"]),
        Ask(A("live", p=away), "no", "known", ["no"]),
        Ask(A("work", t=work), "no", "known", ["no"]),
        Ask(A("world is", rel=rel, c=country, v=leader), "no", "known", ["no"]),
        Ask(A("role name", r=p3), "not told", "unknown", []),
        Ask(A("role name", r=f2), "yours again", "known", [n[6]]),
        Ask(A("have", r=f2), "yours again", "known", ["yes"]),
        Ask(A("description", sup=sup, noun=noun, p=far), "description", "unknown", []),
        Ask(A("role name", r=p2), "control", "known", [n[2]]),
        Ask(A("live", p=home), "control", "known", ["yes"]),
    ]
    rng.shuffle(asks)
    return say, asks


# ------------------------------------------------------------------ examples
def _matches(frame, ask):
    from world import values
    if ask.status == "unknown":
        return frame.status == "unknown"
    got = frame.alternatives + frame.answer if frame.status == "conflicting" else frame.answer
    if frame.status != ask.status:
        return False
    if ask.kind == "describe":
        return any(values.same(a, g) for a in ask.answer for g in got)
    same = lambda a, g: values.same(a, g) or a.lower() == g.lower()          # noqa: E731
    return len(got) == len(ask.answer) and all(any(same(a, g) for g in got) for a in ask.answer)


def answer_examples(pools, per_kind, seed, filler_pool, wording="train", relations="train", source_tag=None,
                    maker=None, kinds=None):
    """Examples (and their checks) until every kind has `per_kind`. Returns (examples, checks, stats).
    maker/kinds: the world maker and its kinds (default: make_world and KINDS; round 4c: negative_world and NEG_KINDS)."""
    from learning.knowledge import KnowledgeStore
    from learning.learner import Learner
    from world.answer import brain_line
    from world.interpret import interpret
    from world.model import WorldModel
    from world.reason import Reasoner
    from world.retrieve import retrieve

    rng = random.Random(seed)
    maker, kinds = maker or make_world, kinds or KINDS
    counts = {k: 0 for k in kinds}
    out, checks = [], []
    stats = {"worlds": 0, "asked": 0, "pipeline_mismatch": 0, "not_interpreted": 0}
    ai_turns = [a for _, a in filler_pool]
    while min(counts.values()) < per_kind and stats["worlds"] < per_kind * 40:
        stats["worlds"] += 1
        say, asks = maker(rng, pools, wording, relations)
        store = KnowledgeStore(Path(":memory:"))
        learner = Learner(store)
        for message in say:
            learner.learn(message)
        world = WorldModel.build(store)
        for ask in asks:
            if counts[ask.kind] >= per_kind:
                continue
            stats["asked"] += 1
            query, _ = interpret(ask.question, known=lambda name: bool(world.find(name)))
            if query is None:
                stats["not_interpreted"] += 1
                continue
            frame = Reasoner(world.restricted(retrieve(world, query).edges)).answer(query)
            if not _matches(frame, ask):
                stats["pipeline_mismatch"] += 1
                continue
            line = brain_line(frame, world)
            reply = training_reply(frame, line, ask.question)
            turns = [line, rng.choice(ai_turns)]
            for _ in range(rng.choice([0, 0, 1, 1, 2])):                 # recent, unrelated turns of the chat
                turns += list(rng.choice(filler_pool))
            out.append(Example(turns + [ask.question], reply, source_tag or f"answer:{ask.kind}"))
            checks.append(checks_for(frame))
            counts[ask.kind] += 1
        store.close()
    stats["per_kind"] = counts
    return out, checks, stats
