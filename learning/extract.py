"""
Reading one of YOUR sentences: what, if anything, can be learned from it?

Only your messages are read here; the model's replies never are.

Each finding is an Observation:  subject --relation--> value,  plus where it came from and how sure to be.

  about you   personal facts and preferences: the memory rules (memory/writer.py) find these, so both
              systems agree on what you said. "My dog's name is Max."  ->  your dog --name--> Max
  taught      world facts:
                "The capital of France is Paris."           ->  France --capital--> Paris
                "France's capital is Paris."                ->  France --capital--> Paris
                "Max is a golden retriever."                ->  Max --is a--> golden retriever
                "Paris is in France."                       ->  Paris --is in--> France
                "Learn that water freezes at 0 degrees."    ->  water --freezes at--> 0 degrees
  skill       procedures:  "To make tea, boil water and add the leaves."  ->  make tea --how to--> boil ...

A world fact is learned when you TEACH it ("learn that ...", "remember that ...", "fact: ...", "did you
know that ...") or when it has a clear definitional form (the X of Y is Z; Y's X is Z; X is a Z; a named
thing is in Z). A passing remark ("the weather is nice") is not learned unless you teach it.

Confidence depends only on the source (D-061):
  you said it about yourself 0.9 (you asked to remember it: 0.95) · you taught it 0.8 · you stated it 0.7
  · you sounded unsure: 0.3 lower. Nothing you say is assumed to be objectively true.

Negated statements ("I don't have a dog", "My dog's name is not Max", D-075) are read as the positive statement
with the same rules, and marked NEGATED: you said it is not so. They are never learned as positive facts.

These are readable rules chosen by us, like the memory rules; every learned fact records which rule found it.
"""

import re
from dataclasses import dataclass, field

from memory import negation
from memory.writer import (CORRECTION_RE, FORGET_RE, HEDGE_RE, LEADING_WORDS, QUESTION_START, RELATIONS, TIME_RE, analyze,
                           strip_trailing)

TEACH_RE = re.compile(r"^(?:please\s+)?(?:learn|know|note|remember|keep in mind|fact|fyi|did you know|here is a fact)"
                      r"\b\s*(?:that\s+|this\s*:\s*|:\s*|,\s*)?(?P<rest>.+)$", re.I)
ABOUT_SYSTEM = re.compile(r"^(?:you|you're|youre|your|yours)\b", re.I)
END = r"\s*[.!]*$"
SUBJECT = r"(?P<subj>(?:the\s+|a\s+|an\s+)?[\w'-]+(?:\s[\w'-]+){0,3}?)"
CONFIDENCE = {"you_stated": 0.9, "you_requested": 0.95, "taught": 0.8, "stated": 0.7}
HEDGE_PENALTY = 0.3
# relations with one value at a time: a new value replaces the old one (keeping the history)
SINGLE = {"name", "age", "lives in", "is from", "was born", "birthday", "works", "goes to", "capital", "breed",
          "color", "colour", "population", "boiling point", "freezing point", "currency", "language", "how to"}

# "<thing> <verb>s <...>" without "learn that" is a general statement (Stage 2c: "Water freezes at 0 degrees
# Celsius."), unless the verb only reports talk, thought or appearance: those are not facts about the thing
NOT_FACT_VERBS = {"says", "thinks", "means", "seems", "looks", "sounds", "feels", "wants", "needs", "knows", "gets",
                  "hopes", "wishes", "guesses", "believes", "happens", "does"}
NOT_SUBJECTS = {"it", "this", "that", "there", "he", "she", "they", "we", "everything", "something", "nothing",
                "everyone", "someone", "nobody", "one", "which", "what", "who", "here", "life", "time"}
ROLE_SUBJECT = "Rolesubjectx"           # stands for "my <role>" while a sentence about your dog/sister is read
# the people and animals "my ..." can introduce (the memory rules' list, plus more animals and relatives)
ROLE_WORDS = RELATIONS | {"goldfish", "mouse", "puppy", "kitten", "pig", "cow", "goat", "chicken", "duck", "frog",
                          "gerbil", "ferret", "canary", "tortoise", "grandson", "granddaughter", "nephew", "niece",
                          "colleague", "classmate", "baby", "stepmother", "stepfather", "partner"}
# your own things, said in the first person: "I own a bike", "I speak Hebrew" (memory reads names, likes, places),
# and what you want: "I want a dog", "I need a bike", "I would like a cup of tea" (D-079)
FIRST_PERSON_VERBS = {"own": "owns", "speak": "speaks", "play": "plays", "drive": "drives", "ride": "rides",
                      "collect": "collects", "have": "has", "want": "wants", "need": "needs", "would like": "would like"}
WISHES = {"want", "need", "would like"}
# a wish is a fact only about a THING: not an action ("I want to know ..."), a person ("I need you to ...") or a
# word that points elsewhere ("I want that")
NOT_A_THING = re.compile(r"^(?:to|you|me|it|this|that|these|those|them|him|her|us|nothing|something|anything|"
                         r"everything)\b", re.I)

# personal facts: memory's key -> (whose, relation, one value at a time?)
PERSONAL = {"user.name": ("you", "name", True), "user.age": ("you", "age", True),
            "user.location": ("you", "lives in", True), "user.origin": ("you", "is from", True),
            "user.born": ("you", "was born", True), "user.birthday": ("you", "birthday", True),
            "user.work": ("you", "works", True), "user.study": ("you", "studies", False),
            "user.goes_to": ("you", "goes to", True)}


@dataclass
class Observation:
    category: str           # about_you | taught | skill
    subject: str            # as written: "you", "my dog", "Max", "France", "water", "make tea"
    subject_kind: str       # you | role | named | concept
    relation: str
    value: str
    text: str               # the sentence that teaches it (your words): shown to the brain later
    source: str             # user_stated | user_taught
    confidence: float
    reason: str
    role: str = None        # for "my <role>" subjects: "dog", "sister"
    path: str = None        # for "my <role>'s <thing>" subjects: "pony" (your uncle's pony)
    single: bool = False    # one value at a time
    correction: bool = False
    value_named: bool = False   # the value is a name: it becomes an entity (a connection)
    negated: bool = False       # you said it is NOT so (D-075)


@dataclass
class Reading:
    observations: list = field(default_factory=list)
    forget: str = None      # "forget that ...": what to forget
    note: str = ""          # why nothing was learned


def _clean(v):
    return re.sub(r"\s+", " ", v).strip().strip(" .!,;:")


def _polish(s):
    s = s.strip()
    s = s[:1].upper() + s[1:]
    return s if s[-1:] in ".!?" else s + "."


def _concept(phrase):
    """'the capital' -> 'capital';  'A whale' -> 'whale'."""
    return re.sub(r"^(?:the|a|an)\s+", "", _clean(phrase), flags=re.I)


def _kind(phrase):
    """named: written with a capital letter (France, Max); otherwise a concept (water)."""
    return "named" if re.match(r"^[A-Z]", _concept(phrase)) else "concept"


def _is_named(value):
    return bool(re.fullmatch(r"(?:[A-Z][\w'-]*)(?:\s[A-Z][\w'-]*){0,3}", _clean(value)))


# --------------------------------------------------------------------- world facts
def _world(sentence, taught):
    """(subject, relation, value, rule, definitional?) for one sentence about the world, or None."""
    s = sentence
    m = re.match(r"^(?:the\s+)?(?P<rel>[a-z][a-z ]{1,30}?)\s+of\s+" + SUBJECT + r"\s+(?:is|are|was)\s+(?P<val>.+?)" + END,
                 s, re.I)
    if m and m["rel"].lower() not in {"one", "some", "all", "most", "many", "none"}:
        return m["subj"], m["rel"].lower(), m["val"], "\"the <relation> of <thing> is <value>\"", True
    m = re.match(r"^(?P<subj>(?-i:[A-Z])[\w'-]*(?:\s(?-i:[A-Z])[\w'-]*)*)'s\s+(?P<rel>[a-z][a-z ]{1,30}?)\s+(?:is|are)\s+"
                 r"(?P<val>.+?)" + END, s, re.I)
    if m:
        return m["subj"], m["rel"].lower(), m["val"], "\"<thing>'s <relation> is <value>\"", True
    m = re.match(r"^" + SUBJECT + r"\s+(?:is|are)\s+(?:a|an)\s+(?P<val>.+?)" + END, s, re.I)
    if m and (taught or _kind(m["subj"]) == "named" or re.match(r"^(?:a|an)\s", m["subj"], re.I)):
        return m["subj"], "is a", m["val"], "\"<thing> is a <kind>\"", True
    m = re.match(r"^(?P<subj>(?-i:[A-Z])[\w'-]*(?:\s(?-i:[A-Z])[\w'-]*)*)\s+(?:is|are)\s+(?P<rel>in|near|on|part of|next to)"
                 r"\s+(?P<val>.+?)" + END, s, re.I)
    if m:
        return m["subj"], f"is {m['rel'].lower()}", m["val"], "\"<Name> is in <place>\"", True
    m = re.match(r"^" + SUBJECT + r"\s+(?:is|are)\s+(?P<val>\d+(?:\.\d+)?|[a-z]+(?:[- ][a-z]+)?)\s+(?:years?|yrs?)\s+old"
                 + END, s, re.I)
    if m:
        return m["subj"], "age", m["val"] + " years old", "\"<thing> is <N> years old\"", True
    m = re.match(r"^" + SUBJECT + r"\s+(?:has|have|owns|own)\s+(?:a|an|one)\s+(?P<rel>[a-z]+(?:\s[a-z]+)?)\s+(?:named|called)\s+"
                 r"(?P<val>(?-i:[A-Z])[\w'-]*(?:\s(?-i:[A-Z])[\w'-]*)*)" + END, s, re.I)
    if m:
        return m["subj"], m["rel"].lower(), m["val"], "\"<thing> has a <relation> named <Name>\"", True
    m = re.match(r"^" + SUBJECT + r"\s+(?P<verb>[a-z]{2,}(?<!s)s)\b\s*(?P<prep>(?:at|in|on|to|from|by|for|with|into|near|"
                 r"under|over)\b)?\s*(?P<val>.+?)" + END, s, re.I)
    if m and m["val"] and m["verb"].lower() not in NOT_FACT_VERBS and not TIME_RE.search(s) and \
            len(m["val"].split()) <= 6 and _concept(m["subj"]).lower() not in NOT_SUBJECTS:
        relation = (m["verb"] + (" " + m["prep"] if m["prep"] else "")).lower()
        return m["subj"], relation, m["val"], "\"<thing> <verb>s <...>\" (a general statement)", False
    if not taught:
        return None
    # taught only: a general "<thing> <verb> ..." sentence
    m = re.match(r"^" + SUBJECT + r"\s+(?P<verb>(?:can|cannot|can't|will|won't|does not|doesn't|do not|don't)\s+[a-z]+"
                 r"|is|are|was|were|has|have|[a-z]+(?:s|ed))\b\s*(?P<prep>(?:at|in|on|to|from|of|by|for|with|into|about)\b)?"
                 r"\s*(?P<val>.+?)" + END, s, re.I)
    if m and m["val"]:
        relation = (m["verb"] + (" " + m["prep"] if m["prep"] else "")).lower()
        return m["subj"], relation, m["val"], "\"<thing> <verb> <...>\" (taught)", False
    m = re.match(r"^(?P<subj>(?:the\s+)?[\w'-]+)\s+(?P<verb>[a-z]+)\s+(?P<val>.+?)" + END, s, re.I)   # "bees make honey"
    if m:
        return m["subj"], m["verb"].lower(), m["val"], "\"<thing> <verb> <...>\" (taught)", False
    return None


def _about_role(sentence, taught):
    """ "My sister lives in Zorbia", "My dog's breed is poodle", "My sister has a cat named Tom": a fact about
    your <role>, read with the same world rules. Names ("my dog is called Max") are memory's (step 1).
    Returns (role, relation, value, rule) or None."""
    m = re.match(r"^my\s+(?P<role>best friend|[a-z]+)(?P<poss>'s|s')?\s+(?P<rest>.+)$", sentence, re.I)
    if not m or m["role"].lower() not in ROLE_WORDS:
        return None
    rest = m["rest"]
    owned = re.match(r"^(?P<thing>[a-z]+)\s+(?P<pred>.+)$", rest, re.I) if m["poss"] else None
    if owned and owned["thing"].lower() in ROLE_WORDS:          # "my uncle's pony likes apples"
        found = _world(f"{ROLE_SUBJECT} {owned['pred']}", taught)
        if found is not None and found[0] == ROLE_SUBJECT and found[1] not in ("name", "named", "called"):
            _, relation, value, rule, _ = found
            return (m["role"].lower(), relation, value, rule.replace("<thing>", "my <role>'s <thing>"),
                    owned["thing"].lower())
    if re.match(r"^(?:is|was)\s+(?:called|named)\b|^name\s+is\b|^(?:is|was)\s+(?-i:[A-Z])", rest, re.I):
        return None                      # its name: read by the memory rules
    found = _world(f"{ROLE_SUBJECT}{m['poss'] or ''} {rest}", taught)
    if found is None or found[0] != ROLE_SUBJECT or found[1] in ("name", "named", "called"):
        return None
    _, relation, value, rule, _ = found
    return m["role"].lower(), relation, value, rule.replace("<thing>", "my <role>").replace("<Name>", "my <role>"), None


def _procedure(sentence):
    for pattern, rule in [(r"^(?:to|in order to)\s+(?P<task>[a-z][^,]{2,60}),\s*(?P<steps>.+?)" + END, "\"To <task>, <steps>\""),
                          (r"^how to\s+(?P<task>[^:]{2,60}):\s*(?P<steps>.+?)" + END, "\"How to <task>: <steps>\""),
                          (r"^the (?:best )?way to\s+(?P<task>.+?)\s+is\s+(?:to\s+)?(?P<steps>.+?)" + END,
                           "\"The way to <task> is <steps>\"")]:
        m = re.match(pattern, sentence, re.I)
        if m:
            return _clean(m["task"]).lower(), _clean(m["steps"]), rule
    return None


# ------------------------------------------------------------------------ reading
def read(message):
    """Everything that can be learned from one of your messages. A negated sentence is read as its positive
    statement and every observation from it is marked negated (D-075); other sentences exactly as before."""
    text = re.sub(r"\s+", " ", message).strip()
    forget = FORGET_RE.match(text)
    if forget and not text.endswith("?"):
        return Reading(forget=forget["rest"].strip(), note="you asked to forget something")
    parts = [strip_trailing(p) for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    positives = [negation.positive(p) for p in parts]
    if not any(positives):
        return _read(text)
    reading = Reading()
    for part, positive in zip(parts, positives):
        if positive is None:
            reading.observations += _read(part).observations
            continue
        for o in _read(positive).observations:
            o.negated, o.correction, o.value_named = True, False, False
            o.text = _polish(part)                      # your words, as you said them
            o.reason += " (negated: you said it is NOT so)"
            reading.observations.append(o)
    if not reading.observations:
        reading.note = ("nothing to learn: questions, small talk and passing remarks are not learned (say \"learn "
                        "that ...\" to teach something)")
    return reading


NEGATIVE_VALUE = re.compile(r"^(?:no|not|none)\b", re.I)


def _read(message):
    """The rules for one message without negation."""
    text = re.sub(r"\s+", " ", message).strip()
    reading = Reading()

    # 1. about you: exactly what the memory rules found (one source of truth for personal facts)
    for c in analyze(message).candidates:
        if c.subject != "user" or not c.value or c.type == "episodic":
            continue
        correction = bool(CORRECTION_RE.match(text))
        if c.key in PERSONAL:
            _, relation, single = PERSONAL[c.key]
            how = re.search(r"\b(?:born|works)\s+(in|on|at|as|for)\b", c.summary)
            if how and relation in ("was born", "works"):         # "was born in Paris", "works at the bakery"
                relation = f"{relation} {how.group(1)}"
            obs = Observation("about_you", "you", "you", relation, c.value, c.content, "user_stated", c.confidence,
                              c.reason, single=single, correction=correction, value_named=_is_named(c.value))
        elif c.key.startswith("rel:"):
            role = c.key[4:]
            obs = Observation("about_you", f"my {role}", "role", "name", c.value, c.content, "user_stated",
                              c.confidence, c.reason, role=role, single=True, correction=correction, value_named=True)
        elif c.key.startswith("pref:favorite:"):
            obs = Observation("about_you", "you", "you", f"favorite {c.key.split(':', 2)[2]}", c.value, c.content,
                              "user_stated", c.confidence, c.reason, single=True, correction=correction)
        elif c.key.startswith("pref:"):
            likes = "dislikes" if re.search(r"\b(?:don't|do not|dont|never|hate|hates|dislike|dislikes|can't stand|"
                                            r"cannot stand|detest)\b",
                                            c.content, re.I) else "likes"
            obs = Observation("about_you", "you", "you", likes, c.value, c.content, "user_stated", c.confidence, c.reason)
        elif c.key.startswith("user.") and c.key[5:].replace("_", " ") in RELATIONS:
            continue                     # "My sister is 12 years old": about your sister (read in step 2)
        elif c.key.startswith("user."):
            obs = Observation("about_you", "you", "you", c.key[5:].replace("_", " "), c.value, c.content,
                              "user_stated", c.confidence, c.reason, single=True, correction=correction)
        else:
            continue
        reading.observations.append(obs)

    # 2. the world and skills: sentence by sentence
    for raw in re.split(r"(?<=[.!?])\s+", text):
        s = strip_trailing(raw.strip())
        did_you_know = re.match(r"^did you know\b", s, re.I)
        if not s or (s.endswith("?") and not did_you_know):
            continue                     # a question ("Did you know that ...?" teaches, though)
        s = s.rstrip("?")
        correction = bool(CORRECTION_RE.match(s))
        while LEADING_WORDS.match(s):
            s = LEADING_WORDS.sub("", s)
        taught = False
        teach = TEACH_RE.match(s)
        if teach:
            taught, s = True, teach["rest"].strip()
        s = re.sub(r"^that\s+", "", s, flags=re.I)
        hedged = bool(HEDGE_RE.search(s))
        if hedged:                       # "I think the capital of X is Y": unsure, but about the world
            s = re.sub(r"^\s*(?:that\s+)?", "", HEDGE_RE.sub("", s).strip(" ,"), flags=re.I)
        words = re.findall(r"[a-z']+", s.lower())
        if not words or words[0] in QUESTION_START:
            continue
        source = "user_taught" if taught else "user_stated"
        confidence = CONFIDENCE["taught" if taught else "stated"] - (HEDGE_PENALTY if hedged else 0.0)
        role = _about_role(s, taught)
        if role is not None:             # "My sister lives in Zorbia.": about your sister
            name, relation, value, rule, owned = role
            reading.observations.append(Observation(
                "about_you", f"my {name}'s {owned}" if owned else f"my {name}", "path" if owned else "role", relation,
                _clean(value), _polish(s), source, confidence,
                f"about your {name}{chr(39) + 's ' + owned if owned else ''}: matched the rule {rule}" +
                (" (you sounded unsure)" if hedged else ""), role=name, path=owned,
                single=relation in SINGLE, correction=correction, value_named=_is_named(value)))
            continue
        s = re.sub(r"^i'd\s+like\b", "I would like", s, flags=re.I)
        mine = re.match(r"^i\s+(?P<verb>" + "|".join(FIRST_PERSON_VERBS) + r")\s+(?P<val>.+?)" + END, s, re.I)
        if mine and mine["verb"].lower() in WISHES and NOT_A_THING.match(mine["val"]):
            mine = None
        if mine and not TIME_RE.search(s) and not any(o.category == "about_you" for o in reading.observations):
            reading.observations.append(Observation(
                "about_you", "you", "you", FIRST_PERSON_VERBS[mine["verb"].lower()], _clean(mine["val"]), _polish(s),
                source, CONFIDENCE["you_stated"] - (HEDGE_PENALTY if hedged else 0.0),
                f"about you: matched the rule \"I {mine['verb'].lower()} <...>\"", correction=correction,
                value_named=_is_named(mine["val"])))
            continue
        if ABOUT_SYSTEM.match(s) or re.match(r"^(?:i|i'm|im|my|me|we|our)\b", s, re.I):
            continue                     # about the system: not learned; about you: handled by step 1
        s_core = s
        procedure = _procedure(s_core)
        if procedure:
            task, steps, rule = procedure
            reading.observations.append(Observation(
                "skill", task, "concept", "how to", steps, _polish(s_core), source, confidence,
                f"a procedure: matched the rule {rule}" + (" (taught)" if taught else ""), single=True,
                correction=correction))
            continue
        world = _world(s_core, taught)
        if world is None:
            continue
        subject, relation, value, rule, _ = world
        subject = _concept(subject)
        if subject.lower() in {"it", "this", "that", "there", "he", "she", "they", "we", "everything", "something"}:
            continue
        reason = f"a world fact: matched the rule {rule}" + (" (taught: \"learn that ...\")" if taught else "")
        if hedged:
            reason += " (you sounded unsure, so confidence is lower)"
        reading.observations.append(Observation(
            "taught", subject, _kind(subject), relation, _clean(value), _polish(s_core), source, confidence, reason,
            single=relation in SINGLE, correction=correction, value_named=_is_named(value)))
    # a value that starts with a negation is never a thing you have or are ("I have no idea"): D-075
    reading.observations = [o for o in reading.observations if not NEGATIVE_VALUE.match(o.value)]
    if not reading.observations:
        reading.note = ("nothing to learn: questions, small talk and passing remarks are not learned (say \"learn "
                        "that ...\" to teach something)")
    return reading
