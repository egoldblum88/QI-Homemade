"""
Negated statements: "I don't have a dog", "My dog's name is not Max", "I am not from Paris", "I have no car",
"Max isn't my dog", "My sister doesn't live in Rome", "I no longer live in Paris", "I don't live there anymore".

    positive(sentence) -> the same statement without the negation ("I have a dog"), or None when the sentence is not
                          a negated statement

Memory and learning read the positive statement with their usual rules, and then record that YOU SAID IT IS NOT SO.
Before this (D-075), a negation was either ignored (and whatever it contradicted stayed "true") or read backwards:
"My dog's name is not Max" stored the name "not Max"; "I have no dog" stored that you own "no dog".

English grammar only: no content, no list of things, nothing about any particular fact.
Not read as a negation here:
  - "I don't like / love / enjoy X": that is a dislike, read by the preference rules (as before);
  - "never", and the past ("I didn't have a dog"): they say nothing clear about now;
  - questions, and idioms that are not statements about a thing ("I have no idea").
"""

import re

PREFERENCE_VERBS = {"like", "love", "enjoy", "adore", "prefer", "mind", "hate", "dislike"}
NOT_STATEMENTS = {"idea", "clue", "doubt", "choice", "reason", "way", "chance", "comment", "problem", "problems",
                  "time", "plans", "worries"}
FIRST_PERSON = {"i", "you", "we", "they"}


def third_person(verb):
    """ "live" -> "lives", "watch" -> "watches", "study" -> "studies", "have" -> "has" (for "my sister doesn't ...")."""
    if verb == "have":
        return "has"
    if verb in ("do", "go"):
        return verb + "es"
    if re.search(r"(?:s|sh|ch|x|z)$", verb):
        return verb + "es"
    if re.search(r"[^aeiou]y$", verb):
        return verb[:-1] + "ies"
    return verb + "s"


def _article(noun_phrase):
    return "an" if re.match(r"[aeiou]", noun_phrase.lower()) else "a"


def positive(sentence):
    """The statement without its negation, or None (see the module docstring)."""
    s = re.sub(r"\s+", " ", sentence or "").strip()
    if not s or s.endswith("?"):
        return None
    s = s.rstrip(" .!")
    s = re.sub(r"\s+(?:any ?more)$", "", s, flags=re.I)
    m = re.match(r"^(?P<subj>.+?)\s+no longer\s+(?P<rest>.+)$", s, re.I)
    if m:                                                           # "I no longer live in Paris"
        return f"{m['subj']} {m['rest']}."
    m = re.match(r"^(?P<subj>.+?)\s+(?:do not|don't|dont|does not|doesn't|doesnt)\s+(?P<verb>[a-z]+)\b(?P<rest>.*)$",
                 s, re.I)
    if m:                                                           # "I don't have a dog", "she doesn't live in Rome"
        verb = m["verb"].lower()
        if verb in PREFERENCE_VERBS:
            return None
        rest = re.sub(r"^\s+any\s+", f" {_article('x')} ", m["rest"], flags=re.I) if verb in ("have", "own") else m["rest"]
        if verb in ("have", "own"):
            noun = rest.strip().split(" ")[-1].lower() if rest.strip() else ""
            if noun in NOT_STATEMENTS or not rest.strip():
                return None
        subject = m["subj"]
        verb = verb if subject.lower() in FIRST_PERSON else third_person(verb)
        return f"{subject} {verb}{rest}."
    m = re.match(r"^(?P<subj>.+?)\s+(?P<have>have|has|own|owns)\s+no\s+(?P<thing>.+)$", s, re.I)
    if m:                                                           # "I have no car"
        if m["thing"].split()[-1].lower() in NOT_STATEMENTS:
            return None
        return f"{m['subj']} {m['have']} {_article(m['thing'])} {m['thing']}."
    m = re.match(r"^(?P<subj>.+?)(?:\s+(?P<be>am|is|are|was|were)\s+not|\s+(?P<neg>isn't|isnt|aren't|arent|wasn't|"
                 r"wasnt|weren't|werent)|(?P<short>'m|'s|'re)\s+not)\s+(?P<rest>.+)$", s, re.I)
    if m:                                                           # "My dog's name is not Max", "I'm not from Paris"
        if m["be"]:
            be = " " + m["be"]
        elif m["neg"]:
            be = " " + {"isn't": "is", "isnt": "is", "aren't": "are", "arent": "are", "wasn't": "was", "wasnt": "was",
                        "weren't": "were", "werent": "were"}[m["neg"].lower()]
        else:
            be = m["short"]
        return f"{m['subj']}{be} {m['rest']}."
    return None
