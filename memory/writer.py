"""
Memory formation: is a message worth remembering, and as what?

Only YOUR messages are ever analyzed. The model's replies are never read here, so its guesses
can never become memories (and memory/store.py refuses model-generated text anyway).

What becomes a memory: first-person statements of lasting information.
    fact          "My name is Etai."  "I live in Georgia."  "I'm 15."  "My birthday is May 3."
    relationship  "I have a dog named Max."  "My sister's name is Ana."  "Max is my dog."
    preference    "I like basketball."  "I don't like rain."  "My favorite subject is history."
    episodic      "I won my game today."  "I have a test tomorrow."  (an event, with a time or an event verb)

What does not: questions, greetings and small talk, requests ("tell me a story"), arithmetic,
passing states ("I'm tired"), and statements about the system itself ("you are ..."), unless
you explicitly ask: "Remember that ..." always stores (with higher importance), and
"Forget that ..." removes the matching memory.

Small spelling slips in "favorite" are corrected before the rules are applied ("favoite", "fav",
"fave", "favourite" all count as "favorite"), and lowercase values ("green", "pizza") are accepted
wherever the sentence clearly states a fact or a preference. The general "my X is Y" rule stays strict
(a capitalized name or a number), unless X is a known kind of personal fact (hobby, job, eye color, ...),
so "my phone is broken" is still not stored.

These are plain, readable rules, chosen by us for now because the brain cannot yet judge this
itself. Every stored memory records which rule stored it (its "reason"), shown in the chat's
memory panel. Nothing here writes a reply or says anything about what the system is.
"""

import re
from dataclasses import dataclass, field

from memory import negation

BASE_IMPORTANCE = {"fact": 0.7, "relationship": 0.6, "preference": 0.5, "episodic": 0.4}
EXPLICIT_BONUS = 0.3
CONFIDENCE = {"stated": 0.9, "requested": 0.95, "hedged": 0.6}

NAME = r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*)?"          # Etai, etai, Mary Ann
CAP_NAME = r"(?-i:[A-Z][A-Za-z'\-]*)"         # a capitalized name only, even inside case-insensitive rules
REL = r"[a-z]+(?:\s[a-z]+)?"                                         # dog, best friend
END = r"\s*[.!]*$"

QUESTION_START = {"what", "who", "whom", "whose", "where", "when", "why", "how", "which", "is", "are", "am",
                  "was", "were", "do", "does", "did", "can", "could", "will", "would", "should", "shall",
                  "may", "might", "have", "has", "had", "whats", "what's", "who's", "where's", "hows", "how's"}
REQUEST_START = {"tell", "say", "repeat", "spell", "count", "calculate", "compute", "add", "subtract", "give",
                 "show", "write", "explain", "help", "let's", "lets", "stop", "continue", "describe", "name",
                 "list", "make", "sing", "read", "translate", "hi", "hello", "hey", "thanks", "thank", "bye",
                 "goodbye", "good", "ok", "okay", "yes", "no", "yeah", "nope", "sure", "hmm", "cool", "nice", "wow"}
NOT_A_NAME = {"Fine", "Good", "Ok", "Okay", "Here", "Back", "Sorry", "Ready", "Done", "Sure", "Tired", "Hungry",
              "Bored", "Busy", "Happy", "Sad", "Home", "Late", "Early", "Well", "Great", "Alright", "Not", "No"}

REMEMBER_RE = re.compile(r"^(?:please\s+)?(?:remember|don't forget|do not forget|keep in mind|make a note)"
                         r"\b\s*(?:that\s+|this\s*:\s*|this\s+|:\s*)?(?P<rest>.+)$", re.I)
FORGET_RE = re.compile(r"^(?:please\s+)?(?:forget|delete|erase|stop remembering)\s+(?:about\s+|that\s+|"
                       r"the memory (?:that|about)\s+|what i said about\s+)?(?P<rest>.+?)\s*[.!]*$", re.I)
LEADING_WORDS = re.compile(r"^(?:correction|update|i was wrong|i meant|wait|no|nope|actually|sorry|well|oh|ok|okay|"
                           r"yes|yeah|also|and|but|so|by the way)\s*[,:!.]?\s+", re.I)
CORRECTION_RE = re.compile(r"^(?:correction|update|i was wrong|i meant|no|nope|actually|sorry)\b", re.I)
HEDGE_RE = re.compile(r"\b(i think|maybe|probably|perhaps|i guess|not sure|might)\b", re.I)
TIME_RE = re.compile(r"\b(today|yesterday|tomorrow|tonight|this (?:morning|afternoon|evening|week|weekend|month|year)"
                     r"|last (?:night|week|weekend|month|year|summer|winter)|next (?:week|weekend|month|year)"
                     r"|on (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", re.I)
EVENT_START = re.compile(r"^(?:i|we)\s+(?:just\s+|finally\s+|recently\s+)?(?:won|lost|passed|failed|finished|started|"
                         r"moved|met|bought|visited|graduated|broke|got|went|made|built|learned|joined|quit|"
                         r"have a|have an|am going|'m going|will)\b", re.I)
PASSING_STATE = re.compile(r"^(?:i am|i'm|im)\s+(?:so\s+|very\s+|really\s+|a bit\s+|kind of\s+|feeling\s+)?"
                           r"[a-z]+(?:\s+(?:today|now|right now|tonight))?" + END, re.I)
ABOUT_SYSTEM = re.compile(r"^(?:you|you're|youre|your|yours)\b", re.I)


@dataclass
class Candidate:
    type: str
    content: str            # your own words (a cleaned sentence)
    summary: str            # the memory panel's line, e.g. "User's name: Etai"
    key: str                # what it is about; a newer memory with the same key replaces the older one
    subject: str
    entities: list
    source: str             # user_stated | user_requested
    confidence: float
    importance: float
    reason: str
    value: str = ""         # the remembered value itself ("Etai", "green"), for the learning layer


@dataclass
class Negation:
    """A negated statement ("My dog's name is not Max", "I don't have a dog"): nothing is stored from it, but the
    memories it contradicts are set aside (memory/system.py). summary: the memory line the POSITIVE statement would
    have had ("Dog's name: Max"): a memory with the same key and this summary is contradicted. None: possession is
    denied ("I don't have a dog"), so every memory with the key (rel:dog) is contradicted."""
    sentence: str
    key: str
    summary: str = None


@dataclass
class Analysis:
    candidates: list = field(default_factory=list)
    forget: str = None      # what to forget, if this message asks to forget something
    explicit: bool = False  # the message asked to remember
    note: str = ""          # why nothing was stored (shown in the chat details)
    negations: list = field(default_factory=list)     # negated statements (D-075)


VERB_FORMS = {"like": "likes", "love": "loves", "enjoy": "enjoys", "adore": "adores", "prefer": "prefers"}
FAVORITE_WORDS = {"favorite", "favourite", "fav", "fave", "favs", "faves", "favorit", "favourit"}
# personal facts whose value is often lowercase ("my hobby is painting"): only for these may the general
# "my X is Y" rule store a lowercase value
PERSONAL_ATTRIBUTES = {"hobby", "hobbies", "job", "work", "profession", "major", "nickname", "middle name",
                       "last name", "surname", "language", "native language", "first language", "team", "eye color",
                       "eye colour", "eyes", "hair", "hair color", "hair colour", "height", "grade", "school",
                       "hometown", "home town", "city", "town", "country", "state", "zodiac sign", "star sign"}
RELATIONS = {"dog", "cat", "puppy", "kitten", "bird", "parrot", "fish", "hamster", "rabbit", "horse", "turtle",
             "snake", "lizard", "pony", "pet", "mom", "mother", "mum", "dad", "father", "sister", "brother", "son",
             "daughter", "wife", "husband", "friend", "best friend", "boyfriend", "girlfriend", "grandma",
             "grandmother", "grandpa", "grandfather", "aunt", "uncle", "cousin", "teacher", "boss", "neighbor",
             "neighbour", "partner", "coach", "roommate"}


def _edit_distance(a, b):
    """How many single-letter changes turn a into b (insert, delete or replace)."""
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def fix_favorite(text):
    """Read small slips of "favorite" as "favorite". Returns (text, the word that was corrected or None)."""
    fixed = []

    def repl(m):
        word = m.group(0)
        low = word.lower()
        if low == "favorite":
            return word
        if low in FAVORITE_WORDS or (len(low) >= 6 and low[:2] == "fa" and
                                     min(_edit_distance(low, "favorite"), _edit_distance(low, "favourite")) <= 1):
            fixed.append(word)
            return ("F" if word[0].isupper() else "f") + "avorite"
        return word
    out = re.sub(r"\b[A-Za-z]+\b", repl, text)
    return out, (fixed[0] if fixed else None)


# ------------------------------------------------------------------ the patterns
def _clean(v):
    return v.strip().rstrip(".!").strip()


def _name(v):
    return " ".join(w[:1].upper() + w[1:] for w in _clean(v).split())


# Each rule: (type, the rule in plain words, regex, key maker, summary maker).
# The FIRST rule that matches a sentence decides. Order matters: specific rules come before general ones.
def _fav_key(m):
    return f"pref:favorite:{m['cat'] or _clean(m['v']).lower()}"


PATTERNS = [
    # facts about you
    ("fact", '"my name is <name>"',
     r"^(?:my name is|my name's|my names is|i am called|i'm called|call me|people call me|you can call me|"
     r"everyone calls me)\s+(?P<v>" + NAME + r")\b",
     lambda m: "user.name", lambda m: f"User's name: {_name(m['v'])}"),
    ("fact", '"I am <age>"', r"^(?:i am|i'm|im)\s+(?P<v>\d{1,3})(?:\s+years?\s+old|\s+yrs?\s+old)?" + END,
     lambda m: "user.age", lambda m: f"User's age: {m['v']}"),
    ("fact", '"I live in <place>"', r"^i\s+(?:live|am living|'m living|currently live|now live)\s+in\s+(?P<v>.+?)" + END,
     lambda m: "user.location", lambda m: f"User lives in: {_clean(m['v'])}"),
    ("fact", '"I\'m from <place>"', r"^(?:i am|i'm|im)\s+from\s+(?P<v>.+?)" + END,
     lambda m: "user.origin", lambda m: f"User is from: {_clean(m['v'])}"),
    ("fact", '"I was born in/on <...>"', r"^i\s+was\s+born\s+(?P<how>in|on)\s+(?P<v>.+?)" + END,
     lambda m: "user.born", lambda m: f"User was born {m['how']}: {_clean(m['v'])}"),
    ("fact", '"my birthday is <date>"', r"^my\s+birthday\s+is\s+(?:on\s+)?(?P<v>.+?)" + END,
     lambda m: "user.birthday", lambda m: f"User's birthday: {_clean(m['v'])}"),
    ("fact", '"I work as/at <...>"', r"^i\s+work\s+(?P<how>as|at|for|in)\s+(?P<v>.+?)" + END,
     lambda m: "user.work", lambda m: f"User works {m['how']}: {_clean(m['v'])}"),
    ("fact", '"I study <...>"', r"^i\s+(?:study|am studying|'m studying)\s+(?P<v>.+?)" + END,
     lambda m: "user.study", lambda m: f"User studies: {_clean(m['v'])}"),
    ("fact", '"I go to <...>"', r"^i\s+go\s+to\s+(?P<v>.+?)" + END,
     lambda m: "user.goes_to", lambda m: f"User goes to: {_clean(m['v'])}"),
    # favorites (before the "<Name> is my ..." rule, so "Pizza is my favorite food" is not read as a name)
    ("preference", '"my favorite <thing> is <...>"',
     r"^my\s+favou?rite\s+(?P<cat>[a-z]+(?:\s[a-z]+){0,2})\s+(?:is|are)\s+(?P<v>.+?)" + END,
     lambda m: f"pref:favorite:{m['cat']}", lambda m: f"User's favorite {m['cat']}: {_clean(m['v'])}"),
    ("preference", '"favorite <thing>: <...>"',
     r"^(?:my\s+)?favorite\s+(?P<cat>[a-z]+(?:\s[a-z]+){0,2})\s*[:=\-]\s*(?P<v>.+?)" + END,
     lambda m: f"pref:favorite:{m['cat']}", lambda m: f"User's favorite {m['cat']}: {_clean(m['v'])}"),
    ("preference", '"<...> is my favorite"',
     r"^(?P<v>.+?)\s+(?:is|are)\s+my\s+favou?rite(?:\s+(?P<cat>[a-z]+(?:\s[a-z]+)?))?" + END,
     _fav_key, lambda m: f"User's favorite{' ' + m['cat'] if m['cat'] else ''}: {_clean(m['v'])}"),
    # people, pets and things with names
    ("relationship", '"I have a <pet/person> named <name>"',
     r"^i\s+(?:have|got|own|have got)\s+(?:a|an|one|two|three|\d+)\s+(?P<rel>" + REL + r")\s+(?:named|called)\s+"
     r"(?P<v>" + NAME + r")\b",
     lambda m: f"rel:{m['rel']}", lambda m: f"{m['rel'].capitalize()}'s name: {_name(m['v'])}"),
    ("relationship", '"my <pet/person>\'s name is <name>"',
     r"^my\s+(?P<rel>" + REL + r")(?:'s|s)\s+name\s+is\s+(?P<v>" + NAME + r")\b",
     lambda m: f"rel:{m['rel']}", lambda m: f"{m['rel'].capitalize()}'s name: {_name(m['v'])}"),
    ("relationship", '"my <pet/person> is called <name>"',
     r"^my\s+(?P<rel>" + REL + r")\s+is\s+(?:called|named)\s+(?P<v>" + NAME + r")\b",
     lambda m: f"rel:{m['rel']}", lambda m: f"{m['rel'].capitalize()}'s name: {_name(m['v'])}"),
    ("relationship", '"my <pet/person> is <Name>"',
     r"^my\s+(?P<rel>" + "|".join(sorted(RELATIONS, key=len, reverse=True)) + r")\s+is\s+(?P<v>" + CAP_NAME + r")" + END,
     lambda m: f"rel:{m['rel'].lower()}", lambda m: f"{m['rel'].capitalize()}'s name: {_name(m['v'])}"),
    ("relationship", '"<Name> is my <pet/person>"', r"^(?P<v>" + CAP_NAME + r")\s+is\s+my\s+(?P<rel>" + REL + r")" + END,
     lambda m: f"rel:{m['rel']}", lambda m: f"{m['rel'].capitalize()}'s name: {_name(m['v'])}"),
    # likes and dislikes
    ("preference", '"I don\'t like <...>"',
     r"^i\s+(?:really\s+|also\s+|just\s+)?(?:don't|do not|dont|never)\s+(?:like|enjoy|love)\s+(?P<v>.+?)" + END,
     lambda m: f"pref:{_clean(m['v']).lower()}", lambda m: f"User doesn't like {_clean(m['v'])}"),
    ("preference", '"I hate <...>"', r"^i\s+(?:really\s+)?(?:hate|dislike|can't stand|cannot stand|detest)\s+(?P<v>.+?)" + END,
     lambda m: f"pref:{_clean(m['v']).lower()}", lambda m: f"User doesn't like {_clean(m['v'])}"),
    ("preference", '"I like/love/enjoy <...>"',
     r"^i\s+(?:really\s+|also\s+|do\s+|just\s+)?(?P<verb>like|love|enjoy|adore|prefer)\s+(?P<v>.+?)" + END,
     lambda m: f"pref:{_clean(m['v']).lower()}",
     lambda m: f"User {VERB_FORMS[m['verb'].lower()]} {_clean(m['v'])}"),
    # known personal facts may have lowercase values ("my hobby is painting", "my eye color is brown")
    ("fact", '"my <hobby/job/eye color/...> is <...>"',
     r"^my\s+(?P<attr>" + "|".join(sorted(PERSONAL_ATTRIBUTES, key=len, reverse=True)) + r")\s+(?:is|are)\s+"
     r"(?P<v>.+?)" + END,
     lambda m: f"user.{m['attr'].lower().replace(' ', '_')}", lambda m: f"User's {m['attr'].lower()}: {_clean(m['v'])}"),
    # the general "my X is Y" last. Only a NAMED value or a number ("My school is Lincoln High",
    # "My number is 7"): "my phone is broken" is a passing state and is not stored.
    ("fact", '"my <thing> is <Name or number>"',
     r"^my\s+(?P<attr>[a-z]+(?:\s[a-z]+){0,2})\s+(?:is|are)\s+(?P<v>(?-i:[A-Z0-9]).*?)" + END,
     lambda m: f"user.{m['attr'].replace(' ', '_')}", lambda m: f"User's {m['attr']}: {_clean(m['v'])}"),
]
PATTERNS = [(t, label, re.compile(p, re.I), key, summary) for t, label, p, key, summary in PATTERNS]
I_AM_NAME = re.compile(r"^(?:i am|i'm|im)\s+(?P<v>" + CAP_NAME + r")" + END, re.I)   # "I'm tired" is not a name


# ------------------------------------------------------------------ analysis
# a conversational word at the end of a sentence says nothing about the fact (D-079): "I want a dog though",
# "I live in Paris anyway". Both readers (memory, learning) ignore it.
TRAILING_WORDS = re.compile(r"(?:[\s,]+(?:though|anyway|anyways|actually|too|as well))+(?=[\s.!]*$)", re.I)


def strip_trailing(sentence):
    """ "I want a dog though." -> "I want a dog." """
    return TRAILING_WORDS.sub("", sentence)


def sentences(text):
    """Split into sentences, and split "X and I/my Y" into two statements."""
    parts = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        parts += re.split(r",?\s+and\s+(?=(?:i|i'm|my)\s)", s, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


def _polish(s):
    s = s.strip()
    s = s[:1].upper() + s[1:]
    return s if s[-1:] in ".!?" else s + "."


def _entities(sentence, match):
    found = set(re.findall(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z'\-]+\b", sentence)) - {"I", "My", "I'm"}
    for group in ("v", "rel", "cat"):
        value = match.groupdict().get(group) if match else None
        if value:
            found.add(_clean(value))
    return sorted(found)


def classify(sentence, explicit=False):
    """One sentence -> a Candidate, or (None, why not)."""
    s = LEADING_WORDS.sub("", sentence.strip())
    while LEADING_WORDS.match(s):
        s = LEADING_WORDS.sub("", s)
    words = re.findall(r"[a-z']+", s.lower())
    if not words:
        return None, "no words"
    if s.endswith("?") or (words[0] in QUESTION_START and not explicit):
        return None, "a question (questions are not stored)"
    if re.search(r"\d\s*[-+*/x=]\s*\d", s):
        return None, "arithmetic (not stored)"
    if ABOUT_SYSTEM.match(s) and not explicit:
        return None, "about the system itself (stored only if you say \"remember\")"
    source = "user_requested" if explicit else "user_stated"
    confidence = CONFIDENCE["hedged"] if HEDGE_RE.search(s) else CONFIDENCE["requested" if explicit else "stated"]
    s_for_match = HEDGE_RE.sub("", s).strip(" ,") if HEDGE_RE.search(s) else s
    s_for_match = re.sub(r"^that\s+", "", s_for_match, flags=re.I)
    s_for_match, corrected = fix_favorite(s_for_match)

    def make(type, key, summary, match, rule):
        importance = min(1.0, BASE_IMPORTANCE[type] + (EXPLICIT_BONUS if explicit else 0.0))
        reason = rule + (" + you asked to remember it" if explicit else "")
        if corrected:
            reason += f" (\"{corrected}\" read as \"favorite\")"
        if confidence == CONFIDENCE["hedged"]:
            reason += " (you sounded unsure, so confidence is lower)"
        value = _clean(match["v"]) if match is not None and "v" in match.groupdict() and match["v"] else ""
        return Candidate(type, _polish(s_for_match), summary, key, "user", _entities(s_for_match, match), source,
                         confidence, importance, reason, value), ""

    if ABOUT_SYSTEM.match(s):              # only reached when explicit: "remember that you ..." (kept as you said it)
        candidate, _ = make("fact", None, f"Said about the system: {_polish(s_for_match)}", None,
                            "a statement about the system")
        candidate.subject = "system"
        return candidate, ""
    for type, label, pattern, key, summary in PATTERNS:
        m = pattern.match(s_for_match)
        if not m:
            continue
        if type == "fact" and key(m) == "user.name" and m["v"].split()[0].lower() in {"not", "no"}:
            return None, "a negative statement (not stored)"
        if type == "preference" and not explicit and \
                re.fullmatch(r"you|u|yourself|talking to you", _clean(m["v"]).lower()):
            return None, "about the system itself (stored only if you say \"remember\")"
        return make(type, key(m), summary(m), m, f"a {type}: matched the rule {label}")
    m = I_AM_NAME.match(s_for_match)
    if m and m["v"] not in NOT_A_NAME:
        return make("fact", "user.name", f"User's name: {m['v']}", m, "a fact: matched the rule \"I am <Name>\"")
    if PASSING_STATE.match(s_for_match) and not explicit:
        return None, "a passing state (not stored)"
    if (TIME_RE.search(s_for_match) or EVENT_START.match(s_for_match)) and \
            re.match(r"^(?:i|we|my|i'm|im)\b", s_for_match, re.I):
        when = TIME_RE.search(s_for_match)
        return make("episodic", None, f"Event: {_polish(s_for_match)}", when,
                    "an event you described" + (f" (\"{when.group(0)}\")" if when else ""))
    if explicit:
        type = "episodic" if TIME_RE.search(s_for_match) else "fact"
        return make(type, None, f"Remembered: {_polish(s_for_match)}", None, "you asked to remember it")
    if words[0] in REQUEST_START:
        return None, "small talk or a request (not stored)"
    return None, "no lasting information found (temporary conversation)"


def analyze(message):
    """What should memory do with this message? Returns an Analysis."""
    text = re.sub(r"\s+", " ", message).strip()
    forget = FORGET_RE.match(text)
    if forget and not text.endswith("?"):
        rest = forget["rest"].strip()
        if rest.lower() in {"it", "that", "this", "everything", "all", "all of it", "everything i said"}:
            return Analysis(note="say what to forget, e.g. \"forget that my dog's name is Max\" "
                                 "(to delete everything, use \"Clear all memories\" in the memory panel)")
        return Analysis(forget=rest, note="you asked to forget something")
    explicit = False
    remember = REMEMBER_RE.match(text)
    if remember and not text.endswith("?"):
        explicit, text = True, remember["rest"].strip()
    result, reasons = Analysis(explicit=explicit), []
    seen_keys = set()
    for sentence in sentences(text):
        sentence = strip_trailing(sentence)
        positive = negation.positive(sentence)
        if positive is not None:              # "not Max" is never a name: nothing is stored (D-075)
            found, _ = classify(positive, explicit)
            owned = re.match(r"^i\s+(?:have|own|have got)\s+(?:a|an|one|any)\s+(?P<rel>" + REL + r")s?" + END,
                             positive, re.I)
            if owned:
                result.negations.append(Negation(sentence, f"rel:{owned['rel'].lower()}"))
            elif found is not None and found.key:
                result.negations.append(Negation(sentence, found.key, found.summary))
            reasons.append("a negative statement: nothing stored (memories it contradicts are set aside)")
            continue
        candidate, why_not = classify(sentence, explicit)
        if candidate is None:
            reasons.append(why_not)
            continue
        if candidate.key and candidate.key in seen_keys:
            continue
        seen_keys.add(candidate.key)
        result.candidates.append(candidate)
    if not result.candidates:
        result.note = reasons[0] if reasons else "nothing to store"
    return result
