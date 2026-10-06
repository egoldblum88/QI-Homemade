"""
Retrieval: which memories are relevant to a new message?

Every memory gets a relevance score from three signals (each between 0 and 1):
    words    how much of the message's meaningful words the memory contains, rarer words counting
             more (like a search engine's "inverse document frequency"). A few word groups count
             as the same word: like / love / enjoy / favorite; live / home; dog / cat -> pet
    names    an exact name or thing from the memory appears in the message ("Is Max hungry?")
    kind     the message asks about this kind of thing: "what is my name / who am I" -> your
             name; "where do I live" -> your home; "what do I like" -> preferences; "what
             happened" -> events

    relevance = 0.55 * words + 0.35 * names + 0.45 * kind          (at most 1)

A question about the SYSTEM ("what is your name?") does not retrieve YOUR name: when who the
message is about differs from who the memory is about, relevance is multiplied by 0.3.

Only memories with relevance >= MIN_RELEVANCE, and at least 0.6 times as relevant as the best one,
are kept. Importance (and whether you pinned it) and recency then only ORDER them:
    score = relevance * (0.8 + 0.1 * importance + 0.1 * recency)
so an important but unrelated memory can never enter the prompt. At most MAX_RECALLED are kept
(3: measured, the brain still used a recalled name 94-98% of the time with 3 memories present).

This is word-level relevance written by us. There is no learned "meaning" similarity yet: that
would come from our own networks later, and would be measured before it is used.
"""

import math
import re
import time
from dataclasses import dataclass

MIN_RELEVANCE = 0.30
RELATIVE_CUTOFF = 0.6
MAX_RECALLED = 3
FORGET_RELEVANCE = 0.5          # "forget ..." needs a clear match, and only the best matches are removed
SUBJECT_MISMATCH = 0.3
RECENCY_DAYS = 30

STOPWORDS = set("""a an the and or but if of to in on at for from by with about as into than then so too very just
also again still now here there this that these those it its it's is are was were be been being am do does did
done have has had having can could will would shall should may might must i me my mine myself we us our you your
yours yourself he him his she her they them their what whats what's who whom whose where when why how which
tell say know remember please really any some all no not yes okay ok oh well go goes going get gets got
make makes thing things""".split())
GROUPS = {"love": "like", "loves": "like", "liked": "like", "likes": "like", "enjoy": "like", "enjoys": "like",
          "favorite": "like", "favourite": "like", "prefer": "like", "prefers": "like", "adore": "like",
          "hate": "like", "hates": "like", "dislike": "like", "lives": "live", "living": "live", "home": "live",
          "mom": "mother", "mum": "mother", "dad": "father", "old": "age", "birth": "birthday", "born": "birthday",
          "colour": "color", "fav": "like", "fave": "like"}
EXPANSIONS = {"dog": ["pet"], "cat": ["pet"], "puppy": ["pet", "dog"], "kitten": ["pet", "cat"], "fish": ["pet"],
              "bird": ["pet"], "hamster": ["pet"], "rabbit": ["pet"], "horse": ["pet"], "turtle": ["pet"]}
# What kind of memory a message asks about: (message pattern, memory keys it wants, memory types it wants)
WANTS = [
    (r"\b(?:my name|who am i|what am i called|what do you call me|call me)\b", {"user.name"}, set()),
    (r"\b(?:how old am i|my age)\b", {"user.age"}, set()),
    (r"\b(?:where do i live|where i live|where am i from|where i'm from|my (?:home|city|country|town))\b",
     {"user.location", "user.origin"}, set()),
    (r"\b(?:my birthday|when was i born|when is my birthday|where was i born)\b", {"user.birthday", "user.born"}, set()),
    (r"\b(?:what do i (?:like|love|enjoy|hate|dislike)|what i (?:like|love)|do i like)\b", set(), {"preference"}),
    (r"\b(?:what happened|what did i do|what have i done|remember when)\b", set(), {"episodic"}),
    (r"\b(?:your name|who are you|what are you called)\b", {"system.name"}, set()),
]
WANTS = [(re.compile(p, re.I), keys, types) for p, keys, types in WANTS]
# "my favorite <thing>" wants THAT favorite only (D-063: any favorite used to match, and the brain then answered
# "Your favorite food is green." from a favorite COLOR)
FAVORITE_ASKED = re.compile(r"\bmy\s+(?:favou?rite|fav|fave)\s+([a-z]+)", re.I)


def favorite_category(word):
    return GROUPS.get(word.lower().rstrip("s"), word.lower().rstrip("s"))


@dataclass
class Retrieved:
    memory: object
    relevance: float
    score: float
    parts: dict

    def to_public(self):
        return {**self.memory.to_public(), "relevance": round(self.relevance, 3), "score": round(self.score, 3),
                "parts": {k: round(v, 3) for k, v in self.parts.items()}}


def terms(text, expand=True):
    """Meaningful, normalized words: lowercase, possessives and plural -s removed, word groups joined.

    expand: also add the broader word ("dog" -> "pet"). Used for MEMORIES only, so "what is my pet's name?"
    finds your dog, but "what is my cat's name?" does not (D-063: it did, and the brain then answered with
    the dog's name).
    """
    out = []
    for word in re.findall(r"[a-z0-9']+", text.lower()):
        word = word.strip("'")
        if word.endswith("'s"):
            word = word[:-2]
        if word in STOPWORDS or not word:
            continue
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        word = GROUPS.get(word, word)
        out.append(word)
        if expand:
            out += EXPANSIONS.get(word, [])
    return out


# Ways of ASKING the system something, which are not about the system itself
ASKING = re.compile(r"\b(?:what do you know about|what else do you know about|do you know|do you remember|"
                    r"can you tell me|could you tell me|would you tell me|can you remind me|did i tell you|"
                    r"have i told you|what did i tell you)\b", re.I)


def subject_of(text):
    """Who a message is about: "user" (I / my), "system" (you / your), or "any".

    "What do you know about Max?" and "Do you remember my name?" are not about the system: the asking
    phrase is removed first.
    """
    text = ASKING.sub(" ", text)
    about_user = re.search(r"\b(i|me|my|mine|myself|i'm|im)\b", text, re.I)
    about_system = re.search(r"\b(you|your|yours|yourself|you're)\b", text, re.I)
    if about_system and not about_user:
        return "system"
    if re.search(r"\byour name\b", text, re.I):
        return "system"
    return "user" if about_user else "any"


def retrieve(memories, message, k=MAX_RECALLED, min_relevance=MIN_RELEVANCE, now=None, bias=None):
    """The relevant memories for a message, best first, with their scores.

    bias: {memory id: b} from the internal state (M2, internal_state/system.py). It multiplies the SCORE by
    (1 + b), b <= 0.1, so it only changes the order of memories that already passed the relevance thresholds."""
    if not memories:
        return []
    now = time.time() if now is None else now
    query = set(terms(message, expand=False))
    raw_words = {w.strip("'") for w in re.findall(r"[a-z0-9']+", message.lower())}
    wanted_keys, wanted_types = set(), set()
    for pattern, keys, types in WANTS:
        if pattern.search(message):
            wanted_keys |= keys
            wanted_types |= types
    asked = FAVORITE_ASKED.search(message)
    favorite = favorite_category(asked.group(1)) if asked else None
    subject = subject_of(message)

    memory_terms = [set(terms(m.content)) for m in memories]
    n = len(memories)
    df = {}
    for ts in memory_terms:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    idf = {t: 1.0 + math.log((n + 1) / (df.get(t, 0) + 1)) for t in query}
    query_weight = sum(idf.values())

    found = []
    for memory, mterms in zip(memories, memory_terms):
        words = sum(idf[t] for t in query & mterms) / query_weight if query_weight else 0.0
        names = 1.0 if any(e.lower() in raw_words or e.lower() in query for e in memory.entities) else 0.0
        kind = 1.0 if (memory.key in wanted_keys) else (0.6 if memory.type in wanted_types else 0.0)
        if favorite and memory.key and memory.key.startswith("pref:favorite:"):
            kind = 1.0 if favorite_category(memory.key.split(":", 2)[2]) == favorite else 0.0
        relevance = min(1.0, 0.55 * words + 0.35 * names + 0.45 * kind)
        if subject == "system":                      # "what is your name?" is not about YOUR name
            mismatch = memory.subject != "system"
        else:
            mismatch = subject == "user" and memory.subject == "system"
        if mismatch:
            relevance *= SUBJECT_MISMATCH
        importance = 1.0 if memory.pinned else memory.importance
        recency = math.exp(-max(0.0, now - memory.created) / (RECENCY_DAYS * 86400))
        state = (bias or {}).get(memory.id, 0.0)
        score = relevance * (0.8 + 0.1 * importance + 0.1 * recency) * (1 + state)
        found.append(Retrieved(memory, relevance, score, {"words": words, "names": names, "kind": kind,
                                                          "importance": importance, "recency": recency,
                                                          "subject_mismatch": float(mismatch),
                                                          **({"state_bias": state} if state else {})}))
    best = max((r.relevance for r in found), default=0.0)
    keep = [r for r in found if r.relevance >= min_relevance and r.relevance >= RELATIVE_CUTOFF * best]
    keep.sort(key=lambda r: -r.score)
    return keep[:k]
