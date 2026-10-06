"""
Generated training data for Stage 1.5b: every example is made by this program, and its
correct reply can be worked out from the conversation itself. No AI wrote any of it.

Five categories (see tools/prepare_mix.py for how they are combined with the books):

ARITHMETIC     every a+b (0-99 + 0-99) and a-b (a >= b), in about 20 wordings, including bare
               "10 + 10", "5+5", "WHAT IS 5 + 5?". 10% of the PROBLEMS and 3 whole WORDINGS are
               held out forever, so we can test real generalization, not string memorization.
               (The examples you asked for, like 0+0, 5+5, 10+10, 12+13, 20+25, are always trained on.)
TASKS          small jobs whose answer is IN the message: repeat a phrase, recall a name given in the
               message, compare two numbers, count words, the number after/before n, first/last
               word, spell a word. Several start with "what is", so "what is" stops meaning "number".
               Words and names come from our own public-domain books.
CONTEXT        (round 2) the answer is in an EARLIER turn: a name, word or number given before, a
               corrected name ("no, my name is X"), which of two names was asked for, what the last
               message was, or "You have not told me ..." when it was never given. Unrelated turns
               can come in between. Names and words are mostly made up, so only reading works.
MEMORY         (round 3) using information YOU gave earlier, the way long-term memory shows it to the
               brain: favorites, likes and dislikes, pets, family, places, and "You have not told me
               ..." when it was never given. The replies only repeat YOUR information back ("Your
               favorite food is pizza."). Nothing gives the system favorites, opinions or an identity:
               questions about the system ("what is YOUR favorite ...") are never trained.
CONVERSATION   a few neutral conventions: a greeting gets the same greeting back ("good morning" ->
               "Good morning."), thanks -> "You're welcome.", a goodbye -> the same goodbye.
               Deliberately NOT included: names, feelings, opinions ("how are you", "what is your
               name"): the blank-slate rule. Your own conversation files are added to this category.

Nothing here runs in the chat: the chat never computes or looks up an answer (the Originality
Guard checks that). These generators only make training and test data.
"""

import random
import re
from collections import Counter

from brain.dialogue import Example

# ------------------------------------------------------------------------ arithmetic
ADD_FORMATS = ["{a}+{b}", "{a} + {b}", "{a}+{b}=?", "{a} + {b} = ?", "{a} + {b} =",
               "what is {a}+{b}", "what is {a} + {b}", "What is {a}+{b}?", "What is {a} + {b}?",
               "WHAT IS {a} + {b}?", "what's {a}+{b}?", "what's {a} plus {b}?", "What is {a} plus {b}?",
               "{a} plus {b}", "how much is {a} + {b}?", "how much is {a} plus {b}?", "calculate {a} + {b}",
               "add {a} and {b}", "what is the sum of {a} and {b}?", "compute {a}+{b}"]
SUB_FORMATS = ["{a}-{b}", "{a} - {b}", "{a}-{b}=?", "{a} - {b} = ?", "what is {a}-{b}", "what is {a} - {b}",
               "What is {a} - {b}?", "WHAT IS {a} - {b}?", "what's {a} minus {b}?", "What is {a} minus {b}?",
               "{a} minus {b}", "how much is {a} - {b}?", "calculate {a} - {b}", "subtract {b} from {a}",
               "take {b} from {a}"]
# More wordings, for TRAINING only (added in Stage 1.5b round 2). They are kept in their own list so
# the validation problems (made from the lists above) stay exactly the same as before.
ADD_EXTRA_FORMATS = ["{a} + {b}?", "{a}+{b}?", "what is {a} plus {b}", "what's {a} + {b}",
                     "{a} and {b} added together?", "add {b} to {a}", "the sum of {a} and {b}", "{a} + {b} is",
                     "find {a} + {b}", "what do you get when you add {a} and {b}?"]
SUB_EXTRA_FORMATS = ["{a} - {b}?", "{a}-{b}?", "what is {a} minus {b}", "what's {a} - {b}", "{a} take away {b}",
                     "what do you get when you subtract {b} from {a}?", "{a} - {b} is", "find {a} - {b}",
                     "the difference between {a} and {b}?"]
# Wordings that NEVER appear in training: they test whether the skill carries over to new phrasing.
ADD_HELD_OUT_FORMATS = ["{a} plus {b} equals?", "Please add {a} and {b}.", "tell me {a} + {b}"]
SUB_HELD_OUT_FORMATS = ["{a} minus {b} equals?", "Please subtract {b} from {a}.", "tell me {a} - {b}"]
ALWAYS_TRAINED = [("+", 0, 0), ("+", 1, 1), ("+", 1, 2), ("+", 2, 3), ("+", 5, 5), ("+", 10, 10),
                  ("+", 12, 13), ("+", 20, 25)]


def all_problems(max_number=99):
    problems = [("+", a, b, a + b) for a in range(max_number + 1) for b in range(max_number + 1)]
    problems += [("-", a, b, a - b) for a in range(max_number + 1) for b in range(a + 1)]
    return problems


def split_arithmetic(problems, held_out_fraction=0.1, seed=0):
    """(train, held_out): held-out problems are never trained on. ALWAYS_TRAINED stay in training."""
    fixed = {p for p in problems if p[:3] in ALWAYS_TRAINED}
    rest = [p for p in problems if p not in fixed]
    random.Random(seed).shuffle(rest)
    n = round(len(problems) * held_out_fraction)
    return sorted(fixed) + rest[n:], rest[:n]


def arithmetic_examples(problems, formats="train", per_problem=1, seed=0, source="arithmetic"):
    rng = random.Random(seed)
    out = []
    for op, a, b, answer in problems:
        pool = {("+", "train"): ADD_FORMATS, ("-", "train"): SUB_FORMATS,
                ("+", "extended"): ADD_FORMATS + ADD_EXTRA_FORMATS, ("-", "extended"): SUB_FORMATS + SUB_EXTRA_FORMATS,
                ("+", "held_out"): ADD_HELD_OUT_FORMATS, ("-", "held_out"): SUB_HELD_OUT_FORMATS}[op, formats]
        for template in rng.sample(pool, min(per_problem, len(pool))):
            out.append(Example([template.format(a=a, b=b)], str(answer), source))
    return out


# ----------------------------------------------------------------------------- tasks
NOT_NAMES = {"Mrs", "Miss", "Mr", "Sir", "Lady", "Lord", "Madam", "Madame", "Captain", "Uncle", "Aunt",
             "Aunty", "King", "Queen", "Prince", "Princess", "Doctor", "Professor", "Colonel", "Mother",
             "Father", "God", "Heaven", "London", "England", "English", "French", "Christmas", "Sunday",
             "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Phoenix"}


def word_lists(book_text, n_words=800, n_names=240):
    """Common lowercase words and frequent personal names, taken from our own books."""
    lower = Counter(w for w in re.findall(r"\b[a-z]{3,9}\b", book_text))
    words = [w for w, _ in lower.most_common(n_words)]
    # a name: capitalized in the middle of a sentence, (almost) never written in lowercase,
    # and not a title, place or day
    mid = Counter(re.findall(r"(?<=[a-z,] )([A-Z][a-z]{2,8})\b", book_text))
    names = [w for w, c in mid.most_common() if c >= 20 and lower.get(w.lower(), 0) < c // 10
             and w not in NOT_NAMES]
    return words, names[:n_names]


def copy_vocabulary(book_text, reserved=(), min_count=3):
    """Every lowercase word of 3-10 letters used at least 3 times in our books (about 15,000),
    minus `reserved` (validation names). Round 2: copying practised on many DIFFERENT tokens,
    because practice on 28 syllables only taught copying of those syllables (D-045)."""
    counts = Counter(re.findall(r"\b[a-z]{3,10}\b", book_text))
    banned = {r.lower() for r in reserved}
    return sorted(w for w, c in counts.items() if c >= min_count and w not in banned)


COPY_TASKS = {"task:repeat", "task:name", "task:position", "task:spell"}


def split_names(names, seed=5):
    """(train_names, validation_names): a fixed random 20% of the names is kept for testing only."""
    shuffled = names[:]
    random.Random(seed).shuffle(shuffled)
    cut = max(1, len(shuffled) // 5)
    return shuffled[cut:], shuffled[:cut]


SYLLABLES = ["ka", "lo", "mi", "ra", "ven", "to", "ba", "shi", "ne", "dor", "el", "wy", "zan", "pe",
             "qui", "ro", "fa", "lin", "gu", "sa", "tem", "vo", "ha", "ri", "bel", "don", "ma", "ti"]


def made_up_words(n, seed, capitalize=False):
    """Words that exist nowhere: random syllables. A network can only reproduce them by COPYING
    them from the message, never by remembering them, so they train and test real copying."""
    rng = random.Random(seed)
    out = set()
    while len(out) < n:
        w = "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3)))
        out.add(w.capitalize() if capitalize else w)
    return sorted(out)


def _phrase(rng, words, low=1, high=4):
    return " ".join(rng.choice(words) for _ in range(rng.randint(low, high)))


def task_examples(words, names, n_per_type, seed, copy_words=()):
    """Input-dependent tasks. Returns a list of Examples; the source says which task.

    copy_words: made-up words; half of the repeat and spell tasks use them, so those tasks can
    only be solved by copying from the message.
    """
    rng = random.Random(seed)
    out = []

    def add(kind, message, reply):
        out.append(Example([message], reply, f"task:{kind}"))

    for _ in range(n_per_type):
        p = _phrase(rng, copy_words if copy_words and rng.random() < 0.5 else words)
        add("repeat", rng.choice(["repeat: {p}", "Repeat after me: {p}", "say {p}", "Say: {p}",
                                  "please repeat {p}"]).format(p=p), p)
        name = rng.choice(names)
        template, reply = rng.choice([
            ("My name is {n}. What is my name?", "Your name is {n}."),
            ("I am {n}. What is my name?", "Your name is {n}."),
            ("my name is {n}. what is my name?", "Your name is {n}."),
            ("Call me {n}. What is my name?", "Your name is {n}."),
            ("My friend is called {n}. What is my friend called?", "Your friend is called {n}.")])
        add("name", template.format(n=name), reply.format(n=name))
        a, b = rng.sample(range(1000), 2)
        question, answer = rng.choice([
            (f"Is {a} bigger than {b}?", "Yes." if a > b else "No."),
            (f"Is {a} more than {b}?", "Yes." if a > b else "No."),
            (f"Is {a} less than {b}?", "Yes." if a < b else "No."),
            (f"Which is bigger, {a} or {b}?", str(max(a, b))),
            (f"Which is smaller, {a} or {b}?", str(min(a, b)))])
        add("compare", question, answer)
        p = _phrase(rng, words, 1, 8)
        add("count", rng.choice(["How many words are in: {p}", "Count the words: {p}",
                                 "how many words: {p}"]).format(p=p), str(len(p.split())))
        n = rng.randint(1, 998)
        question, answer = rng.choice([(f"What comes after {n}?", n + 1), (f"What number comes after {n}?", n + 1),
                                       (f"What comes before {n}?", n - 1), (f"What number is before {n}?", n - 1)])
        add("next", question, str(answer))
        p = _phrase(rng, words, 2, 6)
        first = rng.random() < 0.5
        add("position", rng.choice(["What is the {w} word of: {p}", "{w} word: {p}"]).format(
            w="first" if first else "last", p=p), p.split()[0] if first else p.split()[-1])
        w = rng.choice(copy_words if copy_words and rng.random() < 0.5 else words)
        add("spell", rng.choice(["Spell {w}.", "How do you spell {w}?", "spell {w}"]).format(w=w), "-".join(w))
    return out


# ----------------------------------------------------------------------- context use
# The answer is in an EARLIER turn, never in the last message. Unrelated turns can come in between.
# This is the counterweight to "mixed contexts" (brain/train_mix.py), which teach that unrelated
# earlier turns do not matter: here the question asks for something said before, so the network
# must go back and find it (or notice it was never said).
NAME_STATEMENTS = ["My name is {n}.", "my name is {n}", "I am {n}.", "Call me {n}.", "I'm {n}.", "My name's {n}."]
NAME_CORRECTIONS = ["no, my name is {n}", "No, my name is {n}.", "Actually, my name is {n}.", "no, I am {n}",
                    "Sorry, my name is {n}."]
NAME_QUESTIONS = ["What is my name?", "what is my name?", "what is my name", "What's my name?", "Who am I?"]
FRIEND_STATEMENTS = ["My friend is called {n}.", "My friend's name is {n}.", "I have a friend called {n}."]
FRIEND_QUESTIONS = ["What is my friend called?", "What is my friend's name?", "what is my friend called?"]
WORD_STATEMENTS = ["Remember the word {w}.", "The word is {w}.", "My word is {w}.", "remember this word: {w}"]
WORD_QUESTIONS = ["What was the word?", "What is my word?", "Which word did I give you?", "what was the word?"]
NUMBER_STATEMENTS = ["My number is {k}.", "Remember the number {k}.", "my number is {k}", "The number is {k}."]
NUMBER_QUESTIONS = ["What is my number?", "What was the number?", "what is my number?", "Which number did I give you?"]
LAST_QUESTIONS = ["What did I just say?", "what did I just say?", "What was my last message?", "Say that again."]
# Wordings that NEVER appear in training (validation only).
HELD_OUT_CONTEXT = {
    "name": (["People call me {n}.", "The name is {n}."], ["Do you remember my name?", "Tell me my name."]),
    "word": (["Keep this word in mind: {w}."], ["Tell me the word."]),
    "number": (["Keep this number in mind: {k}."], ["Tell me my number."]),
}
CONTEXT_TYPES = ["name", "correction", "friend", "word", "number", "last message", "not given"]


def _fillers(rng, pool, low, high):
    """0-2 unrelated exchanges (user message, reply) as a flat list of turns."""
    turns = []
    for _ in range(rng.randint(low, high)):
        turns += list(rng.choice(pool))
    return turns


def _neutral_filler_pool(pool):
    """Unrelated exchanges that mention nothing a context question could ask about."""
    banned = re.compile(r"\b(name|names|named|call|called|friend|word|words|number|am|i'm|said|say)\b", re.I)
    return [(u, a) for u, a in pool if not banned.search(u + " " + a) and len(u) <= 60 and len(a) <= 60]


def context_examples(names, words, filler_pool, n_per_type, seed, held_out_wordings=False):
    """Multi-turn examples whose answer comes from an earlier turn. Returns a list of Examples.

    names, words: what to use (made up + training names for training; never-trained ones for validation)
    filler_pool:  (message, reply) pairs from the other categories, used as unrelated turns in between
    held_out_wordings: use only the wordings in HELD_OUT_CONTEXT (validation)
    """
    rng = random.Random(seed)
    pool = _neutral_filler_pool(filler_pool)
    ai_turns = [a for _, a in pool]
    out = []

    def add(kind, turns, reply):
        out.append(Example(turns, reply, f"context:{kind}"))

    def said(statement):            # a user turn, then whatever the AI happened to answer
        return [statement, rng.choice(ai_turns)]

    if held_out_wordings:
        for _ in range(n_per_type):
            kind = rng.choice(list(HELD_OUT_CONTEXT))
            statements, questions = HELD_OUT_CONTEXT[kind]
            n, w, k = rng.choice(names), rng.choice(words), rng.randint(0, 999)
            reply = {"name": f"Your name is {n}.", "word": w, "number": str(k)}[kind]
            add("new wording", said(rng.choice(statements).format(n=n, w=w, k=k)) + _fillers(rng, pool, 0, 2)
                + [rng.choice(questions)], reply)
        return out

    for _ in range(n_per_type):
        n, other = rng.sample(names, 2)
        add("name", said(rng.choice(NAME_STATEMENTS).format(n=n)) + _fillers(rng, pool, 0, 2)
            + [rng.choice(NAME_QUESTIONS)], f"Your name is {n}.")

        add("correction", said(rng.choice(NAME_STATEMENTS).format(n=other)) + _fillers(rng, pool, 0, 1)
            + said(rng.choice(NAME_CORRECTIONS).format(n=n)) + _fillers(rng, pool, 0, 1)
            + [rng.choice(NAME_QUESTIONS)], f"Your name is {n}.")

        # two names given: the question decides which one is wanted
        first = [rng.choice(NAME_STATEMENTS).format(n=n), rng.choice(FRIEND_STATEMENTS).format(n=other)]
        rng.shuffle(first)
        turns = said(first[0]) + _fillers(rng, pool, 0, 1) + said(first[1]) + _fillers(rng, pool, 0, 1)
        if rng.random() < 0.5:
            add("friend", turns + [rng.choice(FRIEND_QUESTIONS)], f"Your friend is called {other}.")
        else:
            add("friend", turns + [rng.choice(NAME_QUESTIONS)], f"Your name is {n}.")

        w = rng.choice(words)
        add("word", said(rng.choice(WORD_STATEMENTS).format(w=w)) + _fillers(rng, pool, 0, 2)
            + [rng.choice(WORD_QUESTIONS)], w)

        k = rng.randint(0, 999)
        add("number", said(rng.choice(NUMBER_STATEMENTS).format(k=k)) + _fillers(rng, pool, 0, 2)
            + [rng.choice(NUMBER_QUESTIONS)], str(k))

        before = _fillers(rng, pool, 0, 1)
        message, answer = rng.choice(pool)
        add("last message", before + [message, answer, rng.choice(LAST_QUESTIONS)], message)

        # asked for something that was never said: the only correct answer is that it was not given
        kind = rng.choice(["name", "friend", "word", "number"])
        turns = _fillers(rng, pool, 1, 2)
        if kind == "friend" and rng.random() < 0.5:           # own name given, friend's name not
            turns = said(rng.choice(NAME_STATEMENTS).format(n=n)) + turns
        question, reply = {"name": (NAME_QUESTIONS, "You have not told me your name."),
                           "friend": (FRIEND_QUESTIONS, "You have not told me your friend's name."),
                           "word": (WORD_QUESTIONS, "You have not told me a word."),
                           "number": (NUMBER_QUESTIONS, "You have not told me a number.")}[kind]
        add("not given", turns + [rng.choice(question)], reply)
    return out

# ------------------------------------------------------------------- memory use (round 3)
# Information YOU gave in an earlier turn (as long-term memory shows it to the brain), then a question
# about it. Replies only repeat your information. The system is never given favorites of its own.
FAVORITE_CATEGORIES = ["color", "food", "animal", "sport", "song", "book", "movie", "drink", "fruit", "game",
                       "subject", "season", "flower", "number", "band", "show", "dessert", "snack", "toy", "hobby",
                       "holiday", "singer", "team", "place", "day", "word", "city", "country", "car", "sound"]
PETS = ["dog", "cat", "parrot", "hamster", "rabbit", "horse", "fish", "bird", "turtle", "snake", "puppy", "kitten",
        "lizard", "pony", "goldfish", "mouse"]
FAMILY = ["sister", "brother", "mother", "father", "mom", "dad", "son", "daughter", "wife", "husband", "cousin",
          "aunt", "uncle", "grandmother", "grandfather", "grandma", "grandpa", "friend", "best friend", "teacher",
          "boss", "neighbor", "coach", "roommate"]
# Realistic values kept OUT of training, so the tests use values the brain never saw in these roles.
RESERVED_VALUES = {
    "color": ["green", "blue", "red", "purple", "yellow", "orange", "pink", "black"],
    "food": ["pizza", "pasta", "sushi", "tacos", "pancakes", "soup", "rice", "chocolate"],
    "animal": ["dog", "cat", "horse", "lion", "elephant", "dolphin", "owl", "tiger"],
    "sport": ["tennis", "basketball", "soccer", "swimming", "baseball", "hockey", "golf", "volleyball"],
    "like": ["hiking", "chess", "painting", "music", "reading", "cooking", "dancing", "skiing"],
    "place": ["Georgia", "Ohio", "Texas", "Maine", "Kenya", "Peru", "Norway", "Chile", "Oregon", "Nepal"],
}
RESERVED_WORDS = {w.lower() for values in RESERVED_VALUES.values() for w in values}
FAVORITE_STATEMENTS = ["My favorite {c} is {v}.", "my favorite {c} is {v}", "My fav {c} is {v}.", "{V} is my favorite {c}.",
                       "My favourite {c} is {v}.", "my favorite {c} is {v}!"]
FAVORITE_QUESTIONS = ["What is my favorite {c}?", "what is my favorite {c}", "What's my favorite {c}?",
                      "what's my fav {c}?", "What is my favourite {c}?"]
LIKE_STATEMENTS = [("I like {v}.", "like"), ("i like {v}", "like"), ("I really like {v}.", "like"),
                   ("I love {v}.", "love"), ("i love {v}", "love"), ("I enjoy {v}.", "enjoy")]
DISLIKE_STATEMENTS = ["I don't like {v}.", "i dont like {v}", "I hate {v}.", "I really don't like {v}."]
PET_STATEMENTS = ["I have a {p} named {n}.", "My {p}'s name is {n}.", "My {p} is called {n}.", "{N} is my {p}.",
                  "i have a {p} called {n}", "my {p}s name is {n}"]
PET_QUESTIONS = ["What is my {p}'s name?", "What is my {p} called?", "What's my {p}'s name?", "what is my {p}s name",
                 "what is my {p} called"]
FAMILY_STATEMENTS = ["My {r} is called {n}.", "My {r}'s name is {n}.", "I have a {r} named {n}.", "{N} is my {r}.",
                     "my {r} is called {n}", "My {r} is {N}."]
FAMILY_QUESTIONS = ["What is my {r} called?", "What is my {r}'s name?", "Who is my {r}?", "what is my {r}s name",
                    "What's my {r}'s name?"]
PLACE_KINDS = {  # statement templates, questions, reply
    "live": (["I live in {P}.", "i live in {p}", "I live in {P} now."], ["Where do I live?", "where do i live"],
             "You live in {P}.", "where you live"),
    "from": (["I'm from {P}.", "I am from {P}.", "im from {p}"], ["Where am I from?", "where am i from"],
             "You are from {P}.", "where you are from"),
    "born": (["I was born in {P}.", "i was born in {p}"], ["Where was I born?", "where was i born"],
             "You were born in {P}.", "where you were born"),
    "grew up": (["I grew up in {P}.", "i grew up in {p}"], ["Where did I grow up?", "where did i grow up"],
                "You grew up in {P}.", "where you grew up"),
}
# Wordings that NEVER appear in training (validation only)
HELD_OUT_MEMORY = {
    "favorite": (["The {c} I like most is {v}."], ["Which {c} do I like most?", "Tell me my favorite {c}."]),
    "like": (["I'm really into {v}."], ["What am I into?"]),
    "pet": (["My pet {p} is named {n}."], ["Do you remember my {p}'s name?"]),
    "family": (["The name of my {r} is {n}."], ["Tell me the name of my {r}."]),
    "place": (["My home is in {P}."], ["Where is my home?"]),
}
MEMORY_TYPES = ["favorite", "like", "pet", "family", "place", "not given"]


def _cap(v):
    return v[:1].upper() + v[1:]


def _fill(template, value, **other):
    """Fill a statement template. Returns (statement, the value exactly as written in it), so the reply can
    repeat the same letters: the brain copies tokens, and "Matori" and "matori" are different tokens."""
    written = _cap(value) if ("{V}" in template or "{N}" in template or "{P}" in template) else value
    if "{p}" in template and "{P}" not in template and other.get("lower_place"):
        written = value.lower()
    fields = dict(v=value, V=_cap(value), n=value, N=_cap(value), P=value, **other)
    if other.get("lower_place"):
        fields["p"] = value.lower()
    fields.pop("lower_place", None)
    return template.format(**fields), written


def memory_use_examples(names, values, places, filler_pool, n_per_type, seed, held_out_wordings=False,
                        favorite_values=None, like_values=None):
    """Examples of using information the user gave earlier (see MEMORY above). Returns a list of Examples.

    names:  pet and family names;  values: things (lowercase words) for favorites and likes;
    places: place names;  favorite_values / like_values: optional per-category values (for realistic tests)
    """
    rng = random.Random(seed)
    pool = _neutral_filler_pool(filler_pool)
    ai_turns = [a for _, a in pool]
    out = []

    def said(statement):                     # your turn, then whatever the AI happened to answer
        return [statement, rng.choice(ai_turns)]

    def name():
        n = rng.choice(names)
        return n.lower() if rng.random() < 0.2 else n

    def fav_value(c):
        return rng.choice(favorite_values.get(c, values)) if favorite_values else rng.choice(values)

    def one_memory(kind, exclude=()):
        """A statement of one kind: (statement, what it is about)."""
        if kind == "favorite":
            c = rng.choice([x for x in FAVORITE_CATEGORIES if ("favorite", x) not in exclude] or FAVORITE_CATEGORIES)
            return _fill(rng.choice(FAVORITE_STATEMENTS), fav_value(c), c=c)[0], ("favorite", c)
        if kind == "like":
            v = rng.choice(like_values or values)
            return rng.choice(LIKE_STATEMENTS)[0].format(v=v), ("like", v)
        if kind == "pet":
            pet = rng.choice([x for x in PETS if ("pet", x) not in exclude] or PETS)
            return _fill(rng.choice(PET_STATEMENTS), name(), p=pet)[0], ("pet", pet)
        if kind == "family":
            r = rng.choice([x for x in FAMILY if ("family", x) not in exclude] or FAMILY)
            return _fill(rng.choice(FAMILY_STATEMENTS), name(), r=r)[0], ("family", r)
        if kind == "place":
            k = rng.choice([x for x in PLACE_KINDS if ("place", x) not in exclude] or list(PLACE_KINDS))
            return _fill(rng.choice(PLACE_KINDS[k][0]), rng.choice(places), lower_place=True)[0], ("place", k)
        return f"My name is {name()}.", ("name", None)

    def build(statement, about, question, reply, kind):
        """Your statement among 0-2 other memories and 0-1 unrelated exchanges, then the question."""
        memories = [said(statement)]
        kinds = ["pet", "family", "place", "name"] + (["favorite", "like"] if about[0] != "like" else [])
        for _ in range(rng.randint(0, 2)):
            other, other_about = one_memory(rng.choice(kinds), exclude={about})
            if other_about != about:
                memories.append(said(other))
        rng.shuffle(memories)
        turns = [t for m in memories for t in m] + _fillers(rng, pool, 0, 1) + [question]
        out.append(Example(turns, reply, f"memory:{kind}"))

    if held_out_wordings:
        for _ in range(n_per_type):
            kind = rng.choice(list(HELD_OUT_MEMORY))
            statements, questions = HELD_OUT_MEMORY[kind]
            c, pet, r = rng.choice(["color", "food", "animal", "sport"]), rng.choice(PETS), rng.choice(FAMILY)
            value = {"favorite": fav_value(c), "like": rng.choice(like_values or values), "pet": name(),
                     "family": name(), "place": rng.choice(places)}[kind]
            statement, v = _fill(rng.choice(statements), value, c=c, p=pet, r=r)
            reply = {"favorite": f"Your favorite {c} is {v}.", "like": f"You like {v}.",
                     "pet": f"Your {pet} is called {v}.", "family": f"Your {r} is called {v}.",
                     "place": f"You live in {v}."}[kind]
            build(statement, (kind, None), rng.choice(questions).format(c=c, p=pet, r=r), reply, "new wording")
        return out

    for _ in range(n_per_type):
        # favorites
        c = rng.choice(FAVORITE_CATEGORIES)
        statement, v = _fill(rng.choice(FAVORITE_STATEMENTS), fav_value(c), c=c)
        build(statement, ("favorite", c), rng.choice(FAVORITE_QUESTIONS).format(c=c), f"Your favorite {c} is {v}.",
              "favorite")
        # likes and dislikes (the reply repeats the verb you used)
        v = rng.choice(like_values or values)
        if rng.random() < 0.75:
            statement, verb = rng.choice(LIKE_STATEMENTS)
            question, reply = rng.choice([(f"What do I {verb}?", f"You {verb} {v}."),
                                          ("What do I like?", f"You like {v}."), (f"Do I like {v}?", "Yes.")])
        else:
            statement = rng.choice(DISLIKE_STATEMENTS)
            question, reply = rng.choice([("What don't I like?", f"You don't like {v}."), (f"Do I like {v}?", "No.")])
        build(statement.format(v=v), ("like", v), question, reply, "like")
        # pets
        pet = rng.choice(PETS)
        statement, n = _fill(rng.choice(PET_STATEMENTS), name(), p=pet)
        build(statement, ("pet", pet), rng.choice(PET_QUESTIONS).format(p=pet), f"Your {pet} is called {n}.", "pet")
        # family
        r = rng.choice(FAMILY)
        statement, n = _fill(rng.choice(FAMILY_STATEMENTS), name(), r=r)
        build(statement, ("family", r), rng.choice(FAMILY_QUESTIONS).format(r=r), f"Your {r} is called {n}.", "family")
        # places
        k = rng.choice(list(PLACE_KINDS))
        statements, questions, reply, _ = PLACE_KINDS[k]
        statement, pl = _fill(rng.choice(statements), rng.choice(places), lower_place=True)
        build(statement, ("place", k), rng.choice(questions), reply.format(P=pl), "place")
        # never given: other things may have been said, but not the thing asked about
        kind = rng.choice(["favorite", "like", "pet", "family", "place"])
        if kind == "favorite":
            c = rng.choice(FAVORITE_CATEGORIES)
            question, reply, about = rng.choice(FAVORITE_QUESTIONS).format(c=c), \
                f"You have not told me your favorite {c}.", ("favorite", c)
        elif kind == "like":
            question, reply, about = "What do I like?", "You have not told me what you like.", ("like", None)
        elif kind == "pet":
            pet = rng.choice(PETS)
            question, reply, about = rng.choice(PET_QUESTIONS).format(p=pet), \
                f"You have not told me your {pet}'s name.", ("pet", pet)
        elif kind == "family":
            r = rng.choice(FAMILY)
            question, reply, about = rng.choice(FAMILY_QUESTIONS).format(r=r), \
                f"You have not told me your {r}'s name.", ("family", r)
        else:
            k = rng.choice(list(PLACE_KINDS))
            question, reply, about = rng.choice(PLACE_KINDS[k][1]), f"You have not told me {PLACE_KINDS[k][3]}.", \
                ("place", k)
        memories = []
        for _ in range(rng.randint(0, 3)):
            other_kind = rng.choice(["pet", "family", "place", "name"] +
                                    (["favorite", "like"] if kind != "like" else []))
            other, other_about = one_memory(other_kind, exclude={about})
            if other_about != about:
                memories.append(said(other))
        turns = [t for m in memories for t in m] + _fillers(rng, pool, 0 if memories else 1, 1) + [question]
        out.append(Example(turns, reply, "memory:not given"))
    return out


# ------------------------------------------------------------------------ conversation
GREETINGS =["hi", "hello", "hey", "hi there", "hello there", "good morning", "good afternoon", "good night"]
HELD_OUT_GREETINGS = ["good evening", "good day"]           # never trained: does "good X" -> "Good X." carry over?
THANKS = ["thank you", "thanks", "thank you very much", "thanks a lot", "many thanks"]
FAREWELLS = ["goodbye", "bye", "see you later", "farewell"]


def _variants(text, rng):
    """The same words typed in different ways: case and punctuation."""
    styles = [text, text.capitalize(), text.upper(), text.title()]
    return [rng.choice(styles) + rng.choice(["", ".", "!", "!!", "?"][: 4 if text in THANKS else 5])
            for _ in range(6)]


def conversation_examples(seed, held_out=False):
    """Neutral conventions only (see the module description): the reply mirrors the message."""
    rng = random.Random(seed)
    out = []
    greetings = HELD_OUT_GREETINGS if held_out else GREETINGS
    for g in greetings:
        for v in _variants(g, rng):
            out.append(Example([v], g[0].upper() + g[1:] + ".", "conversation:greeting"))
    if not held_out:
        for t in THANKS:
            for v in _variants(t, rng):
                out.append(Example([v], "You're welcome.", "conversation:thanks"))
        for f in FAREWELLS:
            for v in _variants(f, rng):
                out.append(Example([v], f[0].upper() + f[1:] + ".", "conversation:farewell"))
    return out
