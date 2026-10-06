"""
Conversations as training data (Stage 1.5).

A conversation is a list of turns spoken alternately by two people. A training EXAMPLE
is "some earlier turns (the context) -> the next turn (the reply)".

Where the conversations come from (all human-written, or made by our own program):
  1. prose books: runs of paragraphs of quoted speech        "Where?"  ->  "In the garden."
  2. plays: consecutive speeches                              LANE. Yes, sir.
  3. arithmetic practice written by arithmetic_examples()     what is 5+5  ->  10
  4. YOUR files in data/conversations/, in USER: / AI: format

How an example becomes numbers for the network:

      <user> Where is it? <ai> In the garden. <end>

The network reads everything but is only GRADED on the reply and the <end> marker
(the "loss mask"). That is supervised training: learn to produce the reply GIVEN the
message. While training it always sees the true previous characters ("teacher forcing").
Nothing here decides what the replies should say: they come from the data.
"""

import random
import re
from dataclasses import dataclass

USER, AI, END = "<user>", "<ai>", "<end>"
SPECIALS = (USER, AI, END)

MAX_REPLY_CHARS = 128     # replies longer than this are cut back to their last full sentence
MAX_TURN_CHARS = 240      # earlier turns are shortened to this (from the start)
MAX_CONTEXT_TURNS = 3     # how many turns before the reply are kept as context


@dataclass
class Example:
    context: list         # earlier turns, oldest first; the LAST one is the message answered
    reply: str
    source: str           # e.g. "book:844", "arithmetic", "own:my_chats.txt"


# ------------------------------------------------------------------ prose books
SPEECH_VERBS = ("said|says|cried|replied|asked|answered|exclaimed|returned|continued|added|"
                "whispered|shouted|called|remarked|observed|thought|began|went on")
# A closing single quote: right after punctuation ('No,' said), or before a speech verb
# ('Come here' said), or at the end. Anything else is an apostrophe (don't, the girls' books).
SINGLE_CLOSE = rf"(?:(?<=[.,!?;:\-])'|'(?=\s+(?:{SPEECH_VERBS})\b))"
QUOTE_SPANS = {
    '"': re.compile(r'"([^"]+)"?'),
    "'": re.compile(rf"(?:^|(?<=[\s(\-]))'(.+?)(?:{SINGLE_CLOSE}|'?$)"),
}


def quote_style(text):
    """Does this book use "double" or 'single' quotes for speech?"""
    paragraphs = text.split("\n\n")
    doubles = sum(p.startswith('"') for p in paragraphs)
    singles = sum(p.startswith("'") for p in paragraphs)
    return "'" if singles > doubles else '"'


def _speech_continues(paragraph, q):
    """True if the speaker keeps talking in the next paragraph (the quote is not closed)."""
    if q == '"':
        return paragraph.count('"') % 2 == 1
    if paragraph.rstrip().endswith("'"):
        return False
    opens = [m.start() for m in re.finditer(r"(?:^|(?<=[\s(\-]))'(?=\S)", paragraph)]
    closes = [m.start() for m in re.finditer(SINGLE_CLOSE, paragraph)]
    return bool(opens) and (not closes or opens[-1] > closes[-1])


def prose_conversations(text):
    """Runs of consecutive quoted paragraphs, as lists of spoken turns (narration removed).

    '"Who are you?" said the Caterpillar.'  ->  'Who are you?'
    A paragraph whose quote is not closed continues the same speaker's turn.
    """
    q = quote_style(text)
    conversations, current, continuing = [], [], False
    for paragraph in text.split("\n\n"):
        if paragraph.startswith(q):
            spans = [s.strip() for s in QUOTE_SPANS[q].findall(paragraph) if s.strip()]
            spoken = " ".join(spans)
            if continuing and current:
                current[-1] = f"{current[-1]} {spoken}".strip()
            elif spoken:
                current.append(spoken)
            continuing = _speech_continues(paragraph, q)
        else:
            if len(current) >= 2:
                conversations.append(current)
            current, continuing = [], False
    if len(current) >= 2:
        conversations.append(current)
    return conversations


# ------------------------------------------------------------------------- plays
SPEECH_RE = re.compile(
    r"^(?P<who>(?:[A-Z][A-Z'\-]*\.?\s){0,3}[A-Z][A-Z'\-]+)\s*"     # LADY BRACKNELL / MRS. ERLYNNE
    r"(?:\([^)]*\)|\[[^\]]*\])?\s*\.\s+(?P<speech>\S.*)$")          # optional (direction), then ". "
HEADING_RE = re.compile(r"^(?:ACT|SCENE|FIRST|SECOND|THIRD|FOURTH|FIFTH|END|THE END|CURTAIN|"
                        r"EPILOGUE|PROLOGUE)\b")
DIRECTION_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")


def play_conversations(text):
    """Consecutive speeches in a play, as lists of turns (speaker names and directions removed).

    'CRICHTON (obediently). Thank you, sir.'  ->  'Thank you, sir.'
    Stage directions between speeches are skipped; an ACT or SCENE heading ends a conversation.
    """
    conversations, current, last_speaker = [], [], None
    for paragraph in text.split("\n\n"):
        if HEADING_RE.match(paragraph):
            if len(current) >= 2:
                conversations.append(current)
            current, last_speaker = [], None
            continue
        match = SPEECH_RE.match(paragraph)
        if not match:
            continue                                   # a stage direction: skip it
        speech = re.sub(r"\s+", " ", DIRECTION_RE.sub("", match["speech"])).strip()
        if not re.search(r"[A-Za-z0-9]", speech):
            continue
        if match["who"] == last_speaker and current:
            current[-1] = f"{current[-1]} {speech}"
        else:
            current.append(speech)
        last_speaker = match["who"]
    if len(current) >= 2:
        conversations.append(current)
    return conversations


# ------------------------------------------------------------ your conversation files
LABEL_RE = re.compile(r"^\s*(USER|AI)\s*:\s*(.*)$", re.IGNORECASE)


def parse_conversation_file(text):
    """Read the USER: / AI: format. Returns a list of conversations of (role, text) turns.

    A label can have its text on the same line or on the following lines. An empty line
    ends a conversation. Lines starting with # are comments.
    """
    conversations, turns = [], []

    def finish():
        cleaned = [(role, " ".join(t.split())) for role, t in turns if t.strip()]
        if cleaned:
            conversations.append(cleaned)

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if not line:
            if turns and turns[-1][1].strip():
                finish()
                turns = []
            continue
        label = LABEL_RE.match(line)
        if label:
            role = label[1].lower()
            if turns and turns[-1][0] == role:          # same speaker twice: one turn
                turns[-1] = (role, f"{turns[-1][1]} {label[2]}")
            else:
                turns.append((role, label[2]))
        elif turns:
            turns[-1] = (turns[-1][0], f"{turns[-1][1]} {line}")
    finish()
    return conversations


# ---------------------------------------------------------------- arithmetic practice
ADD_TEMPLATES = ("what is {a}+{b}", "What is {a} + {b}?", "what's {a} plus {b}?", "{a}+{b}=?",
                 "how much is {a} plus {b}?", "{a} + {b} = ?", "What is {a} plus {b}?", "calculate {a}+{b}")
SUB_TEMPLATES = ("what is {a}-{b}", "What is {a} - {b}?", "what's {a} minus {b}?", "{a}-{b}=?",
                 "how much is {a} minus {b}?", "{a} - {b} = ?", "What is {a} minus {b}?", "calculate {a}-{b}")


def arithmetic_problems(max_number=50):
    """Every addition a+b and every subtraction a-b (a >= b) with numbers 0..max_number."""
    problems = [("+", a, b, a + b) for a in range(max_number + 1) for b in range(max_number + 1)]
    problems += [("-", a, b, a - b) for a in range(max_number + 1) for b in range(a + 1)]
    return problems


def split_problems(problems, val_fraction=0.1, seed=0):
    """Hold out a fixed random 10% of problems forever: the model never trains on them."""
    shuffled = problems[:]
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


def arithmetic_examples(problems, phrasings_per_problem=2, seed=0):
    """Each problem written in `phrasings_per_problem` randomly chosen wordings."""
    rng = random.Random(seed)
    examples = []
    for op, a, b, answer in problems:
        templates = ADD_TEMPLATES if op == "+" else SUB_TEMPLATES
        for template in rng.sample(templates, phrasings_per_problem):
            examples.append(Example([template.format(a=a, b=b)], str(answer), "arithmetic"))
    return examples


# ----------------------------------------------------------------- building examples
def clean_reply(text):
    """Tidy one spoken turn to use as a reply, or None if it is unusable."""
    text = " ".join(text.split())
    text = re.sub(r"[,;:]$", ".", text)       # '"I know," said Anne' -> "I know."
    if len(text) > MAX_REPLY_CHARS:
        cut = max(text.rfind(ch, 0, MAX_REPLY_CHARS) for ch in ".!?")
        if cut < 10:
            return None
        text = text[:cut + 1]
    return text if re.search(r"[A-Za-z0-9]", text) else None


def shorten_turn(text):
    """Earlier turns keep only their last MAX_TURN_CHARS characters."""
    text = " ".join(text.split())
    return text if len(text) <= MAX_TURN_CHARS else text[-MAX_TURN_CHARS:].split(" ", 1)[-1]


def examples_from_turns(turns, source, reply_positions=None):
    """Every turn after the first (or only the given positions) becomes one example."""
    examples = []
    for i in (reply_positions if reply_positions is not None else range(1, len(turns))):
        if i < 1:
            continue
        reply = clean_reply(turns[i])
        if reply is None:
            continue
        context = [shorten_turn(t) for t in turns[max(0, i - MAX_CONTEXT_TURNS):i]]
        examples.append(Example(context, reply, source))
    return examples


# -------------------------------------------------------------- examples -> numbers
def _turn_tokens(tokenizer, text, role):
    ids = tokenizer.encode(text)
    if role == "user":
        return [tokenizer.special(USER)] + ids
    return [tokenizer.special(AI)] + ids + [tokenizer.special(END)]


def encode_prompt(tokenizer, context, max_tokens, report=False):
    """Context turns -> ids ending in <ai>, ready for the model to write the reply.

    The last turn is the user's message; earlier turns alternate backwards (ai, user, ...).
    If it is too long, the oldest turns are dropped, then the message is shortened from the start.
    With report=True it also returns (turns_kept, message_was_cut), so nothing is dropped silently.
    """
    roles = ["user" if (len(context) - 1 - i) % 2 == 0 else "ai" for i in range(len(context))]
    pieces = [_turn_tokens(tokenizer, t, r) for t, r in zip(context, roles)]
    ai = tokenizer.special(AI)
    while pieces and sum(map(len, pieces)) + 1 > max_tokens and len(pieces) > 1:
        pieces.pop(0)
    ids = [t for piece in pieces for t in piece]
    cut = len(ids) + 1 > max_tokens
    if cut:                                             # a single, very long message
        ids = [tokenizer.special(USER)] + ids[-(max_tokens - 2):]
    ids = ids + [ai]
    return (ids, len(pieces), cut) if report else ids


def encode_example(tokenizer, example, context_size):
    """One example -> (ids, weights): weights are 1 exactly on the reply and <end>.

    The sequence has at most context_size + 1 tokens (inputs = ids[:-1], targets = ids[1:]).
    """
    reply = tokenizer.encode(example.reply) + [tokenizer.special(END)]
    prompt = encode_prompt(tokenizer, example.context, context_size + 1 - len(reply))
    ids = prompt + reply
    weights = [0] * len(prompt) + [1] * len(reply)
    return ids, weights
