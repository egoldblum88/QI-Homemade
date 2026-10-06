"""
Every brain side by side (Stage 1, 1.5, 1.5b, 1.5b round 2): measured, not claimed.

    python -m brain.evaluate            # full comparison (a few minutes)
    python -m brain.evaluate --quick    # smaller samples (faster, noisier)

Every test uses data that NONE of the brains trained on. Two kinds of numbers are kept apart:

MODEL QUALITY: read directly from the network's probabilities, no sampling involved
  book loss            how well it predicts unseen books, per character (lower = better)
  reply loss           how well it predicts the real reply to an unseen message, per character
  input use            reply loss with a WRONG message minus with the RIGHT one (0 = message ignored)
  pick the reply       rank the real reply among 10 candidates by how much the message raises its
                       probability (pointwise mutual information). Guessing = 10%
  P(correct answer)    the probability the network gives to the exact correct answer
                       (arithmetic, tasks and context questions it never saw)
  category losses      loss per character on each category's never-seen examples
  message changes      how different the first-character distributions are for different messages
  distribution         (average Jensen-Shannon divergence; 0 = every message gets the same distribution)
  format carry-over    probability that a NON-numeric question gets a digit as its first character when
                       an arithmetic exchange came just before it (lower = better)

DECODING QUALITY: what a decoding method actually produces
  exact (greedy)       the reply equals the correct answer when always taking the most likely token
  exact (sampled)      the same with sampling (temperature 0.8, top-k 20), as the chat does
  context (greedy)     the answer is in an EARLIER turn (a name, word or number given before)
  ends by itself       the reply ends with <end> before the length limit (116 messages, sampled)
  repetition           replies that repeat a word at once or a 3-word phrase (116 messages, sampled)
  most common reply    share of the most frequent greedy reply: high = the same generic answer everywhere
  real words, distinct replies, length
  your conversation    the exact messages from your chat, replayed turn by turn

Baselines show what no learning would score (e.g. always giving the most common answer).
Reports: models/evaluation.md, evaluation.json, evaluation.html (open it in a browser, or
http://localhost:8000/evaluation while the chat server runs).
"""

import argparse
import html
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from brain.dialogue import END, Example, arithmetic_problems, encode_example, encode_prompt, split_problems
from brain.generate import (STAGE15B2_MODEL, STAGE15B3_MODEL, STAGE15B4_MODEL, STAGE15B4B_MODEL, STAGE15B4C_MODEL,
                            STAGE15B_MODEL, STAGE15_MODEL, STAGE1_MODEL, generate, load_brain, reply_room, reply_to)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MIX = PROCESSED / "mix"
REPORT = ROOT / "models" / "evaluation.md"

# Test messages for measurement only. They are never used for training.
TEST_MESSAGES = ["hi", "hello", "Hello! How are you?", "what is 5+5", "10 + 10", "Who are you?",
                 "What is your name?", "Where do you live?", "Tell me a story.", "I am sad today.",
                 "Do you like music?", "Good morning.", "Thank you very much.", "Can you help me?",
                 "Why is the sky blue?", "Goodbye!"]
# Your chat after Stage 1.5b, replayed turn by turn (each brain's own replies become the history).
USER_CONVERSATION = ["hi", "what is your name?", "no, my name is Etai", "what is your favorite thing to do?",
                     "what is my name?"]
NON_NUMERIC_QUESTIONS = ["what is your name", "who are you", "Where do you live?", "tell me a story",
                         "Is it cold today?", "what is the time"]


def log_softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


# ------------------------------------------------------------ one adapter per kind of brain
class Stage1Brain:
    """The Stage 1 character network. Conversations are written as book dialogue."""

    def __init__(self, model, tokenizer, name="Stage 1"):
        self.model, self.tok, self.name = model, tokenizer, name
        self.pad = tokenizer.char_to_id["\n"]

    @staticmethod
    def prompt(context):
        return "\n\n" + "\n\n".join(f'"{turn}"' for turn in context) + '\n\n"'

    def _logprobs(self, X, Y):
        return np.concatenate([log_softmax(self.model.forward(X[i:i + 8192])[0])[np.arange(len(Y[i:i + 8192])),
                                                                              Y[i:i + 8192]]
                               for i in range(0, len(Y), 8192)])

    def score(self, pairs):
        """log P(reply + end | context) summed, and the number of characters (reply + 1 end)."""
        N = self.model.context
        X, Y, owner = [], [], []
        for k, (context, reply) in enumerate(pairs):
            prompt = self.tok.encode(self.prompt(context))
            seq = [self.pad] * N + prompt + self.tok.encode(reply + '"')
            for j in range(N + len(prompt), len(seq)):
                X.append(seq[j - N:j]); Y.append(seq[j]); owner.append(k)
        logp = self._logprobs(np.array(X), np.array(Y))
        return np.bincount(owner, weights=logp, minlength=len(pairs)), np.array([len(r) + 1 for _, r in pairs])

    def reply(self, context, temperature=0.8, seed=None):
        result = generate(self.model, self.tok, self.prompt(context), max_chars=200, temperature=temperature,
                          top_k=20, loop_penalty=0.3, stop={'"', "\n"}, seed=seed)
        return result.text.strip(), result.stopped_by == "stop marker"

    def book_nll(self, windows, first=128):
        N = self.model.context
        X, Y = [], []
        for w in windows:
            ids = self.tok.encode(w)
            for j in range(first + 1, len(ids)):
                X.append(ids[j - N:j]); Y.append(ids[j])
        logp = self._logprobs(np.array(X), np.array(Y))
        return float(-logp.sum()), len(Y)

    def first_char_distribution(self, context):
        ids = self.tok.encode(self.prompt(context))
        ids = [self.pad] * max(0, self.model.context - len(ids)) + ids
        p = self.model.next_probabilities(ids[-self.model.context:])
        dist = Counter()
        for i, prob in enumerate(p):
            ch = self.tok.chars[i]
            dist["<end>" if ch == '"' else ch] += prob          # a closing quote ends its reply
        return dist


class ReplyBrain:
    """A Stage 1.5 / 1.5b attention network with <user> / <ai> / <end> markers (any tokenizer)."""

    def __init__(self, model, tokenizer, name):
        from brain.train_chat import pad_batch
        self.model, self.tok, self.name, self.pad_batch = model, tokenizer, name, pad_batch
        self.pieces = [tokenizer.decode([i], show_specials=True) for i in range(tokenizer.vocab_size)]
        self.char_len = np.array([len(tokenizer.decode([i])) or 1 for i in range(tokenizer.vocab_size)])

    def score(self, pairs):
        encoded = [encode_example(self.tok, Example(list(c), r, ""), self.model.context) for c, r in pairs]
        sums = np.zeros(len(pairs))
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
        for start in range(0, len(order), 64):
            batch = order[start:start + 64]
            X, Y, W = self.pad_batch([encoded[i] for i in batch], self.tok.special(END))
            logp = np.take_along_axis(log_softmax(self.model.forward(X)), Y[..., None], axis=-1)[..., 0]
            sums[batch] = (logp * W).sum(axis=1)
        return sums, np.array([len(r) + 1 for _, r in pairs])

    def reply(self, context, temperature=0.8, seed=None):
        result, _ = reply_to(self.model, self.tok, list(context), temperature=temperature, top_k=20,
                             loop_penalty=0.3, seed=seed)
        return result.text.strip(), result.stopped_by == "stop marker"

    def book_nll(self, windows, first=128):
        """Loss on the characters after the first `first` characters of each window."""
        total, chars = 0.0, 0
        for w in windows:
            ids = self.tok.encode(w)[: self.model.context + 1]
            seen, k = 0, len(ids)
            for idx, i in enumerate(ids):                     # first token that starts after `first` chars
                if seen > first:
                    k = idx
                    break
                seen += len(self.tok.decode([i]))
            if k >= len(ids):
                continue
            logp = log_softmax(self.model.forward(np.array([ids[:-1]]))[0])
            targets = np.arange(k, len(ids))
            total -= float(logp[targets - 1, np.array(ids)[targets]].sum())
            chars += int(self.char_len[np.array(ids)[targets]].sum())
        return total, chars

    def first_char_distribution(self, context):
        ids = encode_prompt(self.tok, context, self.model.context - reply_room(self.model))
        p = self.model.next_probabilities(ids)
        dist = Counter()
        for i, prob in enumerate(p):
            piece = self.pieces[i]
            dist[piece if piece.startswith("<") and piece.endswith(">") else piece[:1]] += prob
        return dist


# ----------------------------------------------------------------------- measurements
def per_char_loss(brain, pairs):
    sums, chars = brain.score(pairs)
    return float(-sums.sum() / chars.sum())


def input_use(brain, pairs, rng):
    order = list(range(len(pairs)))
    while any(i == j for i, j in zip(order, range(len(order)))):   # nobody keeps their own message
        rng.shuffle(order)
    wrong = [(pairs[j][0], reply) for j, (_, reply) in zip(order, pairs)]
    return per_char_loss(brain, wrong) - per_char_loss(brain, pairs)


def pick_the_reply(brain, pairs, rng, candidates=10):
    replies = [r for _, r in pairs]
    prior, _ = brain.score([([""], r) for r in replies])
    groups = [[i] + rng.sample([j for j in range(len(pairs)) if j != i], candidates - 1) for i in range(len(pairs))]
    sums, _ = brain.score([(pairs[i][0], replies[j]) for i, group in enumerate(groups) for j in group])
    scores = sums.reshape(len(pairs), candidates) - prior[np.array(groups)]
    return float(np.mean(np.argmax(scores, axis=1) == 0))


def probability_of_correct(brain, examples):
    sums, _ = brain.score([(ex.context, ex.reply) for ex in examples])
    return float(np.mean(np.exp(sums)))


def exact_list(brain, examples, temperature=0, seed=0):
    return [brain.reply(ex.context, temperature=temperature, seed=seed + k)[0] == ex.reply
            for k, ex in enumerate(examples)]


def exact(brain, examples, temperature=0, seed=0):
    return float(np.mean(exact_list(brain, examples, temperature, seed)))


def by_type(examples, right):
    """Exact-match rate per kind of example ("task:name" -> "name")."""
    groups = {}
    for ex, ok in zip(examples, right):
        groups.setdefault(ex.source.split(":", 1)[-1], []).append(ok)
    return {kind: float(np.mean(v)) for kind, v in sorted(groups.items())}


def has_repetition(text):
    """True if a reply repeats a word right away ("dear, dear") or any 3-word phrase ("a good deal of
    ... a good deal of"). Two-word phrases are not counted: "I know ... I know" is ordinary English."""
    words = re.findall(r"[a-z']+", text.lower())
    triples = list(zip(words, words[1:], words[2:]))
    return len(triples) != len(set(triples)) or any(a == b for a, b in zip(words, words[1:]))


def generation_stats(brain, messages):
    """Sampled and greedy replies to many different single messages."""
    sampled = [brain.reply([m], temperature=0.8, seed=2000 + k) for k, m in enumerate(messages)]
    greedy = [brain.reply([m], temperature=0)[0] for m in messages]
    return {
        "ends_by_itself": float(np.mean([stopped for _, stopped in sampled])),
        "repetition_rate": float(np.mean([has_repetition(text) for text, _ in sampled])),
        "most_common_reply": Counter(greedy).most_common(1)[0][1] / len(greedy),
    }


def replay(brain, messages):
    """Your conversation, turn by turn: the brain's own greedy replies become the history."""
    context, replies = [], []
    for message in messages:
        context.append(message)
        text, _ = brain.reply(list(context), temperature=0)
        replies.append(text)
        context.append(text)
    return replies


def js_divergence(p, q):
    keys = set(p) | set(q)
    a = np.array([p.get(k, 0.0) for k in keys]) + 1e-12
    b = np.array([q.get(k, 0.0) for k in keys]) + 1e-12
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    return float(0.5 * (a * np.log(a / m)).sum() + 0.5 * (b * np.log(b / m)).sum())


def message_sensitivity(brain, messages):
    dists = [brain.first_char_distribution([m]) for m in messages]
    values = [js_divergence(dists[i], dists[j]) for i in range(len(dists)) for j in range(i + 1, len(dists))]
    return float(np.mean(values))


def format_carry_over(brain, arithmetic_examples):
    digits = set("0123456789")
    after, before = [], []
    for k, question in enumerate(NON_NUMERIC_QUESTIONS):
        ex = arithmetic_examples[k % len(arithmetic_examples)]
        with_turn = brain.first_char_distribution([ex.context[-1], ex.reply, question])
        alone = brain.first_char_distribution([question])
        after.append(sum(v for c, v in with_turn.items() if c in digits))
        before.append(sum(v for c, v in alone.items() if c in digits))
    return float(np.mean(after)), float(np.mean(before))


def reply_quality(brain, messages, known_words):
    replies, ended = [], []
    for k, message in enumerate(messages):
        text, stopped = brain.reply([message], temperature=0.8, seed=1000 + k)
        replies.append(text)
        ended.append(stopped)
    greedy = [brain.reply([m], temperature=0)[0] for m in messages]
    words = [re.findall(r"[a-z']+", r.lower()) for r in replies]
    all_words = [w.strip("'") for ws in words for w in ws if w.strip("'")]
    repeats = [1 - len(set(zip(ws, ws[1:], ws[2:]))) / max(1, len(ws) - 2) for ws in words if len(ws) >= 3]
    return {
        "ends_by_itself_16": float(np.mean(ended)),
        "average_length": float(np.mean([len(r) for r in replies])),
        "real_word_rate": float(np.mean([w in known_words for w in all_words])) if all_words else 0.0,
        "repeated_trigrams": float(np.mean(repeats)) if repeats else 0.0,
        "distinct_greedy_replies": len(set(greedy)) / len(greedy),
    }, replies, greedy


# ----------------------------------------------------------------------- the data
def load_mix(category, split):
    with open(MIX / f"{category}_{split}.jsonl", encoding="utf-8") as f:
        return [Example(r["context"], r["reply"], r["source"]) for r in map(json.loads, f)]


def problem_key(message):
    """(op, a, b) from an arithmetic message in any of our wordings."""
    numbers = [int(n) for n in re.findall(r"\d+", message)]
    low = message.lower()
    if "subtract" in low or "take" in low:
        return "-", numbers[1], numbers[0]
    op = "-" if ("-" in low or "minus" in low) else "+"
    return op, numbers[0], numbers[1]


def evaluation_sets(seed=0, n_pairs=600, n_pick=200, n_arith=400, n_tasks=300):
    rng = random.Random(seed)
    dialogue = [(ex.context, ex.reply) for ex in load_mix("dialogue", "val")]
    arithmetic = load_mix("arithmetic", "val")
    stage15_trained = {p[:3] for p in split_problems(arithmetic_problems())[0]}      # Stage 1.5's training problems
    never_seen = [ex for ex in arithmetic if ex.source == "arithmetic:new problem"
                  and problem_key(ex.context[0]) not in stage15_trained]
    new_wording = [ex for ex in arithmetic if ex.source == "arithmetic:new wording"]
    tasks = load_mix("tasks", "val")
    context = load_mix("context", "val") if (MIX / "context_val.jsonl").exists() else []
    memory = load_mix("memory", "val") if (MIX / "memory_val.jsonl").exists() else []
    return {
        "dialogue": rng.sample(dialogue, min(n_pairs, len(dialogue))),
        "pick": None,
        "arith_never_seen": rng.sample(never_seen, min(n_arith, len(never_seen))),
        "arith_new_wording": rng.sample(new_wording, min(n_arith // 2, len(new_wording))),
        "tasks": rng.sample(tasks, min(n_tasks, len(tasks))),
        "context": random.Random(seed + 1).sample(context, min(n_tasks, len(context))),
        "memory": random.Random(seed + 2).sample(memory, min(n_tasks, len(memory))),
        "conversation": load_mix("conversation", "val"),
        "probe": TEST_MESSAGES + [ex.context[0] for ex in never_seen[:12]] + [ex.context[0] for ex in tasks[:12]]
                 + [c[-1] for c, _ in dialogue[:12]],
        "counts": {"never_seen_problems": len(never_seen)},
    } | {"pick_size": n_pick}


def baselines():
    """What no learning would score: always answering with the most common training answer."""
    arith_train = load_mix("arithmetic", "train")
    tasks_train = load_mix("tasks", "train")
    for extra in ("context", "memory"):
        if (MIX / f"{extra}_train.jsonl").exists():
            tasks_train += load_mix(extra, "train")
    common = Counter(ex.reply for ex in arith_train).most_common(1)[0][0]
    per_type = {}
    for ex in tasks_train:
        per_type.setdefault(ex.source, Counter())[ex.reply] += 1
    return common, {t: c.most_common(1)[0][0] for t, c in per_type.items()}


def evaluate_all(brains, n_pairs=600, n_pick=200, n_arith=400, n_tasks=300, n_sampled=150, seed=0,
                 with_generation=True, log=print):
    sets = evaluation_sets(seed, n_pairs, n_pick, n_arith, n_tasks)
    common_answer, task_majority = baselines()
    base = {
        "arith_exact_greedy": float(np.mean([ex.reply == common_answer for ex in sets["arith_never_seen"]])),
        "tasks_exact_greedy": float(np.mean([ex.reply == task_majority.get(ex.source) for ex in sets["tasks"]])),
        "context_exact_greedy": float(np.mean([ex.reply == task_majority.get(ex.source) for ex in sets["context"]]))
        if sets["context"] else None,
        "memory_exact_greedy": float(np.mean([ex.reply == task_majority.get(ex.source) for ex in sets["memory"]]))
        if sets["memory"] else None,
        "pick_the_reply": 0.10,
    }
    common = set.intersection(*(set(b.tok.chars) for b in brains))
    val_text = "".join(ch for ch in (PROCESSED / "val.txt").read_text(encoding="utf-8") if ch in common)
    starts = np.random.default_rng(seed).integers(0, len(val_text) - 258, size=120)
    windows = [val_text[s:s + 257] for s in starts]
    known_words = set(re.findall(r"[a-z]+", (PROCESSED / "train.txt").read_text(encoding="utf-8").lower()))
    dialogue = [(c, r) for c, r in sets["dialogue"] if all(ch in common for ch in r)]
    many_messages = TEST_MESSAGES + [c[-1] for c, _ in dialogue[:100]]

    results, samples = {}, {}
    for brain in brains:
        t = time.perf_counter()
        log(f"\nMeasuring {brain.name} ...")
        r = {}
        nll, chars = brain.book_nll(windows)
        r["book_loss"] = nll / chars
        r["reply_loss"] = per_char_loss(brain, dialogue)
        r["input_use"] = input_use(brain, dialogue, random.Random(seed))
        r["pick_the_reply"] = pick_the_reply(brain, dialogue[:sets["pick_size"]], random.Random(seed))
        r["arith_p_correct"] = probability_of_correct(brain, sets["arith_never_seen"])
        r["tasks_p_correct"] = probability_of_correct(brain, sets["tasks"])
        r["arith_loss"] = per_char_loss(brain, [(ex.context, ex.reply) for ex in sets["arith_never_seen"]])
        r["tasks_loss"] = per_char_loss(brain, [(ex.context, ex.reply) for ex in sets["tasks"]])
        if sets["context"]:
            r["context_p_correct"] = probability_of_correct(brain, sets["context"])
            r["context_loss"] = per_char_loss(brain, [(ex.context, ex.reply) for ex in sets["context"]])
        if sets["memory"]:
            r["memory_p_correct"] = probability_of_correct(brain, sets["memory"])
            r["memory_loss"] = per_char_loss(brain, [(ex.context, ex.reply) for ex in sets["memory"]])
        r["message_sensitivity"] = message_sensitivity(brain, sets["probe"])
        r["format_carry_over"], r["digit_without_arithmetic_turn"] = format_carry_over(brain, sets["arith_never_seen"])
        log(f"  model quality: book {r['book_loss']:.3f}, reply {r['reply_loss']:.3f}, input use {r['input_use']:+.3f}, "
            f"pick {100 * r['pick_the_reply']:.1f}%, P(correct) arithmetic {r['arith_p_correct']:.3f} / tasks "
            f"{r['tasks_p_correct']:.3f}, sensitivity {r['message_sensitivity']:.3f}, carry-over "
            f"{100 * r['format_carry_over']:.0f}%")
        if with_generation:
            r["arith_exact_greedy"] = exact(brain, sets["arith_never_seen"])
            r["arith_wording_greedy"] = exact(brain, sets["arith_new_wording"])
            r["arith_exact_sampled"] = exact(brain, sets["arith_never_seen"][:n_sampled], temperature=0.8)
            tasks_right = exact_list(brain, sets["tasks"])
            r["tasks_exact_greedy"] = float(np.mean(tasks_right))
            r["tasks_by_type"] = by_type(sets["tasks"], tasks_right)
            if sets["context"]:
                context_right = exact_list(brain, sets["context"])
                r["context_exact_greedy"] = float(np.mean(context_right))
                r["context_by_type"] = by_type(sets["context"], context_right)
            if sets["memory"]:
                memory_right = exact_list(brain, sets["memory"])
                r["memory_exact_greedy"] = float(np.mean(memory_right))
                r["memory_by_type"] = by_type(sets["memory"], memory_right)
            r["tasks_exact_sampled"] = exact(brain, sets["tasks"][:n_sampled], temperature=0.8)
            r["conversation_exact_greedy"] = exact(brain, sets["conversation"])
            quality, replies, greedy = reply_quality(brain, TEST_MESSAGES, known_words)
            r.update(quality)
            r.update(generation_stats(brain, many_messages))
            r["your_conversation"] = replay(brain, USER_CONVERSATION)
            samples[brain.name] = {m: (greedy[k], replies[k]) for k, m in enumerate(TEST_MESSAGES)}
            log(f"  decoding: arithmetic greedy {100 * r['arith_exact_greedy']:.1f}% (new wording "
                f"{100 * r['arith_wording_greedy']:.1f}%) sampled {100 * r['arith_exact_sampled']:.1f}%, tasks greedy "
                f"{100 * r['tasks_exact_greedy']:.1f}% sampled {100 * r['tasks_exact_sampled']:.1f}%, context "
                f"{100 * r.get('context_exact_greedy', 0):.1f}%, conversation "
                f"{100 * r['conversation_exact_greedy']:.0f}%, ends by itself {100 * r['ends_by_itself']:.0f}%, "
                f"repetition {100 * r['repetition_rate']:.0f}%")
        log(f"  ({time.perf_counter() - t:.0f} s)")
        results[brain.name] = r
    info = {"baselines": base, "never_seen_problems": sets["counts"]["never_seen_problems"],
            "sizes": {"dialogue": len(dialogue), "arith": len(sets["arith_never_seen"]),
                      "arith_wording": len(sets["arith_new_wording"]), "tasks": len(sets["tasks"]),
                      "context": len(sets["context"]), "memory": len(sets["memory"]),
                      "conversation": len(sets["conversation"]),
                      "messages": len(many_messages)}}
    return results, samples, info


# ---------------------------------------------------------------------------- report
ROWS = [  # key, label, better when higher?, kind
    ("book_loss", "Book loss per character (unseen books)", False, "model"),
    ("reply_loss", "Reply loss per character (unseen literary dialogue)", False, "model"),
    ("input_use", "Input use: wrong-message loss minus right-message loss", True, "model"),
    ("pick_the_reply", "Pick the real reply out of 10", True, "model"),
    ("arith_loss", "Arithmetic loss per character (never-seen problems)", False, "model"),
    ("tasks_loss", "Tasks loss per character (never seen)", False, "model"),
    ("context_loss", "Context loss per character (answer in an earlier turn, never seen)", False, "model"),
    ("memory_loss", "Memory-use loss per character (your favorites, likes, pets, family, places)", False, "model"),
    ("arith_p_correct", "P(exact correct answer), arithmetic never seen", True, "model"),
    ("tasks_p_correct", "P(exact correct answer), tasks never seen", True, "model"),
    ("context_p_correct", "P(exact correct answer), context questions never seen", True, "model"),
    ("memory_p_correct", "P(exact correct answer), memory-use questions (values never trained)", True, "model"),
    ("message_sensitivity", "Message changes the distribution (Jensen-Shannon)", True, "model"),
    ("format_carry_over", "Digit after an arithmetic turn, for non-numeric questions", False, "model"),
    ("arith_exact_greedy", "Arithmetic never seen: exact, greedy", True, "decoding"),
    ("arith_wording_greedy", "Arithmetic in never-seen wordings: exact, greedy", True, "decoding"),
    ("arith_exact_sampled", "Arithmetic never seen: exact, sampled (T 0.8)", True, "decoding"),
    ("tasks_exact_greedy", "Tasks never seen: exact, greedy", True, "decoding"),
    ("tasks_exact_sampled", "Tasks never seen: exact, sampled (T 0.8)", True, "decoding"),
    ("context_exact_greedy", "Answer from an earlier turn (never-seen names/words): exact, greedy", True, "decoding"),
    ("memory_exact_greedy", "Uses what you told it (favorites, likes, pets, family, places): exact, greedy", True,
     "decoding"),
    ("conversation_exact_greedy", "Never-seen greetings mirrored exactly, greedy", True, "decoding"),
    ("ends_by_itself", "Termination: replies that end by themselves (116 messages, sampled)", True, "decoding"),
    ("repetition_rate", "Repetition: replies repeating a word at once or a 3-word phrase (116 messages, sampled)",
     False, "decoding"),
    ("most_common_reply", "Most common greedy reply's share of 116 messages (lower = less generic)", False, "decoding"),
    ("distinct_greedy_replies", "Distinct greedy replies to 16 different messages", True, "decoding"),
    ("real_word_rate", "Real-word rate (sampled)", True, "decoding"),
    ("average_length", "Average reply length, characters (sampled)", None, "decoding"),
]
PERCENT = {"pick_the_reply", "format_carry_over", "arith_exact_greedy", "arith_wording_greedy",
           "arith_exact_sampled", "tasks_exact_greedy", "tasks_exact_sampled", "conversation_exact_greedy",
           "ends_by_itself", "distinct_greedy_replies", "real_word_rate", "repeated_trigrams", "context_exact_greedy",
           "repetition_rate", "most_common_reply", "memory_exact_greedy"}


def fmt(key, value):
    if value is None:
        return "-"
    if key in PERCENT:
        return f"{100 * value:.1f}%"
    return f"{value:.1f}" if key == "average_length" else f"{value:.3f}"


def table_rows(results, info):
    names = list(results)
    rows = []
    for key, label, higher, kind in ROWS:
        values = [results[n].get(key) for n in names]
        if all(v is None for v in values):
            continue
        present = [(n, v) for n, v in zip(names, values) if v is not None]
        winners = []
        if higher is not None and len(present) > 1:
            best = fmt(key, (max if higher else min)(v for _, v in present))
            winners = [n for n, v in present if fmt(key, v) == best]
        base = info["baselines"].get(key)
        rows.append((kind, label, [fmt(key, v) for v in values], winners, fmt(key, base) if base is not None else ""))
    return names, rows


def type_tables(results):
    """Per-kind exact-match tables: [(title, kinds, {brain: {kind: value}})]."""
    out = []
    for key, title in (("tasks_by_type", "Tasks never seen, by type (exact, greedy)"),
                       ("context_by_type", "Answer from an earlier turn, by type (exact, greedy)"),
                       ("memory_by_type", "Using what you told it, by type (exact, greedy)")):
        per_brain = {n: r[key] for n, r in results.items() if r.get(key)}
        if per_brain:
            kinds = sorted({k for v in per_brain.values() for k in v})
            out.append((title, kinds, per_brain))
    return out


def render_markdown(results, samples, info, settings):
    names, rows = table_rows(results, info)
    lines = ["# Evaluation: every brain side by side", "",
             f"> Generated by `python -m brain.evaluate` on {date.today().isoformat()} ({settings}). "
             "Every test uses data that none of the brains trained on.", ""]
    for kind, title in (("model", "Model quality (from the probabilities, no sampling)"),
                        ("decoding", "Decoding quality (what generation produces)")):
        lines += [f"## {title}", "", "| Measurement | " + " | ".join(names) + " | Baseline | Best |",
                  "|---|" + "---|" * len(names) + "---|---|"]
        for k, label, values, winners, base in rows:
            if k == kind:
                best = winners[0] if len(winners) == 1 else ("tie: " + ", ".join(winners) if winners else "")
                lines.append(f"| {label} | " + " | ".join(values) + f" | {base} | {best} |")
        lines.append("")
    for title, kinds, per_brain in type_tables(results):
        lines += [f"## {title}", "", "| Type | " + " | ".join(per_brain) + " |", "|---|" + "---|" * len(per_brain)]
        for kind in kinds:
            lines.append(f"| {kind} | " + " | ".join(f"{100 * v[kind]:.0f}%" if kind in v else "-"
                                                     for v in per_brain.values()) + " |")
        lines.append("")
    conversation = {n: r["your_conversation"] for n, r in results.items() if r.get("your_conversation")}
    if conversation:
        lines += ["## Your conversation, replayed (greedy; each brain's own replies become the history)", "",
                  "| You | " + " | ".join(conversation) + " |", "|---|" + "---|" * len(conversation)]
        for k, message in enumerate(USER_CONVERSATION):
            cells = [c[k].replace("|", "/")[:70] or "(nothing)" for c in conversation.values()]
            lines.append(f"| {message} | " + " | ".join(cells) + " |")
        lines.append("")
    s = info["sizes"]
    lines += [f"Test sizes: {s['dialogue']} dialogue pairs, {s['arith']} arithmetic problems never seen by any brain "
              f"(of {info['never_seen_problems']}), {s['arith_wording']} problems in never-seen wordings, {s['tasks']} "
              f"tasks, {s.get('context', 0)} context questions, {s['conversation']} never-seen greetings, "
              f"{s.get('messages', 16)} messages for termination and repetition. Baselines: always the most common "
              "answer (arithmetic; per task type); 10% for picking 1 of 10.", "",
              "Stage 1 cannot read \"+\" (its vocabulary lacks it), and Stages 1 and 1.5 never trained on the tasks or "
              "greetings (and only Stage 1.5b2 on context questions): their scores there show the starting point.", "",
              "## Replies (greedy / sampled)", ""]
    for message in samples.get(names[-1], {}):
        lines.append(f"**You:** {message}  ")
        for n in names:
            if message in samples.get(n, {}):
                g, s_ = samples[n][message]
                lines.append(f"- *{n}:* {g or '(nothing)'} / {s_ or '(nothing)'}")
        lines.append("")
    return "\n".join(lines)


def render_html(results, samples, info, settings):
    names, rows = table_rows(results, info)
    esc = html.escape
    def table(kind):
        head = "".join(f"<th>{esc(n)}</th>" for n in names)
        body = []
        for k, label, values, winners, base in rows:
            if k != kind:
                continue
            cells = "".join(f"<td class='{'best' if n in winners else ''}'>{esc(v)}</td>" for n, v in zip(names, values))
            body.append(f"<tr><th scope='row'>{esc(label)}</th>{cells}<td class='base'>{esc(base)}</td></tr>")
        return f"<table><thead><tr><th>Measurement</th>{head}<th>Baseline</th></tr></thead><tbody>{''.join(body)}</tbody></table>"
    extra = []
    for title, kinds, per_brain in type_tables(results):
        head = "".join(f"<th>{esc(n)}</th>" for n in per_brain)
        body = "".join(f"<tr><th scope='row'>{esc(kind)}</th>" + "".join(
            f"<td>{100 * v[kind]:.0f}%</td>" if kind in v else "<td>-</td>" for v in per_brain.values()) + "</tr>"
                       for kind in kinds)
        extra.append(f"<h2>{esc(title)}</h2><div class='scroll'><table><thead><tr><th>Type</th>{head}</tr></thead>"
                     f"<tbody>{body}</tbody></table></div>")
    conversation = {n: r["your_conversation"] for n, r in results.items() if r.get("your_conversation")}
    if conversation:
        head = "".join(f"<th>{esc(n)}</th>" for n in conversation)
        body = "".join(f"<tr><th scope='row'>{esc(m)}</th>" + "".join(
            f"<td class='text'>{esc(c[k] or '(nothing)')}</td>" for c in conversation.values()) + "</tr>"
                       for k, m in enumerate(USER_CONVERSATION))
        extra.append("<h2>Your conversation, replayed</h2><p>Greedy replies; each brain's own replies become the "
                     f"history.</p><div class='scroll'><table><thead><tr><th>You</th>{head}</tr></thead>"
                     f"<tbody>{body}</tbody></table></div>")
    replies = []
    for message in samples.get(names[-1], {}):
        items = "".join(f"<li><b>{esc(n)}:</b> {esc(samples[n][message][0] or '(nothing)')} "
                        f"<span class='muted'>/ sampled: {esc(samples[n][message][1] or '(nothing)')}</span></li>"
                        for n in names if message in samples.get(n, {}))
        replies.append(f"<div class='msg'><div class='you'>{esc(message)}</div><ul>{items}</ul></div>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Brain Evaluation</title>
<style>
:root {{ --bg:#f5f6f8; --surface:#fff; --text:#16191f; --muted:#687080; --border:#e1e4e9; --accent:#4f46e5; --best:#e6f4ea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0e1014; --surface:#171a21; --text:#e7e9ed; --muted:#98a0ad; --border:#2a2f3a; --accent:#8b84ff; --best:#16301f; }} }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ max-width:1000px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:28px 0 8px; }}
p {{ color:var(--muted); margin:4px 0; }}
.scroll {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; background:var(--surface); border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
th, td {{ padding:8px 10px; border-bottom:1px solid var(--border); text-align:right; font-variant-numeric:tabular-nums; }}
th[scope=row], thead th:first-child {{ text-align:left; font-weight:500; }}
thead th {{ font-size:12px; color:var(--muted); font-weight:600; }}
td.best {{ background:var(--best); font-weight:600; }} td.base {{ color:var(--muted); }}
td.text {{ text-align:left; }}
.msg {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:10px 14px; margin:8px 0; }}
.you {{ font-weight:600; }} .muted {{ color:var(--muted); }} ul {{ margin:6px 0 0; padding-left:18px; }}
</style></head><body><main>
<h1>Brain evaluation</h1>
<p>{esc(date.today().isoformat())} &middot; {esc(settings)} &middot; every test uses data none of the brains trained on.
Green = best. Model quality comes from the network's probabilities (no sampling); decoding quality is what generation produces.</p>
<h2>Model quality</h2><div class="scroll">{table("model")}</div>
<h2>Decoding quality</h2><div class="scroll">{table("decoding")}</div>
{''.join(extra)}
<p>Baselines: always giving the most common training answer (arithmetic; per task type); 10% for picking 1 of 10.
Stage 1 cannot read "+"; Stages 1 and 1.5 never trained on the tasks or greetings, and only round 2 trained on
context questions.</p>
<h2>Replies to the same messages (greedy, then sampled)</h2>{''.join(replies)}
</main></body></html>
"""


def available_brains():
    brains = []
    first_run = STAGE15B_MODEL.with_name("stage1_5b_first.npz")      # kept for comparison, if present
    for path, name in ((STAGE1_MODEL, "Stage 1"), (STAGE15_MODEL, "Stage 1.5"),
                       (first_run, "1.5b first run"), (STAGE15B_MODEL, "Stage 1.5b"),
                       (STAGE15B2_MODEL, "Stage 1.5b2"), (STAGE15B3_MODEL, "Stage 1.5b3"),
                       (STAGE15B4_MODEL, "Stage 1.5b4"), (STAGE15B4B_MODEL, "Stage 1.5b4b"),
                       (STAGE15B4C_MODEL, "Stage 1.5b4c")):
        if name in ("1.5b first run", "Stage 1.5b2", "Stage 1.5b3", "Stage 1.5b4", "Stage 1.5b4b", "Stage 1.5b4c") and \
                not path.exists():
            continue
        if path.exists():
            model, tok, _ = load_brain(path)
            brains.append(Stage1Brain(model, tok, name) if name == "Stage 1" else ReplyBrain(model, tok, name))
        else:
            print(f"(skipping {name}: {path.relative_to(ROOT)} not found)")
    return brains


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare the brains on data none of them trained on.")
    parser.add_argument("--quick", action="store_true", help="smaller samples (faster, noisier)")
    args = parser.parse_args(argv)
    brains = available_brains()
    if not brains:
        print("No trained brains found.")
        return 1
    sizes = dict(n_pairs=200, n_pick=80, n_arith=120, n_tasks=100, n_sampled=60) if args.quick else {}
    results, samples, info = evaluate_all(brains, **sizes)
    settings = "quick run, small samples" if args.quick else "full run"
    REPORT.write_text(render_markdown(results, samples, info, settings), encoding="utf-8")
    REPORT.with_suffix(".html").write_text(render_html(results, samples, info, settings), encoding="utf-8")
    REPORT.with_suffix(".json").write_text(json.dumps({"results": results, "info": info}, indent=2), encoding="utf-8")
    print("\n" + render_markdown(results, samples, info, settings).split("## Replies")[0])
    print(f"Reports: {REPORT.relative_to(ROOT)}, evaluation.html (dashboard), evaluation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
