"""
Stage 1.5b training: one attention brain, trained on FIVE clearly separated data categories.

    python -m brain.train_mix                      # the full run -> models/stage1_5b2.npz (about 70 minutes)
    python -m brain.train_mix --minutes 7 --out models/try.npz    # a short, time-limited run
    python -m brain.train_mix --position absolute --untied         # the Stage 1.5b network design
    python -m brain.train_mix --init-from models/stage1_5b2.npz --steps 5000 --lr 0.001 \
        --mix "book=0.22,dialogue=0.18,conversation=0.04,arithmetic=0.18,tasks=0.14,context=0.12,memory=0.12" \
        --out models/stage1_5b3.npz                                  # round 3: using what you told it
    python -m brain.train_mix --init-from models/stage1_5b3.npz --steps 5000 --lr 0.0007 \
        --mix "book=0.18,dialogue=0.14,conversation=0.03,arithmetic=0.17,tasks=0.12,context=0.10,memory=0.10,answer=0.16" \
        --out models/stage1_5b4.npz                                  # round 4: saying a reasoning result (D-068)
    (round 4b: the same, with facts about you in the answer data,
     --stage-name 1.5b4b --out models/stage1_5b4b.npz)
    python -m brain.train_mix --tokenizer char     # compare: one token per character

What it learns from (made by tools/prepare_mix.py), and how often, is explicit:

    category        share of steps   what it teaches
    book language        25%         English itself (the books' text; every position graded)
    dialogue             20%         literary exchanges (books and plays)
    conversation          5%         neutral conventions + your own conversation files
    arithmetic           22%         a+b and a-b (0-99) in many wordings
    tasks                16%         answers that can only come from reading the message
    context              12%         answers that are in an EARLIER turn (round 2)

(The size of the network was chosen by measured 7-minute comparison runs, D-037 and D-038. Round 2
added the context category and took its share from arithmetic and tasks, D-046.)

For every category except book language, the network reads "<user> message <ai> reply <end>"
and is graded ONLY on the reply (the loss mask). For the other examples, half the time 1-2
unrelated earlier turns are put in front ("mixed contexts"), so the model learns that the reply
depends on the LAST message, not on the format of whatever came before. The context category is
the counterweight: there the question asks for something from an earlier turn.

The network design (round 2, chosen by `python -m brain.diagnose copy` and 7-minute runs):
rotary (relative) positions and a tied output layer; --position absolute --untied gives Stage 1.5b's.

Progress lines show a separate validation score per category, measured PER CHARACTER (so the
character tokenizer and our subword tokenizer can be compared fairly), plus exact-match accuracy
on arithmetic problems and tasks it never trained on.

The weights start random: no personality, name, opinions or goals are written in.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from brain.bpe import BPETokenizer
from brain.dialogue import END, SPECIALS, Example, encode_example
from brain.generate import reply_to
from brain.tokenizer import CharTokenizer
from brain.train import AdamW, clip_gradients, format_time, learning_rate
from brain.train_chat import matrices_only, pad_batch
from brain.transformer import TinyTransformer, masked_cross_entropy

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MIX = PROCESSED / "mix"
MODELS = ROOT / "models"
CATEGORIES = ["dialogue", "conversation", "arithmetic", "tasks", "context", "memory", "answer"]
DEFAULT_MIX = {"book": 0.25, "dialogue": 0.20, "conversation": 0.05, "arithmetic": 0.22, "tasks": 0.16,
               "context": 0.12}
# Every category except book language gets unrelated earlier turns half the time. (The first
# Stage 1.5b run left dialogue out, and the model learned "after arithmetic comes a number" for
# every ordinary question: 89-99% digits. See docs/DECISIONS.md D-041.)
# (Not the context and memory categories: their earlier turns ARE the point, so they are never replaced.)
MIXED_CONTEXT = {"dialogue", "conversation", "arithmetic", "tasks"}
TOKENS_PER_STEP = 4096                                       # every step costs about the same
BEST_BY = ["book", "dialogue", "arithmetic", "tasks", "context", "memory", "answer"]   # (conversation: tiny)


def load(category, split, required=True):
    path = MIX / f"{category}_{split}.jsonl"
    if not path.exists() and not required:
        return []
    if not path.exists():
        raise SystemExit(f"Missing {path.relative_to(ROOT)}. Build the data first:\n"
                         "    python tools/prepare_mix.py")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [Example(r["context"], r["reply"], r["source"]) for r in rows]


def build_tokenizer(kind, text, bpe_vocab):
    if kind == "char":
        return CharTokenizer.from_text(text, SPECIALS)
    return BPETokenizer.train(text, bpe_vocab, SPECIALS)


class Mixer:
    """Makes one batch at a time from a chosen category."""

    def __init__(self, tokenizer, context, train_sets, book_ids, rng):
        self.tok, self.context, self.sets, self.book_ids, self.rng = tokenizer, context, train_sets, book_ids, rng
        self.pad = tokenizer.special(END)
        # earlier turns to put in front: SHORT exchanges, kept per category so that every
        # category is equally likely to appear before a message (not mostly arithmetic)
        self.distractors = {c: [(ex.context[-1], ex.reply) for ex in train_sets[c]
                                if len(ex.context[-1]) <= 80 and len(ex.reply) <= 80]
                            for c in train_sets if c in MIXED_CONTEXT}
        self.distractors = {c: pool for c, pool in self.distractors.items() if pool}

    def with_mixed_context(self, ex):
        """Half the time: replace the earlier turns with 1-2 unrelated exchanges from any category."""
        if self.rng.random() < 0.5:
            return ex
        before = []
        for _ in range(self.rng.randint(1, 2)):
            pool = self.distractors[self.rng.choice(list(self.distractors))]
            before += list(self.rng.choice(pool))
        return Example(before + ex.context[-1:], ex.reply, ex.source)

    def batch(self, category):
        if category == "book":
            n = max(1, TOKENS_PER_STEP // self.context)
            starts = self.np_rng().integers(0, len(self.book_ids) - self.context - 1, size=n)
            window = self.book_ids[starts[:, None] + np.arange(self.context + 1)].astype(np.int64)
            return window[:, :-1], window[:, 1:], np.ones((n, self.context), dtype=np.float32)
        examples = self.sets[category]
        sample = [examples[self.rng.randrange(len(examples))] for _ in range(8)]
        typical = max(8, int(np.mean([len(e.context[-1]) + len(e.reply) for e in sample]) // 2 + 4))
        n = int(min(256, max(32, TOKENS_PER_STEP // typical)))
        chosen = [examples[self.rng.randrange(len(examples))] for _ in range(n)]
        if category in MIXED_CONTEXT:
            chosen = [self.with_mixed_context(ex) for ex in chosen]
        encoded = [encode_example(self.tok, ex, self.context) for ex in chosen]
        width = max(len(ids) for ids, _ in encoded)
        keep = max(16, int(1.5 * TOKENS_PER_STEP // width))     # cap the batch at its planned size
        return pad_batch(encoded[:keep], self.pad)

    def np_rng(self):
        return np.random.default_rng(self.rng.randrange(2**32))


def token_char_lengths(tokenizer):
    """How many characters each token stands for (special markers count as 1)."""
    return np.array([len(tokenizer.decode([i])) or 1 for i in range(tokenizer.vocab_size)], dtype=np.float64)


def loss_per_char(model, batches, char_len):
    """Total loss divided by the number of CHARACTERS predicted: comparable across tokenizers."""
    total = chars = 0.0
    for X, Y, W in batches:
        logits = model.forward(X).astype(np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        nll = -np.take_along_axis(log_probs, Y[..., None], axis=-1)[..., 0]
        total += float((nll * W).sum())
        chars += float((char_len[Y] * W).sum())
    return total / max(chars, 1.0)


def load_checks(category, split):
    """(example, what its reply must contain) for the answer category's training check."""
    path = MIX / f"{category}_{split}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [(Example(r["context"], r["reply"], r["source"]), r.get("check", [])) for r in rows]


def says_it(model, tokenizer, items):
    """Share of greedy replies that contain everything they must say (the answer, or "not told")."""
    right = 0
    for ex, check in items:
        reply = reply_to(model, tokenizer, ex.context, temperature=0, loop_penalty=1.0)[0].text.lower()
        right += all(c.lower() in reply for c in check)
    return right / max(1, len(items))


def exact_match(model, tokenizer, examples):
    right = sum(reply_to(model, tokenizer, ex.context, temperature=0, loop_penalty=1.0)[0].text == ex.reply
                for ex in examples)
    return right / max(1, len(examples))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the Stage 1.5b brain on the five-category mix.")
    parser.add_argument("--tokenizer", choices=["bpe", "char"], default="bpe")
    parser.add_argument("--bpe-vocab", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--position", choices=["rotary", "absolute"], default="rotary",
                        help="rotary = relative positions (round 2); absolute = Stage 1.5b's")
    parser.add_argument("--untied", action="store_true",
                        help="a separate output table (Stage 1.5b's design) instead of the tied one")
    parser.add_argument("--steps", type=int, default=12000, help="training steps (ignored with --minutes)")
    parser.add_argument("--minutes", type=float, default=None, help="train for this long instead of --steps")
    parser.add_argument("--lr", type=float, default=3e-3, help="peak learning rate")
    parser.add_argument("--mix", default=None, help='e.g. "book=0.25,dialogue=0.25,conversation=0.05,'
                                                    'arithmetic=0.25,tasks=0.2"')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=str(MODELS / "stage1_5b2.npz"))
    parser.add_argument("--tokenizer-from", default=str(MODELS / "stage1_5b.npz"),
                        help="reuse this brain's subword tokens (so a comparison changes only the network and "
                             "data); 'new' learns a new tokenizer")
    parser.add_argument("--stage-name", default=None, help="the stage this brain is labeled with (e.g. 1.5b4b)")
    parser.add_argument("--init-from", default=None,
                        help="continue training a saved Stage 1.5b brain (same tokenizer and shape)")
    args = parser.parse_args(argv)

    mix = dict(DEFAULT_MIX)
    if args.mix:
        mix.update({k: float(v) for k, v in (part.split("=") for part in args.mix.split(","))})
    total = sum(mix.values())
    mix = {k: v / total for k, v in mix.items() if v > 0}
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = {MODELS / n for n in ("stage1.npz", "stage1_5.npz", "stage1_5b.npz", "stage1_5b_first.npz",
                                 "stage1_5b2.npz", "stage1_5b3.npz", "stage1_5b4.npz", "stage1_5b4b.npz",
                                 "stage1_5b4b_interrupted.npz", "stage1_5_reading.npz", "stage1_5_short.npz",
                                 "stage1_5b4c.npz")}
    if out.exists() and out.resolve() in {k.resolve() for k in kept}:
        raise SystemExit(f"{out.name} is an earlier stage's brain, kept for comparison. Choose another --out.")

    # ---- data, tokenizer
    book_train = (PROCESSED / "train.txt").read_text(encoding="utf-8")
    book_val = (PROCESSED / "val.txt").read_text(encoding="utf-8")
    optional = {"context", "memory", "answer"}         # needed only when training uses them
    train_sets = {c: load(c, "train", required=(c not in optional or mix.get(c, 0) > 0)) for c in CATEGORIES}
    val_sets = {c: load(c, "val", required=(c not in optional or mix.get(c, 0) > 0)) for c in CATEGORIES}
    mix_text = "".join(" ".join(ex.context) + " " + ex.reply + "\n" for c in CATEGORIES for ex in train_sets[c])
    started_setup = time.perf_counter()
    if args.init_from:                             # continue a saved brain with its own tokenizer
        from brain.generate import load_brain
        model, tokenizer, init_record = load_brain(args.init_from)
        args.tokenizer = init_record.get("tokenizer_kind", "bpe")
    else:
        source = Path(args.tokenizer_from) if args.tokenizer_from != "new" else None
        if args.tokenizer == "bpe" and source is not None and source.exists():
            from brain.generate import load_brain
            _, tokenizer, _ = load_brain(source)
            print(f"  (reusing the subword tokens of {source.name})")
        else:
            tokenizer = build_tokenizer(args.tokenizer, book_train + mix_text, args.bpe_vocab)
        model = TinyTransformer(tokenizer.vocab_size, args.context, args.dim, args.layers, args.heads, seed=args.seed,
                                position=args.position, tied=not args.untied)
    book_ids = np.array(tokenizer.encode(book_train), dtype=np.uint16)
    val_ids = np.array(tokenizer.encode(book_val), dtype=np.uint16)
    args.context = model.context
    char_len = token_char_lengths(tokenizer)

    print("Stage 1.5b training: the attention brain on separate data categories\n")
    kind = ("subword (our BPE)" if args.tokenizer == "bpe" else "character")
    print(f"  tokenizer: {kind}, {tokenizer.vocab_size:,} tokens; the books are "
          f"{len(book_train) / len(book_ids):.2f} characters per token")
    c = model.config
    start = (f"continuing from {Path(args.init_from).name}" if args.init_from
             else "all start random")
    print(f"  model: {c['layers']} blocks x {c['dim']} wide, {c['heads']} heads, {c['context']}-token context, "
          f"{model.position} positions, {'tied' if model.tied else 'separate'} output layer: "
          f"{model.num_parameters():,} parameters ({start}; no personality written in)")
    print("  data (share of training steps):")
    print(f"    {'book language':14} {100 * mix.get('book', 0):>4.0f}%   {len(book_train):>11,} characters")
    for c in CATEGORIES:
        print(f"    {c:14} {100 * mix.get(c, 0):>4.0f}%   {len(train_sets[c]):>11,} examples")
    print(f"  (setup took {time.perf_counter() - started_setup:.0f} s)")

    # ---- fixed validation sets
    vrng = np.random.default_rng(123)
    n_win = max(4, 24576 // args.context)
    starts = vrng.integers(0, len(val_ids) - args.context - 1, size=n_win)
    windows = val_ids[starts[:, None] + np.arange(args.context + 1)].astype(np.int64)
    val_batches = {"book": [(windows[i:i + 16, :-1], windows[i:i + 16, 1:],
                             np.ones((len(windows[i:i + 16]), args.context), dtype=np.float32))
                            for i in range(0, n_win, 16)]}
    pad = tokenizer.special(END)
    for c in [c for c in CATEGORIES if val_sets[c]]:
        sample = random.Random(7).sample(val_sets[c], min(512, len(val_sets[c])))
        enc = sorted((encode_example(tokenizer, ex, args.context) for ex in sample), key=lambda e: len(e[0]))
        val_batches[c] = [pad_batch(enc[i:i + 64], pad) for i in range(0, len(enc), 64)]
    arith_probe = [ex for ex in val_sets["arithmetic"] if ex.source == "arithmetic:new problem"][:80]
    tasks_probe = random.Random(8).sample(val_sets["tasks"], min(80, len(val_sets["tasks"])))
    context_probe = random.Random(9).sample(val_sets["context"], min(80, len(val_sets["context"])))
    memory_probe = random.Random(10).sample(val_sets["memory"], min(80, len(val_sets["memory"])))
    answer_items = load_checks("answer", "val") if mix.get("answer", 0) > 0 else []
    answer_probe = random.Random(11).sample(answer_items, min(80, len(answer_items)))

    mixer = Mixer(tokenizer, args.context, train_sets, book_ids, rng)
    optimizer = AdamW(model.params, weight_decay=0.1, decay=matrices_only)
    categories, weights = list(mix), list(mix.values())
    info = {"stage": args.stage_name or ("1.5b4" if "answer" in mix else "1.5b3" if "memory" in mix else
            ("1.5b2" if "context" in mix else "1.5b")),
            "trained_on": date.today().isoformat(),
            "tokenizer_kind": args.tokenizer,
            "mix": mix, "train_examples": {c: len(train_sets[c]) for c in CATEGORIES},
            "dialogue_examples": sum(len(train_sets[c]) for c in CATEGORIES)}
    if "answer" in mix:
        info["answer_format"] = "status line"      # how this brain reads a reasoning result (world/answer.py)
        if any(ex.source.startswith("answer:self") for ex in train_sets["answer"]):
            info["self_format"] = "self line"      # and the self-model's answer (self_model/say.py, D-080)
    if args.init_from:
        info["continued_from"] = {"file": Path(args.init_from).name,
                                  "its_training": init_record.get("history", []) + [init_record.get("mix")]}
    if args.tokenizer == "bpe":
        info["merges"] = [list(m) for m in tokenizer.merges]

    def save(scores):
        model.save(out, tokenizer.chars, tokenizer.specials,
                   {**info, "val_per_char": {k: round(v, 4) for k, v in scores.items()}},
                   created_by="brain/train_mix.py")

    def evaluate():
        scores = {c: loss_per_char(model, val_batches[c], char_len) for c in val_batches}
        scores["arith_exact"] = exact_match(model, tokenizer, arith_probe)
        scores["tasks_exact"] = exact_match(model, tokenizer, tasks_probe)
        scores["context_exact"] = exact_match(model, tokenizer, context_probe) if context_probe else float("nan")
        scores["memory_exact"] = exact_match(model, tokenizer, memory_probe) if memory_probe else float("nan")
        scores["answer_says"] = says_it(model, tokenizer, answer_probe) if answer_probe else float("nan")
        return scores

    # ---- training
    steps = args.steps
    budget = args.minutes * 60 if args.minutes else None
    if budget:
        steps = 10**9                        # decided after measuring the speed (below)
    every = max(50, steps // 12) if not budget else 250
    print(f"\nTraining {'for ' + format_time(budget) + ' (minutes:seconds)' if budget else f'{steps:,} steps'}. "
          "Scores are per character (lower is better); exact = never-seen problems/tasks answered exactly.")
    header_every = 10
    started = last_time = time.perf_counter()
    last_step, examples_seen, window_examples, first_rate = 0, 0, 0, None
    best, recent, log_rows = float("inf"), [], []
    i = 0
    try:
        while i < steps:
            i += 1
            category = rng.choices(categories, weights)[0]
            X, Y, W = mixer.batch(category)
            lr = learning_rate(i - 1, steps if steps < 10**9 else 10**6, args.lr, warmup=200)
            loss, dlogits = masked_cross_entropy(model.forward(X), Y, W)
            if not math.isfinite(loss) or loss > 20:
                raise SystemExit(f"Training became unstable at step {i} (loss {loss}). Try a smaller --lr.")
            grads = model.backward(dlogits)
            clip_gradients(grads, 1.0)
            optimizer.step(model.params, grads, lr)
            recent.append(loss)
            window_examples += len(X)
            if budget and i == 60:                              # measure the speed, then fix the plan
                per_step = (time.perf_counter() - started) / 60
                steps = max(120, int(budget / (per_step * 1.12)))    # ~12% of the time goes to checks
                every = max(50, steps // 12)
            if i % every == 0 or i == steps:
                now = time.perf_counter()
                rate = window_examples / (now - last_time)        # training speed only (checks excluded)
                full_window = i - last_step >= every // 2          # a tiny last window says nothing
                first_rate = first_rate or rate
                examples_seen += window_examples
                last_step, window_examples = i, 0
                scores = evaluate()
                main_score = np.mean([scores[c] for c in BEST_BY if c in scores])
                marker = ""
                if main_score < best:
                    best, marker = main_score, " (saved)"
                    save(scores)
                elapsed = time.perf_counter() - started
                left = elapsed / i * (steps - i)
                if (i // every) % header_every == 1 or every == i:
                    print(f"  {'step':>11} {'train':>6} | {'book':>5} {'dialog':>6} {'conv':>5} {'arith':>5} "
                          f"{'tasks':>5} {'contxt':>6} {'memory':>6} {'answer':>6} | {'arith':>5} {'tasks':>5} "
                          f"{'contxt':>6} {'memory':>6} {'says':>5} | {'ex/s':>5} | time")
                print(f"  {i:>5}/{steps:<5} {sum(recent) / len(recent):6.3f} | {scores['book']:5.3f} "
                      f"{scores['dialogue']:6.3f} {scores['conversation']:5.3f} {scores['arithmetic']:5.3f} "
                      f"{scores['tasks']:5.3f} {scores.get('context', float('nan')):6.3f} "
                      f"{scores.get('memory', float('nan')):6.3f} {scores.get('answer', float('nan')):6.3f} | "
                      f"{100 * scores['arith_exact']:4.0f}% {100 * scores['tasks_exact']:4.0f}% "
                      f"{100 * scores['context_exact']:5.0f}% {100 * scores['memory_exact']:5.0f}% "
                      f"{100 * scores['answer_says']:4.0f}% | "
                      f"{rate:5.0f} | {format_time(elapsed)} elapsed, ~{format_time(left)} left{marker}")
                if full_window and rate < first_rate / 4:
                    print(f"  WARNING: training is now {first_rate / rate:.0f}x slower than at the start. Close other "
                          "heavy programs. If it stays slow, press Ctrl+C (the best brain so far is saved) and rerun.")
                log_rows.append([i, round(sum(recent) / len(recent), 4)] +
                                [round(scores.get(k, float("nan")), 4)
                                 for k in ["book", *CATEGORIES, "arith_exact", "tasks_exact", "context_exact",
                                           "memory_exact", "answer_says"]] +
                                [round(rate, 1), round(elapsed, 1)])
                recent = []
                last_time = time.perf_counter()                    # restart the speed timer after the checks
    except KeyboardInterrupt:
        print(f"\nStopped by you at step {i}. The best brain so far is saved in {out.name}.")
    if not log_rows:
        print("Nothing was saved (stopped before the first check).")
        return 1
    with open(out.with_name(out.stem + "_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "book", *CATEGORIES, "arith_exact", "tasks_exact", "context_exact",
                         "memory_exact", "answer_says", "examples_per_second", "seconds"])
        writer.writerows(log_rows)
    print(f"\nSaved the best brain to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    print("Next:  python -m brain.evaluate      (every brain, side by side)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
