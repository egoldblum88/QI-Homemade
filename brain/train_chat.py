"""
Stage 1.5 training: the attention brain, in two phases.

    python -m brain.train_chat                   # both phases (about 45 minutes on an M4)
    python -m brain.train_chat --quick           # 3-minute test run (a weak brain)
    python -m brain.train_chat --finetune-only   # redo phase B only, about 30 minutes (e.g. after adding your
                                                 # own conversations); reuses the phase-A brain

PHASE A, reading. Next-character prediction on the books, like Stage 1 but with
    attention and 256 characters of context. It learns spelling, words, grammar and style.
PHASE B, conversation. Supervised training on "earlier turns -> reply" examples
    (made by tools/prepare_dialogues.py). The model reads
        <user> message <ai> reply <end>
    but is graded ONLY on the reply and <end> (the loss mask). Every 4th step is a book
    step, so it keeps its English while it learns to reply ("replay").

The weights start random: nothing about personality, opinions or identity is written in.
Whatever tendencies the model ends up with come from the training text.

Output: models/stage1_5.npz (the brain), models/stage1_5_reading.npz (after phase A),
        models/stage1_5_log.csv (the learning curve)
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from brain.dialogue import END, SPECIALS, Example, encode_example
from brain.generate import reply_to
from brain.tokenizer import CharTokenizer
from brain.train import AdamW, clip_gradients, format_time, learning_rate
from brain.transformer import TinyTransformer, masked_cross_entropy

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OWN_REPEAT = 5            # your own conversation examples count 5 times as much
BOOK_EVERY = 4            # in phase B, every 4th step is a book (replay) step
PROBES = ["Who are you?", "what is 5+5", "Where do you live?"]   # printed while training; never trained on


# --------------------------------------------------------------------------- data
def load_examples(path):
    if not path.exists():
        raise SystemExit(f"Missing {path.relative_to(ROOT)}. Build the conversation data first:\n"
                         "    python tools/prepare_dialogues.py")
    with open(path, encoding="utf-8") as f:
        return [Example(**json.loads(line)) for line in f]


def read_text(path):
    if not path.exists():
        raise SystemExit(f"Missing {path.relative_to(ROOT)}. Prepare the books first:\n"
                         "    python tools/prepare_data.py")
    return path.read_text(encoding="utf-8")


def book_batch(ids, context, batch_size, rng):
    """Random windows of book text: every position is graded (weights all 1)."""
    starts = rng.integers(0, len(ids) - context - 1, size=batch_size)
    window = ids[starts[:, None] + np.arange(context + 1)].astype(np.int64)
    return window[:, :-1], window[:, 1:], np.ones((batch_size, context), dtype=np.float32)


def pad_batch(encoded, pad_id):
    """Stack variable-length examples, padding on the right (padding is never graded)."""
    width = max(len(ids) for ids, _ in encoded) - 1
    X = np.full((len(encoded), width), pad_id, dtype=np.int64)
    Y = np.full((len(encoded), width), pad_id, dtype=np.int64)
    W = np.zeros((len(encoded), width), dtype=np.float32)
    for row, (ids, weights) in enumerate(encoded):
        n = len(ids) - 1
        X[row, :n], Y[row, :n], W[row, :n] = ids[:-1], ids[1:], weights[1:]
    return X, Y, W


def dialogue_batches(encoded, batch_size, rng):
    """Endless batches of similar-length examples (less padding = faster), in random order."""
    order = np.argsort([len(ids) for ids, _ in encoded], kind="stable")
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    while True:
        for c in rng.permutation(len(chunks)):
            yield [encoded[i] for i in chunks[c]]


def step(model, optimizer, X, Y, W, lr):
    """One learning step on one batch. Returns the loss before the update."""
    loss, dlogits = masked_cross_entropy(model.forward(X), Y, W)
    grads = model.backward(dlogits)
    clip_gradients(grads, 1.0)
    optimizer.step(model.params, grads, lr)
    return loss


def mean_loss(model, batches):
    total = weight = 0.0
    for X, Y, W in batches:
        loss, _ = masked_cross_entropy(model.forward(X), Y, W)
        total += loss * float(W.sum())
        weight += float(W.sum())
    return total / max(weight, 1.0)


def matrices_only(name):
    """Weight decay applies to weight matrices, not to biases, norms or embeddings."""
    return name.endswith(".W")


# --------------------------------------------------------------------------- phases
def run_phase(name, model, optimizer, steps, peak_lr, next_batch, evaluate, save, log_rows):
    print(f"\n{name}: {steps:,} steps")
    started, recent, best = time.perf_counter(), [], float("inf")
    every = max(1, steps // 12)
    first_speed, last_time, last_step = None, started, 0
    for i in range(1, steps + 1):
        try:
            loss = step(model, optimizer, *next_batch(i), learning_rate(i - 1, steps, peak_lr))
        except KeyboardInterrupt:
            raise SystemExit(f"\nStopped by you at step {i} of {name}. The best brain of this phase so far "
                             "is saved; run the command again to train fully.")
        if not math.isfinite(loss) or loss > 20:
            raise SystemExit(f"Training became unstable at step {i} (loss {loss}). "
                             "Try a smaller learning rate, e.g. --reading-lr 0.002")
        recent.append(loss)
        if i % every == 0 or i == steps:
            now = time.perf_counter()
            speed = (now - last_time) / (i - last_step)            # seconds per step lately
            first_speed = first_speed or speed
            last_time, last_step = now, i
            scores = evaluate()
            main_score = scores["val"]
            elapsed = time.perf_counter() - started
            marker = ""
            if main_score < best:
                best, marker = main_score, "  (saved)"
                save(main_score)
            extra = "".join(f" | {k} {v:.3f}" for k, v in scores.items() if k != "val")
            print(f"  step {i:>5}/{steps} | train {sum(recent) / len(recent):.3f} | val {main_score:.3f}"
                  f"{extra} | {speed:.2f} s/step | {format_time(elapsed)} elapsed, "
                  f"~{format_time(speed * (steps - i))} left{marker}")
            if speed > 4 * first_speed:
                print(f"  WARNING: training is now {speed / first_speed:.0f}x slower than at the start. "
                      "Close other heavy programs (a second training, many browser tabs). If it stays slow, "
                      "press Ctrl+C (the best brain so far is saved) and run the command again.")
            log_rows.append([name, i, round(sum(recent) / len(recent), 4), round(main_score, 4),
                             round(elapsed, 1)])
            recent = []
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the Stage 1.5 attention brain.")
    parser.add_argument("--reading-steps", type=int, default=2400, help="phase A steps (books)")
    parser.add_argument("--chat-steps", type=int, default=10000, help="phase B steps (conversations)")
    parser.add_argument("--reading-lr", type=float, default=4e-3, help="phase A peak learning rate")
    parser.add_argument("--chat-lr", type=float, default=1.5e-3, help="phase B peak learning rate")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="3-minute test run")
    parser.add_argument("--finetune-only", action="store_true", help="skip phase A; reuse the reading brain")
    parser.add_argument("--out", default=str(MODELS / "stage1_5.npz"))
    args = parser.parse_args(argv)
    if args.quick:
        args.reading_steps, args.chat_steps = 150, 150

    out = Path(args.out)
    reading_path = out.with_name(out.stem + "_reading.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- data: books (phase A) and conversations (phase B)
    train_text = read_text(PROCESSED / "train.txt")
    val_text = read_text(PROCESSED / "val.txt")
    train_ex = load_examples(PROCESSED / "dialogues_train.jsonl")
    val_ex = load_examples(PROCESSED / "dialogues_val.jsonl")
    all_dialogue_text = "".join(" ".join(ex.context) + ex.reply for ex in train_ex)
    if args.finetune_only:
        if not reading_path.exists():
            raise SystemExit(f"No reading brain at {reading_path.relative_to(ROOT)}. Run without --finetune-only.")
        model, record = TinyTransformer.load(reading_path)
        tokenizer = CharTokenizer(record["vocab"], record["specials"])
    else:
        tokenizer = CharTokenizer.from_text(train_text + all_dialogue_text, SPECIALS)
        model = TinyTransformer(tokenizer.vocab_size, args.context, args.dim, args.layers, args.heads, seed=args.seed)
    train_ids = np.array(tokenizer.encode(train_text), dtype=np.uint8)
    val_ids = np.array(tokenizer.encode(val_text), dtype=np.uint8)

    own = [ex for ex in train_ex if ex.source.startswith("own:")]
    encoded_train = [encode_example(tokenizer, ex, model.context) for ex in train_ex + own * (OWN_REPEAT - 1)]
    encoded_val = [encode_example(tokenizer, ex, model.context) for ex in val_ex]
    pad = tokenizer.special(END)

    print("Stage 1.5 training: the attention brain\n")
    print(f"  vocabulary: {len(tokenizer.chars)} characters + {len(tokenizer.specials)} markers "
          f"{tokenizer.specials} = {tokenizer.vocab_size} tokens")
    print(f"  books:         {len(train_ids):>10,} training characters, {len(val_ids):,} validation")
    print(f"  conversations: {len(train_ex):>10,} training examples ({len(own)} yours, counted x{OWN_REPEAT}), "
          f"{len(val_ex):,} validation")
    print(f"  model: {model.config['layers']} blocks x {model.config['dim']} wide, {model.config['heads']} "
          f"attention heads, sees {model.context} characters: {model.num_parameters():,} parameters")
    if args.finetune_only:
        print("  (continuing from the phase-A reading brain; nothing about personality is written in)")
    else:
        print("  (all weights start random; nothing about personality is written in)")

    # fixed validation sets, so every check measures the same thing
    val_rng = np.random.default_rng(123)
    val_books = [book_batch(val_ids, model.context, 16, val_rng) for _ in range(6)]
    val_chat = [pad_batch(encoded_val[i:i + 64], pad)
                for i in range(0, min(len(encoded_val), 768), 64)]
    log_rows = []
    info = {"stage": 1.5, "trained_on": date.today().isoformat(),
            "train_chars": int(len(train_ids)), "dialogue_examples": len(train_ex),
            "train_sha256": hashlib.sha256(train_text.encode()).hexdigest()[:16]}

    def save_to(path, extra):
        def save(score):
            model.save(path, tokenizer.chars, tokenizer.specials, {**info, **extra(score)})
        return save

    # ---- phase A: reading
    if not args.finetune_only:
        optimizer = AdamW(model.params, weight_decay=0.1, decay=matrices_only)
        run_phase("PHASE A, reading the books", model, optimizer, args.reading_steps, args.reading_lr,
                  lambda i: book_batch(train_ids, model.context, args.batch, rng),
                  lambda: {"val": mean_loss(model, val_books)},
                  save_to(reading_path, lambda s: {"phase": "reading", "val_book_loss": round(s, 4)}),
                  log_rows)
        model, _ = TinyTransformer.load(reading_path)     # continue from the best reading brain

    # ---- phase B: conversation
    optimizer = AdamW(model.params, weight_decay=0.1, decay=matrices_only)
    batches = dialogue_batches(encoded_train, args.batch, rng)

    def next_batch(i):
        if i % BOOK_EVERY == 0:
            return book_batch(train_ids, model.context, args.batch // 2, rng)
        return pad_batch(next(batches), pad)

    def evaluate():
        return {"val": mean_loss(model, val_chat), "book": mean_loss(model, val_books)}

    run_phase("PHASE B, learning to reply (graded only on the replies)", model, optimizer,
              args.chat_steps, args.chat_lr, next_batch, evaluate,
              save_to(out, lambda s: {"phase": "conversation", "val_reply_loss": round(s, 4),
                                      "val_book_loss": round(mean_loss(model, val_books), 4)}),
              log_rows)

    # ---- the best brain, a few sample replies
    model, record = TinyTransformer.load(out)
    with open(out.with_name(out.stem + "_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "step", "train_loss", "val_loss", "seconds"])
        writer.writerows(log_rows)
    print(f"\nSaved {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: reply loss "
          f"{record['val_reply_loss']} on validation conversations, book loss {record['val_book_loss']}")
    print("\nA few replies (messages the model never trained on):")
    for message in PROBES:
        result, _ = reply_to(model, tokenizer, [message], temperature=0, loop_penalty=0.3)
        print(f"  USER: {message}\n  AI:   {result.text}")
    print("\nNext:  python -m brain.evaluate      (compare with Stage 1)")
    print("       python -m chat.server         (talk to it)")
    return 0



if __name__ == "__main__":
    sys.exit(main())
