"""
Stage 1 training: teach the brain to predict the next character of real books.

    python -m brain.train            # full training (about 2 minutes on an M4 Mac)
    python -m brain.train --quick    # 15-second test run (weaker model)

It reads data/processed/train.txt and val.txt (made in Stage 0) and saves the trained
brain to models/stage1.npz. Chatting never trains; only this command does.

Every training step:
  1. pick BATCH random places in the training text                    (batching)
  2. input = the N characters before each place; target = the next one
  3. forward pass: probabilities for the next character               (model.forward)
  4. loss: how surprised the model was by the real next characters    (cross_entropy)
  5. backward pass: a gradient for every parameter                    (model.backward)
  6. AdamW nudges every parameter a little downhill                   (AdamW.step)
Every few hundred steps it measures the loss on the VALIDATION books, which it never
trains on, and prints a short sample so you can watch it learn.
"""

import argparse
import csv
import hashlib
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from brain.generate import PAD_CHAR, generate
from brain.model import CharMLP, cross_entropy
from brain.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "processed" / "train.txt"
VAL_FILE = ROOT / "data" / "processed" / "val.txt"
DEFAULT_OUT = ROOT / "models" / "stage1.npz"


# --------------------------------------------------------------------------- optimizer
class AdamW:
    """Adam with decoupled weight decay (Kingma & Ba 2014; Loshchilov & Hutter 2017).

    For every parameter it keeps two running averages:
      m = average gradient            ("which way has downhill been lately?")
      v = average squared gradient    ("how big are this parameter's gradients usually?")
    and steps by  lr * m / sqrt(v). Parameters with large, noisy gradients take careful
    steps; rarely-used ones (like rare characters' embeddings) still get to move.
    Weight decay gently pulls the weight matrices toward zero, which reduces overfitting.
    """

    def __init__(self, params, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01, decay=None):
        self.betas, self.eps, self.weight_decay = betas, eps, weight_decay
        # which parameters get weight decay: by default the Stage 1 names W1, W2
        self.decay = decay or (lambda name: name.startswith("W"))
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        b1, b2 = self.betas
        for name, p in params.items():
            g = grads[name]
            self.m[name] = b1 * self.m[name] + (1 - b1) * g
            self.v[name] = b2 * self.v[name] + (1 - b2) * g * g
            m_hat = self.m[name] / (1 - b1 ** self.t)      # correct the start-up bias
            v_hat = self.v[name] / (1 - b2 ** self.t)
            if self.weight_decay and self.decay(name):
                p -= lr * self.weight_decay * p
            p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_gradients(grads, max_norm):
    """If the gradient is unusually large overall, scale it down (prevents wild jumps)."""
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if norm > max_norm:
        for g in grads.values():
            g *= max_norm / norm
    return norm


def learning_rate(step, total, peak, warmup=100):
    """Warm up for `warmup` steps, then follow a cosine curve down to 10% of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# ------------------------------------------------------------------------- the data
def get_batch(ids, context, batch_size, rng, pad_id=None, pad_fraction=0.1):
    """Random training examples: X = N characters (B, N), Y = the character after each (B,).

    When writing starts, there are fewer than N characters of history, so generation pads
    the start with newlines. To teach the model what that looks like, in `pad_fraction` of
    the examples a random-length beginning of the context is replaced by newlines.
    """
    starts = rng.integers(0, len(ids) - context, size=batch_size)
    X = ids[starts[:, None] + np.arange(context)]
    Y = ids[starts + context]
    if pad_id is not None and pad_fraction > 0:
        n = int(batch_size * pad_fraction)
        cut = rng.integers(1, context, size=n)                      # how many to replace
        X[:n][np.arange(context)[None, :] < cut[:, None]] = pad_id
    return X, Y


def train_step(model, optimizer, X, Y, lr, max_grad_norm=1.0):
    """One complete learning step. Returns the loss before the update."""
    logits, cache = model.forward(X)
    loss, dlogits = cross_entropy(logits, Y)
    grads = model.backward(dlogits, cache)
    clip_gradients(grads, max_grad_norm)
    optimizer.step(model.params, grads, lr)
    return loss


def evaluate(model, ids, positions, chunk=8192):
    """Average loss at the given positions of a text (no learning happens here)."""
    total = 0.0
    for i in range(0, len(positions), chunk):
        starts = positions[i:i + chunk]
        X = ids[starts[:, None] + np.arange(model.context)]
        Y = ids[starts + model.context]
        logits, _ = model.forward(X)
        total += cross_entropy(logits, Y)[0] * len(starts)
    return total / len(positions)


# -------------------------------------------------------- counting brains (baselines)
def counting_baselines(train_ids, val_ids, vocab_size, k=0.01):
    """How well pure COUNTING predicts the validation text (lower loss is better).

    unigram: P(c) = how often c appears
    bigram:  P(c | previous char)          trigram: P(c | previous two chars)
    Each count gets +k ("add-k smoothing") so nothing has probability zero.
    """
    V = vocab_size
    t = train_ids.astype(np.int64)
    v = val_ids.astype(np.int64)
    result = {"uniform guess": math.log(V)}

    uni = np.bincount(t, minlength=V) + k
    result["unigram counts"] = float(-np.log(uni[v] / uni.sum()).mean())

    bi = np.bincount(t[:-1] * V + t[1:], minlength=V * V).reshape(V, V) + k
    bi_p = bi / bi.sum(axis=1, keepdims=True)
    result["bigram counts"] = float(-np.log(bi_p[v[:-1], v[1:]]).mean())

    tri = np.bincount((t[:-2] * V + t[1:-1]) * V + t[2:], minlength=V ** 3).reshape(V, V, V) + k
    tri_p = tri / tri.sum(axis=2, keepdims=True)
    result["trigram counts"] = float(-np.log(tri_p[v[:-2], v[1:-1], v[2:]]).mean())
    return result


def load_text(path):
    if not path.exists():
        raise SystemExit(f"Missing {path.relative_to(ROOT)}.\n"
                         "Prepare the data first (Stage 0):\n"
                         "    python tools/download_books.py\n"
                         "    python tools/prepare_data.py")
    return path.read_text(encoding="utf-8")


def format_time(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


# ----------------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the Stage 1 brain on the prepared books.")
    parser.add_argument("--steps", type=int, default=40000, help="number of training steps")
    parser.add_argument("--batch", type=int, default=512, help="examples per step")
    parser.add_argument("--lr", type=float, default=3e-3, help="peak learning rate")
    parser.add_argument("--context", type=int, default=32, help="characters the model sees")
    parser.add_argument("--embed", type=int, default=16, help="numbers per character embedding")
    parser.add_argument("--hidden", type=int, default=512, help="hidden units")
    parser.add_argument("--eval-every", type=int, default=2000, help="check progress every N steps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="short test run (about 15 seconds)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where to save the model")
    args = parser.parse_args(argv)
    if args.quick:
        args.steps, args.eval_every = 4000, 500

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1-3. tokenizer, vocabulary, numerical representation
    train_text, val_text = load_text(TRAIN_FILE), load_text(VAL_FILE)
    tokenizer = CharTokenizer.from_text(train_text)
    train_ids = np.array(tokenizer.encode(train_text), dtype=np.uint8)
    val_ids = np.array(tokenizer.encode(val_text), dtype=np.uint8)
    print("Stage 1 training: the character-level brain\n")
    print(f"  vocabulary: {tokenizer.vocab_size} characters -> ids 0..{tokenizer.vocab_size - 1}")
    print(f"  example:    'Alice' -> {tokenizer.encode('Alice')}")
    print(f"  training text:   {len(train_ids):>10,} characters")
    print(f"  validation text: {len(val_ids):>10,} characters (never trained on)\n")

    # the counting brains: what our network must beat
    baselines = counting_baselines(train_ids, val_ids, tokenizer.vocab_size)
    print("Counting baselines on the validation books (loss: lower is better)")
    for name, value in baselines.items():
        print(f"  {name:16} {value:.3f}")
    print()

    # 4. model parameters
    model = CharMLP(tokenizer.vocab_size, args.context, args.embed, args.hidden, seed=args.seed)
    optimizer = AdamW(model.params)
    print(f"Model: sees {args.context} characters -> embeddings of {args.embed} -> "
          f"{args.hidden} hidden units -> {tokenizer.vocab_size} scores")
    print(f"  {model.num_parameters():,} parameters, all starting random\n")

    pad_id = tokenizer.char_to_id[PAD_CHAR]
    val_positions = rng.integers(0, len(val_ids) - args.context, size=20000)
    log_rows, best_val, recent = [], float("inf"), []
    started = time.perf_counter()

    def save(val_loss, steps_done):
        model.save(out_path, tokenizer.chars, {
            "stage": 1, "trained_on": date.today().isoformat(), "steps": steps_done,
            "val_loss": round(val_loss, 4), "baselines": {k: round(v, 4) for k, v in baselines.items()},
            "train_chars": int(len(train_ids)), "val_chars": int(len(val_ids)),
            "train_sha256": hashlib.sha256(train_text.encode()).hexdigest()[:16],
            "hyperparameters": {"batch": args.batch, "lr": args.lr, "seed": args.seed},
        })

    # 5-9. forward pass, loss, backpropagation, optimizer, training
    step = 0
    try:
        for step in range(1, args.steps + 1):
            lr = learning_rate(step - 1, args.steps, args.lr)
            X, Y = get_batch(train_ids, args.context, args.batch, rng, pad_id=pad_id)
            recent.append(train_step(model, optimizer, X, Y, lr))

            if step % args.eval_every == 0 or step == args.steps:
                val_loss = evaluate(model, val_ids, val_positions)
                train_loss = sum(recent) / len(recent)
                recent = []
                elapsed = time.perf_counter() - started
                left = elapsed / step * (args.steps - step)
                marker = ""
                if val_loss < best_val:
                    best_val, marker = val_loss, "  (saved)"
                    save(val_loss, step)
                print(f"step {step:>6}/{args.steps} | train loss {train_loss:.3f} | "
                      f"val loss {val_loss:.3f} | {format_time(elapsed)} elapsed, "
                      f"~{format_time(left)} left{marker}")
                sample = generate(model, tokenizer, "\n\n", max_chars=90, seed=step).text
                print("   sample: " + sample.replace("\n", " / "))
                log_rows.append([step, round(train_loss, 4), round(val_loss, 4), lr, round(elapsed, 1)])
    except KeyboardInterrupt:
        print(f"\nStopped early at step {step}. The best model so far is kept.")

    if not log_rows:
        print("Nothing was saved (stopped before the first check). Run the command again.")
        return 1

    # final exam: the loss on the WHOLE validation set, using the best saved model
    model, _ = CharMLP.load(out_path)
    all_positions = np.arange(len(val_ids) - model.context)
    final_val = evaluate(model, val_ids, all_positions)
    save(final_val, step)
    with open(out_path.with_name(out_path.stem + "_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "learning_rate", "seconds"])
        writer.writerows(log_rows)

    print("\nResult on all validation text (lower is better)")
    for name, value in baselines.items():
        print(f"  {name:16} {value:.3f}")
    print(f"  {'OUR NETWORK':16} {final_val:.3f}   <- perplexity {math.exp(final_val):.1f}: "
          f"as unsure as choosing among ~{math.exp(final_val):.0f} characters")
    print(f"\nSaved {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}")
    print("Try it:   python -m brain.generate \"Once upon a time\"")
    print("Chat:     python -m chat.server")
    return 0


if __name__ == "__main__":
    sys.exit(main())
