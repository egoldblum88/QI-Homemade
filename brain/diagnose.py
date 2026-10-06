"""
Diagnostics: small, unambiguous tests of the INPUT -> OUTPUT path of our network.

    python -m brain.diagnose overfit                      # can a fresh network learn 10 tiny examples?
    python -m brain.diagnose arithmetic                   # can it GENERALIZE addition? (about 5 minutes)
    python -m brain.diagnose probe "10 + 10"              # what does the trained brain do with a message?
    python -m brain.diagnose probe "what is your name" --after "10 + 10" "21"   # with earlier turns
    python -m brain.diagnose copy                         # which network can COPY new names? (~15 min)

ARITHMETIC: a fresh network is trained on arithmetic ALONE (the Stage 1.5 problems, 0-50) and
tested on problems it never saw. This separates "can this architecture learn arithmetic at all?"
from "did the full training give arithmetic enough practice?". Measured on 2026-09-30: 43% of
never-seen problems exact after 3,000 steps, 85% after 6,000, 89% after 7,500.

OVERFIT: a fresh network is trained, through exactly the same code as the real training
(encode_example -> pad_batch -> step with AdamW and the loss mask), on the 10 diagnostic
examples below until it should know them by heart. If it cannot reproduce them exactly,
something in the architecture or training is broken. A CONTROL run gives the network the
same message for every example: then it cannot tell them apart, so it must fail. That shows
the test really measures whether the network reads the message.

COPY: the architecture check for copying. The same-sized network is trained on copying ALONE,
once per variant (absolute or rotary positions; separate or tied output layer; copy practice on
28 syllables or on many different words), with the same seed and number of steps. It is then
tested on made-up names it never saw, on REAL names it never saw, on phrases longer than any it
trained on, and on a name given in an EARLIER turn. Written after the Stage 1.5b investigation
(docs/DECISIONS.md D-044, D-045) showed the trained brain copies the first piece of a name and
then makes the rest up. Measured on 2026-09-30 (1,500 steps each): see HOW_TO_RUN.md Part 10.

PROBE: shows the exact tokens fed to a trained brain, its probability for each possible first
reply token, the greedy reply, and what random sampling produces. It separates MODEL quality
(the probabilities) from DECODING quality (what a sampling method turns them into).

The diagnostic examples below exist only for these tests. They are never used to train the
real brain, and they are not its personality.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from brain.dialogue import END, SPECIALS, Example, encode_example, encode_prompt
from brain.generate import load_brain, reply_room, reply_to
from brain.tokenizer import CharTokenizer
from brain.train import AdamW
from brain.train_chat import matrices_only, pad_batch, step
from brain.transformer import TinyTransformer, masked_cross_entropy

DIAGNOSTIC_EXAMPLES = [
    # unambiguous input -> output pairs (testing only; never used for the real brain)
    ("hello", "hello"),
    ("what is 1 + 1", "2"),
    ("what is 2 + 2", "4"),
    ("what is 5 + 5", "10"),
    ("what is 10 + 10", "20"),
    ("hi", "Hello."),
    ("how are you", "I am doing well."),
    ("what is your name", "I have not been given a name."),     # neutral: no identity is invented
    ("good morning", "Good morning."),
    ("thank you", "You're welcome."),
]


def answer_probability(model, tokenizer, context, answer):
    """The network's probability of producing exactly `answer` and then <end> (no sampling involved)."""
    ids, weights = encode_example(tokenizer, Example(list(context), answer, "probe"), model.context)
    logits = model.forward(np.array([ids[:-1]]))[0].astype(np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    return float(np.exp(sum(log_probs[j, ids[j + 1]] for j in range(len(ids) - 1) if weights[j + 1])))


def overfit(pairs, steps=400, dim=64, layers=2, heads=2, lr=3e-3, seed=0, same_message=False, log=print,
            position="absolute", tied=False):
    """Train a fresh network on `pairs` and report how exactly it reproduces them."""
    chars = "".join(m + r for m, r in pairs)
    tokenizer = CharTokenizer(chars, SPECIALS)
    model = TinyTransformer(tokenizer.vocab_size, context=64, dim=dim, layers=layers, heads=heads, seed=seed,
                            position=position, tied=tied)
    shown = [("?" if same_message else m, r) for m, r in pairs]        # the control hides the messages
    encoded = [encode_example(tokenizer, Example([m], r, "diagnostic"), model.context) for m, r in shown]
    X, Y, W = pad_batch(encoded, tokenizer.special(END))
    optimizer = AdamW(model.params, weight_decay=0.0, decay=matrices_only)
    for i in range(1, steps + 1):
        loss = step(model, optimizer, X, Y, W, lr)
        if log and (i % 100 == 0 or i == 1):
            log(f"    step {i:>4}: loss on the replies {loss:.4f}")
    results = []
    for (message, reply), (shown_message, _) in zip(pairs, shown):
        greedy, _ = reply_to(model, tokenizer, [shown_message], temperature=0, loop_penalty=1.0)
        results.append({"message": message, "expected": reply, "greedy": greedy.text,
                        "exact": greedy.text == reply,
                        "probability": answer_probability(model, tokenizer, [shown_message], reply)})
    exact = sum(r["exact"] for r in results) / len(results)
    return {"final_loss": loss, "exact": exact, "results": results,
            "distinct_outputs": len({r["greedy"] for r in results})}


def probe(model, tokenizer, context, samples=20, expected=None):
    """Everything about how a trained brain answers one message (with optional earlier turns)."""
    ids = encode_prompt(tokenizer, context, model.context - reply_room(model))
    probs = model.next_probabilities(ids)
    top = np.argsort(probs)[::-1][:8]
    greedy, _ = reply_to(model, tokenizer, context, temperature=0, loop_penalty=1.0)
    sampled = []
    for s in range(samples):
        r, _ = reply_to(model, tokenizer, context, temperature=0.8, top_k=20, seed=s)
        sampled.append(r.text)
    report = {
        "tokens_text": tokenizer.decode(ids, show_specials=True),
        "tokens": ids,
        "first_token": [(tokenizer.decode([i], show_specials=True), float(probs[i])) for i in top],
        "p_digit_first": float(sum(probs[tokenizer.char_to_id[d]] for d in "0123456789"
                                   if d in tokenizer.char_to_id)),
        "greedy": greedy.text, "greedy_stopped_by": greedy.stopped_by,
        "sampled": Counter(sampled).most_common(),
    }
    if expected is not None:
        report["p_expected"] = answer_probability(model, tokenizer, context, expected)
        report["sampled_correct"] = sum(s == expected for s in sampled) / samples
    return report


def arithmetic_generalization(steps, log=print):
    """Train a fresh network on arithmetic only; report exact answers on never-seen problems."""
    import random
    import time
    from brain.dialogue import arithmetic_examples, arithmetic_problems, split_problems
    from brain.train import learning_rate
    train_p, held_p = split_problems(arithmetic_problems())
    train_ex = arithmetic_examples(train_p, phrasings_per_problem=2)
    held_ex = arithmetic_examples(held_p, phrasings_per_problem=1, seed=1)
    tokenizer = CharTokenizer("".join(e.context[0] + e.reply for e in train_ex + held_ex), SPECIALS)
    model = TinyTransformer(tokenizer.vocab_size, context=48, dim=128, layers=4, heads=4, seed=0)
    optimizer = AdamW(model.params, weight_decay=0.1, decay=matrices_only)
    encoded = [encode_example(tokenizer, e, model.context) for e in train_ex]
    rng, started = random.Random(0), time.perf_counter()
    log(f"Training a fresh network on {len(train_ex):,} arithmetic examples only; testing on "
        f"{len(held_ex)} problems it never sees.")
    for i in range(1, steps + 1):
        step(model, optimizer, *pad_batch(rng.sample(encoded, 64), tokenizer.special(END)),
             learning_rate(i - 1, steps, 3e-3))
        if i % max(1, steps // 6) == 0 or i == steps:
            right = sum(reply_to(model, tokenizer, e.context, temperature=0, loop_penalty=1.0)[0].text == e.reply
                        for e in held_ex)
            log(f"  step {i:>5}: {100 * right / len(held_ex):5.1f}% of never-seen problems exact "
                f"({time.perf_counter() - started:.0f} s)")
    return 0


# ------------------------------------------------------------------------ copying
COPY_TRAIN_TEMPLATES = [("repeat: {p}", "{p}"), ("Say: {p}", "{p}"),
                        ("My name is {n}. What is my name?", "Your name is {n}."),
                        ("Call me {n}. What is my name?", "Your name is {n}.")]


def copy_sets(book_text, seed=0):
    """Two kinds of copy practice, and four never-seen test sets.

    "syllables": made-up names and words from 28 syllables (as in the Stage 1.5b data)
    "diverse":   the same, plus capitalized and lowercase words from the whole book vocabulary, so
                 copying is practised on many more different tokens
    The tests use made-up names never trained, and REAL names from the books that are never used
    for training (the real brain's validation names).
    """
    import random
    import re
    from brain.tasks import made_up_words, split_names, word_lists
    rng = random.Random(seed)
    names = made_up_words(2000, seed=30, capitalize=True)
    words = made_up_words(2000, seed=31)
    _, book_names = split_names(word_lists(book_text)[1])
    reserved = {n.lower() for n in book_names}
    vocabulary = sorted(set(re.findall(r"\b[a-z]{3,10}\b", book_text)) - reserved)
    practice = {"syllables": (names, words),
                "diverse": (names + [w.capitalize() for w in vocabulary], words + vocabulary)}
    seen = set(names) | {w.capitalize() for w in words} | {w.capitalize() for w in vocabulary}
    new_names = [w for w in made_up_words(2400, seed=77, capitalize=True) if w not in seen][:200]
    new_words = [w.lower() for w in new_names]
    tests = {
        "new made-up name": [Example([f"My name is {n}. What is my name?"], f"Your name is {n}.", "copy")
                             for n in new_names[:100]],
        "real name never trained": [Example([f"My name is {n}. What is my name?"], f"Your name is {n}.", "copy")
                                    for n in book_names],
        "4-5 new words (trained: 1-3)": [Example([f"repeat: {p}"], p, "copy") for p in
                                         (" ".join(rng.sample(new_words, rng.randint(4, 5))) for _ in range(100))],
        "name in an earlier turn": [Example([f"My name is {n}.", "Yes.", "What is my name?"], f"Your name is {n}.",
                                            "copy") for n in new_names[100:200]],
    }
    return practice, tests


def copy_training_example(rng, names, words):
    message, reply = rng.choice(COPY_TRAIN_TEMPLATES)
    p = " ".join(rng.choice(words) for _ in range(rng.randint(1, 3)))
    n = rng.choice(names)
    example = Example([message.format(p=p, n=n)], reply.format(p=p, n=n), "copy")
    if "{n}" in message and rng.random() < 0.3:         # sometimes the name comes in an earlier turn
        example = Example([f"My name is {n}.", rng.choice(["Yes.", "No.", "Hello.", "Good."]), "What is my name?"],
                          example.reply, "copy")
    return example


COPY_VARIANTS = [("absolute", False, "syllables"), ("rotary", False, "syllables"), ("rotary", True, "syllables"),
                 ("absolute", True, "diverse"), ("rotary", False, "diverse"), ("rotary", True, "diverse")]


def variant_name(position, tied, practice):
    return f"{position}{', tied' if tied else ''}, {practice}"


def copying(steps=1500, variants=COPY_VARIANTS, seed=0, log=print):
    """Train the real-sized network on copying alone, once per variant. Returns the results."""
    import random
    import time
    from brain.generate import STAGE15B_MODEL
    from brain.train import learning_rate
    book_text = (Path(__file__).resolve().parent.parent / "data" / "processed" / "train.txt").read_text(encoding="utf-8")
    if STAGE15B_MODEL.exists():             # the real brain's own subword tokens, so the pieces are the same
        _, tokenizer, _ = load_brain(STAGE15B_MODEL)
    else:
        tokenizer = CharTokenizer(book_text, SPECIALS)
    practice, tests = copy_sets(book_text, seed)
    results = {}
    for position, tied, kind in variants:
        name = variant_name(position, tied, kind)
        names, words = practice[kind]
        rng, started = random.Random(seed), time.perf_counter()
        model = TinyTransformer(tokenizer.vocab_size, context=96, dim=128, layers=4, heads=4, seed=seed,
                                position=position, tied=tied)
        optimizer = AdamW(model.params, weight_decay=0.1, decay=matrices_only)
        log(f"\n{name}: training {steps} steps on copying only ...")
        for i in range(1, steps + 1):
            batch = [encode_example(tokenizer, copy_training_example(rng, names, words), model.context)
                     for _ in range(64)]
            loss = step(model, optimizer, *pad_batch(batch, tokenizer.special(END)), learning_rate(i - 1, steps, 3e-3))
            if i % max(1, steps // 5) == 0:
                log(f"  step {i:>5}: loss on the replies {loss:.3f} ({time.perf_counter() - started:.0f} s)")
        results[name] = {}
        for test, examples in tests.items():
            right = sum(reply_to(model, tokenizer, ex.context, temperature=0, loop_penalty=1.0)[0].text == ex.reply
                        for ex in examples)
            results[name][test] = right / len(examples)
        log("  never-seen tests, exact: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in results[name].items()))
    return results, {k: len(v) for k, v in tests.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose the input -> output path of our network.")
    sub = parser.add_subparsers(dest="command", required=True)
    o = sub.add_parser("overfit", help="train a fresh network on 10 tiny examples")
    o.add_argument("--steps", type=int, default=400)
    o.add_argument("--position", choices=["absolute", "rotary"], default="absolute")
    o.add_argument("--tied", action="store_true", help="tied output layer (the round 2 design)")
    c = sub.add_parser("copy", help="architecture variants, trained on copying alone")
    c.add_argument("--steps", type=int, default=1500)
    c.add_argument("--only", nargs="*", type=int, default=None,
                   help="run only these variants (numbers 1-6 in the order listed in the output)")
    a = sub.add_parser("arithmetic", help="train a fresh network on arithmetic alone; test on unseen problems")
    a.add_argument("--steps", type=int, default=6000)
    p = sub.add_parser("probe", help="inspect how a trained brain answers a message")
    p.add_argument("message")
    p.add_argument("--after", nargs="*", default=[], help="earlier turns: user, ai, user, ai, ...")
    p.add_argument("--expected", help="the answer you consider correct (e.g. 20)")
    p.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    if args.command == "overfit":
        print("Can a fresh network learn these 10 input -> output pairs exactly?\n")
        run = overfit(DIAGNOSTIC_EXAMPLES, steps=args.steps, position=args.position, tied=args.tied)
        for r in run["results"]:
            mark = "OK  " if r["exact"] else "FAIL"
            print(f"  {mark} {r['message']!r:22} -> {r['greedy']!r:34} (P(exact)={r['probability']:.3f})")
        print(f"\n  exact: {100 * run['exact']:.0f}%, {run['distinct_outputs']} different replies for 10 messages")
        print("\nCONTROL: the same training, but every example gets the same message '?'")
        control = overfit(DIAGNOSTIC_EXAMPLES, steps=args.steps, same_message=True, log=None, position=args.position,
                          tied=args.tied)
        print(f"  exact: {100 * control['exact']:.0f}% (it cannot tell the examples apart, as expected)")
        passed = run["exact"] == 1.0 and control["exact"] <= 0.2
        print(f"\nRESULT: {'PASS' if passed else 'FAIL'}: the network "
              f"{'can' if passed else 'can NOT'} learn an input -> output relationship.")
        return 0 if passed else 1

    if args.command == "arithmetic":
        return arithmetic_generalization(args.steps)

    if args.command == "copy":
        print("Architecture check: which network COPIES names and words it never saw?")
        print("(same size, seed and steps; only the listed choices differ)")
        for k, v in enumerate(COPY_VARIANTS, 1):
            print(f"  {k}. {variant_name(*v)}")
        chosen = [COPY_VARIANTS[k - 1] for k in args.only] if args.only else COPY_VARIANTS
        results, sizes = copying(args.steps, chosen)
        tests = list(next(iter(results.values())))
        print(f"\n  {'variant':34} " + " ".join(f"{t[:18]:>18}" for t in tests))
        for name, r in results.items():
            print(f"  {name:34} " + " ".join(f"{100 * r[t]:17.0f}%" for t in tests))
        print("  (test sizes: " + ", ".join(f"{t} {n}" for t, n in sizes.items()) + ")")
        return 0

    model, tokenizer, _ = load_brain(args.model)
    context = list(args.after) + [args.message]
    r = probe(model, tokenizer, context, expected=args.expected)
    print(f"Fed to the network ({len(r['tokens'])} tokens): {r['tokens_text']!r}")
    print(f"  as numbers: {r['tokens']}")
    print("Next-token probabilities for the first reply character (MODEL quality):")
    print("  " + ", ".join(f"{t!r} {p:.3f}" for t, p in r["first_token"]))
    if args.expected:
        print(f"  probability of exactly {args.expected!r} then <end>: {r['p_expected']:.3f}")
    print(f"Greedy decoding (always the most likely character): {r['greedy']!r} [{r['greedy_stopped_by']}]")
    print("Sampling, temperature 0.8, top-k 20, 20 tries (DECODING quality):")
    for text, n in r["sampled"][:8]:
        print(f"  {n:>2}x  {text[:90]!r}")
    if args.expected:
        print(f"  sampled replies equal to {args.expected!r}: {100 * r['sampled_correct']:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
