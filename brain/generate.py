"""
Text generation: a trained brain writes, one token (character) at a time.

    python -m brain.generate "hello"                 # Stage 1.5: its reply to a message
    python -m brain.generate "Once upon a time" --continue   # just continue the text
    python -m brain.generate "Once upon a time" --model models/stage1.npz   # the Stage 1 brain

How one character is produced:
  1. take the text so far as numbers (Stage 1 sees the last 32; Stage 1.5 the last 256)
  2. the model gives a probability for every possible next token
  3. adjust those probabilities with the controls below, then pick one at random
  4. append it and repeat, until a stop token or the length limit

The controls, and why they exist:
  temperature   divides the scores before softmax. Below 1 = safer, more repetitive;
                above 1 = more adventurous, more mistakes. 0 = always take the top choice.
  top_k         only the k most likely tokens may be picked: cuts off rare junk.
  loop_penalty  small models fall into loops ("and the and the"). If the last few tokens
                already appeared earlier, the tokens that followed them before get their
                probability multiplied by this number.
  max_chars     hard length limit, so generation always ends.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from brain.bpe import BPETokenizer
from brain.model import CharMLP
from brain.tokenizer import CharTokenizer
from brain.transformer import TinyTransformer

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
STAGE1_MODEL = MODELS / "stage1.npz"
STAGE15_MODEL = MODELS / "stage1_5.npz"
STAGE15B_MODEL = MODELS / "stage1_5b.npz"
STAGE15B2_MODEL = MODELS / "stage1_5b2.npz"       # Stage 1.5b, round 2
STAGE15B3_MODEL = MODELS / "stage1_5b3.npz"       # Stage 1.5b, round 3: using what you told it
STAGE15B4_MODEL = MODELS / "stage1_5b4.npz"       # Stage 1.5b, round 4: saying a reasoning result
STAGE15B4B_MODEL = MODELS / "stage1_5b4b.npz"     # Stage 1.5b, round 4b: the same, with facts about you
STAGE15B4C_MODEL = MODELS / "stage1_5b4c.npz"     # Stage 1.5b, round 4c (D-076, D-077): negative answers too
STAGE15B4D_MODEL = MODELS / "stage1_5b4d.npz"     # Stage 1.5b, round 4d (D-080): says the self-model's answer too
DEFAULT_MODEL = STAGE1_MODEL          # kept for Stage 1 code; see newest_model()
PAD_CHAR = "\n"
LOOP_WINDOW = 8                       # how many recent tokens are checked for repeats
MODEL_CLASSES = {"CharMLP": CharMLP, "TinyTransformer": TinyTransformer}


class ModelNotFoundError(FileNotFoundError):
    pass


def newest_model():
    """The newest brain available: Stage 1.5b round 4d, else 4c, else 4b, else 4, else 3, else 2, else 1.5b, else 1.5,
    else 1."""
    for path in (STAGE15B4D_MODEL, STAGE15B4C_MODEL, STAGE15B4B_MODEL, STAGE15B4_MODEL, STAGE15B3_MODEL, STAGE15B2_MODEL, STAGE15B_MODEL,
                 STAGE15_MODEL):
        if path.exists():
            return path
    return STAGE1_MODEL


def load_brain(path=None):
    """Load (model, tokenizer, record). Fails with clear instructions if not trained yet."""
    path = Path(path) if path else newest_model()
    if not path.exists():
        command = {"stage1_5b2.npz": "python -m brain.train_mix",
                   "stage1_5b.npz": "python -m brain.train_mix --position absolute --untied --tokenizer-from new "
                                    "--out models/stage1_5b.npz   (details: HOW_TO_RUN.md Part 9)",
                   "stage1_5.npz": "python -m brain.train_chat"}.get(path.name, "python -m brain.train")
        raise ModelNotFoundError(f"No trained model found at {path}.\n"
                                 f"Train it first:\n    {command}")
    with np.load(path, allow_pickle=False) as data:
        kind = json.loads(str(data["meta_json"])).get("model") if "meta_json" in data.files else None
    if kind not in MODEL_CLASSES:
        raise ValueError(f"{path} was not created by this project's training code")
    model, record = MODEL_CLASSES[kind].load(path)
    if record.get("merges"):                       # Stage 1.5b: our subword (BPE) tokenizer
        tokenizer = BPETokenizer(record["vocab"], record["merges"], record.get("specials", []))
    else:
        tokenizer = CharTokenizer(record["vocab"], record.get("specials", []))
    return model, tokenizer, record


@dataclass
class Generation:
    text: str                       # the generated characters (special tokens left out)
    context_ids: list               # what the model saw before its first token
    stopped_by: str                 # "stop marker" or "length limit"
    steps: list = field(default_factory=list)          # details of the first few choices
    chosen_probabilities: list = field(default_factory=list)
    elapsed_ms: float = 0.0
    ids: list = field(default_factory=list)            # the generated token ids
    stop_token: str = None                             # which token ended it (None = length limit)
    turns_used: int = None                             # Stage 1.5: how many turns fit in the prompt
    turns_given: int = None                            # ... out of how many were available
    message_cut: bool = False                          # ... and whether the message had to be shortened


def repeat_followers(seq, window=LOOP_WINDOW):
    """Items that followed earlier copies of the last `window` items (works on text or id lists)."""
    if len(seq) <= window:
        return set()
    suffix = list(seq[-window:])
    items = list(seq)
    return {items[i + window] for i in range(len(items) - window)
            if items[i:i + window] == suffix}


def choose_next(probs, temperature, top_k, rng):
    """Apply temperature and top-k to probabilities, then pick one index."""
    if temperature <= 0:
        return int(np.argmax(probs)), probs
    logp = np.log(np.maximum(probs, 1e-12)) / temperature
    adjusted = np.exp(logp - logp.max())
    if top_k and top_k < len(adjusted):
        cutoff = np.sort(adjusted)[-top_k]
        adjusted[adjusted < cutoff] = 0.0
    adjusted /= adjusted.sum()
    return int(rng.choice(len(adjusted), p=adjusted)), adjusted


def generate(model, tokenizer, prompt="", max_chars=200, temperature=0.8, top_k=20,
             loop_penalty=0.3, stop=None, seed=None, record_steps=0, prompt_ids=None, stop_ids=None):
    """Continue a prompt (text, or ready-made ids). Generation ends at a stop character/token."""
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    stop_ids = set(stop_ids or ()) | {tokenizer.char_to_id[c] for c in (stop or ()) if c in tokenizer.char_to_id}
    history = list(prompt_ids) if prompt_ids is not None else tokenizer.encode(prompt)
    # Loop detection looks ONLY at what is being written now, never at the prompt: otherwise
    # repeating a question would push the model away from its own earlier (correct) answer.
    written = []
    if getattr(model, "fixed_context", False):                # Stage 1 needs exactly N inputs
        pad = tokenizer.char_to_id.get(PAD_CHAR, 0)
        history = [pad] * max(0, model.context - len(history)) + history
    context_ids = history[-model.context:]
    generated, steps, chosen_probs = [], [], []
    stopped_by, stop_token = "length limit", None

    for step in range(max_chars):
        model_probs = model.next_probabilities(history[-model.context:])
        probs = model_probs.copy()
        if loop_penalty < 1.0:
            for token in repeat_followers(written):
                probs[token] *= loop_penalty
            probs /= probs.sum()
        next_id, _ = choose_next(probs, temperature, top_k, rng)

        if step < record_steps:     # what the MODEL believed, before our sampling controls
            top = np.argsort(model_probs)[::-1][:3]
            steps.append({"top": [[tokenizer.decode([i], show_specials=True), round(float(model_probs[i]), 3)]
                                  for i in top],
                          "chose": tokenizer.decode([next_id], show_specials=True)})
        chosen_probs.append(float(model_probs[next_id]))
        if next_id in stop_ids:
            stopped_by = "stop marker"
            stop_token = tokenizer.decode([next_id], show_specials=True)
            break
        generated.append(next_id)
        history.append(next_id)
        written.append(next_id)

    return Generation(text=tokenizer.decode(generated), context_ids=context_ids, stopped_by=stopped_by,
                      steps=steps, chosen_probabilities=chosen_probs, ids=generated,
                      elapsed_ms=(time.perf_counter() - started) * 1000, stop_token=stop_token)


def reply_room(model):
    """How many tokens a reply may use: half the context (128 of 256), the rest is for the prompt."""
    from brain.dialogue import MAX_REPLY_CHARS
    return min(MAX_REPLY_CHARS, model.context // 2)


def reply_to(model, tokenizer, context, **options):
    """Stage 1.5: the model's reply to a conversation (a list of turns, the last is the message).

    The reply ends at <end>. It also ends at <user> or <ai> (a new turn would be starting: the
    reply must never cross into it) and at a new line (replies in training never contain one,
    so a line break means the model has drifted back into book-writing).
    """
    from brain.dialogue import encode_prompt
    room = reply_room(model)
    ids, turns_used, message_cut = encode_prompt(tokenizer, context, model.context - room, report=True)
    options.setdefault("max_chars", room)
    stop = {tokenizer.special(name) for name in tokenizer.specials}
    stop |= {tokenizer.char_to_id[c] for c in "\n" if c in tokenizer.char_to_id}
    result = generate(model, tokenizer, prompt_ids=ids, stop_ids=stop, **options)
    result.turns_used, result.turns_given, result.message_cut = turns_used, len(context), message_cut
    return result, ids


def main(argv=None):
    parser = argparse.ArgumentParser(description="Let a trained brain write.")
    parser.add_argument("prompt", nargs="?", default="hello", help="a message (or text to continue)")
    parser.add_argument("--continue", dest="just_continue", action="store_true",
                        help="continue the text instead of replying to it")
    parser.add_argument("--chars", type=int, default=None, help="maximum characters to write")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None, help="same seed = same text")
    parser.add_argument("--model", default=None, help="model file (default: the newest brain)")
    args = parser.parse_args(argv)

    try:
        model, tokenizer, record = load_brain(args.model)
    except ModelNotFoundError as e:
        print(e)
        return 1
    options = {"temperature": args.temperature, "top_k": args.top_k, "seed": args.seed}
    if tokenizer.specials and not args.just_continue:
        if args.chars:
            options["max_chars"] = args.chars
        result, _ = reply_to(model, tokenizer, [args.prompt], **options)
        print(f"USER: {args.prompt}\nAI:   {result.text}")
    else:
        result = generate(model, tokenizer, args.prompt, max_chars=args.chars or 300, **options)
        print(args.prompt + result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
