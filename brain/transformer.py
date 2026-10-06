"""
The Stage 1.5 brain: a tiny transformer that reads up to 256 characters.

    ids                      the text as numbers                       (B, T)
      | token embedding (+ position embedding, for position="absolute")
    x                        one vector per position                   (B, T, D)
      | Block x L            attention (look back) + MLP (process)     (B, T, D)
      | final LayerNorm
      | output layer
    logits                   at EVERY position: a score for each       (B, T, V)
                             possible next token

Why this is better than the Stage 1 brain:
  - it sees 256 characters instead of 32
  - attention uses the same weights at every position, and any position can look up any
    earlier character, so "rabbit" is recognized wherever it appears in your message
  - it is trained on message -> reply examples, not only on books

Where each token is: two options (chosen when a brain is created, stored in its file).
  position="absolute"  a learned vector for each position 0..255 is added to each token (Stage 1.5,
                       Stage 1.5b). Finding "the token right after the one I just copied" must be
                       learned separately for every position.
  position="rotary"    no position vectors; inside attention, queries and keys are rotated by
                       their position (brain/layers.py), so attention compares RELATIVE positions
                       ("one step back") the same way everywhere. Added after the Stage 1.5b
                       investigation showed copying breaks after the first piece (docs/DECISIONS.md D-044).

The output layer: two options.
  tied=False  a separate output table (Stage 1.5, 1.5b): "write the token I just read" must be learned
              separately for every token, so copying only works for tokens it practised copying.
  tied=True   the output layer IS the token table, used backwards (Press & Wolf, 2017): scores are
              "how similar is my final vector to each token's own vector", so copying a token works
              the same way for every token in the vocabulary (docs/DECISIONS.md D-045).

Everything here, including every backward pass, is our own code (see brain/layers.py).
"""

import json

import numpy as np

from brain.layers import Block, Embedding, LayerNorm, Linear

PROJECT_TAG = "ai-homemade"
MODEL_TYPE = "TinyTransformer"


def masked_cross_entropy(logits, targets, weights):
    """Average loss over the positions whose weight is 1, and the gradient for the logits.

    weights = 1 on positions the model is graded on (e.g. only the reply), 0 elsewhere.
    """
    B, T, V = logits.shape
    flat, t = logits.reshape(-1, V), targets.reshape(-1)
    w = weights.reshape(-1).astype(logits.dtype)
    total = max(float(w.sum()), 1.0)
    shifted = flat - flat.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    rows = np.arange(len(t))
    loss = float(-(log_probs[rows, t] * w).sum() / total)
    dlogits = np.exp(log_probs)
    dlogits[rows, t] -= 1.0
    dlogits *= (w / total)[:, None]
    return loss, dlogits.reshape(B, T, V)


class TinyTransformer:
    fixed_context = False        # unlike Stage 1, shorter inputs need no padding

    def __init__(self, vocab_size, context=256, dim=128, layers=4, heads=4, seed=0, dtype=np.float32,
                 position="absolute", tied=False):
        if position not in ("absolute", "rotary"):
            raise ValueError(f"position must be 'absolute' or 'rotary', not {position!r}")
        self.config = {"vocab_size": int(vocab_size), "context": int(context), "dim": int(dim),
                       "layers": int(layers), "heads": int(heads)}
        if position != "absolute":                    # older brains have no entry: absolute
            self.config["position"] = position
        if tied:                                      # older brains have no entry: separate output table
            self.config["tied"] = True
        rng = np.random.default_rng(seed)
        self.tok = Embedding(vocab_size, dim, rng, dtype=dtype)       # what each token is
        self.pos = Embedding(context, dim, rng, dtype=dtype) if position == "absolute" else None  # where (0..255)
        rotary = position == "rotary"
        self.blocks = [Block(dim, heads, rng, layers, dtype=dtype, rotary=rotary) for _ in range(layers)]
        self.ln_f = LayerNorm(dim, dtype=dtype)
        self.head = None if tied else Linear(dim, vocab_size, rng, bias=False, dtype=dtype)
        self.parts = {"tok": self.tok, **({"pos": self.pos} if self.pos else {}),
                      **{f"block{i}": b for i, b in enumerate(self.blocks)},
                      "ln_f": self.ln_f, **({"head": self.head} if self.head else {})}
        # one flat dictionary of every learnable array (the optimizer updates them in place)
        self.params = {f"{p}.{k}": v for p, part in self.parts.items() for k, v in part.params.items()}

    @property
    def context(self):
        return self.config["context"]

    @property
    def position(self):
        return self.config.get("position", "absolute")

    @property
    def tied(self):
        return self.config.get("tied", False)

    def num_parameters(self):
        return int(sum(p.size for p in self.params.values()))

    def forward(self, ids):
        """ids: integer array (B, T) with T <= context. Returns logits (B, T, V)."""
        T = ids.shape[1]
        if T > self.context:
            raise ValueError(f"sequence of {T} tokens is longer than the context ({self.context})")
        x = self.tok.forward(ids)
        if self.pos is not None:
            x = x + self.pos.forward(np.arange(T))
        for block in self.blocks:
            x = block.forward(x)
        h = self.ln_f.forward(x)
        if self.head is not None:
            return self.head.forward(h)
        self.h = h                                     # tied: score = similarity to each token's vector
        return h @ self.tok.params["E"].T

    def backward(self, dlogits):
        """Chain rule back through every layer. Returns gradients with the same names as params."""
        if self.head is not None:
            dx = self.ln_f.backward(self.head.backward(dlogits))
        else:
            E = self.tok.params["E"]
            D, V = E.shape[1], E.shape[0]
            d_table_out = dlogits.reshape(-1, V).T @ self.h.reshape(-1, D)     # the table's use as output
            dx = self.ln_f.backward(dlogits @ E)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.tok.backward(dx)
        if self.head is None:
            self.tok.grads["E"] = self.tok.grads["E"] + d_table_out            # ... plus its use as input
        if self.pos is not None:
            self.pos.backward(dx.sum(axis=0))      # every sequence shares the same positions
        return {f"{p}.{k}": g for p, part in self.parts.items() for k, g in part.grads.items()}

    def next_probabilities(self, context_ids):
        """Probabilities for the next token after a list of ids (only the last `context` are used)."""
        ids = np.asarray(context_ids[-self.context:], dtype=np.int64).reshape(1, -1)
        logits = self.forward(ids)[0, -1].astype(np.float64)
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def attention_map(self, layer=-1):
        """After a forward pass: the attention weights of one layer, averaged over heads (T, T).

        Row i says how much position i looked at each earlier position when predicting.
        """
        weights = self.blocks[layer].attn.last_weights
        return None if weights is None else weights[0].mean(axis=0)

    # --------------------------------------------------------------- saving / loading
    def save(self, path, vocab, specials, info, created_by="brain/train_chat.py"):
        record = {"project": PROJECT_TAG, "created_by": created_by, "model": MODEL_TYPE,
                  "config": self.config, "vocab": vocab, "specials": list(specials), **info}
        arrays = {f"param_{name}": value for name, value in self.params.items()}
        np.savez(path, meta_json=np.array(json.dumps(record)), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if "meta_json" not in data.files:
                raise ValueError(f"{path} was not created by this project's training code")
            record = json.loads(str(data["meta_json"]))
            if record.get("project") != PROJECT_TAG or record.get("model") != MODEL_TYPE:
                raise ValueError(f"{path} is not a {MODEL_TYPE} model from this project")
            model = cls(**record["config"])
            for name, array in model.params.items():
                array[...] = data[f"param_{name}"]      # copy in place: layers share these arrays
        return model, record
