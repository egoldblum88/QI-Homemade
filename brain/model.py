"""
The Stage 1 brain: a small neural network that predicts the next character.

HOW INFORMATION FLOWS (the forward pass), for a batch of B examples:

    context      the last N characters, as numbers             X        shape (B, N)
       |  look up each character's learned vector (its "embedding")
    embeddings   N vectors of E numbers each                   emb      shape (B, N, E)
       |  lay the N vectors side by side into one long vector
    flat                                                       x        shape (B, N*E)
       |  hidden layer: each hidden unit takes a weighted sum of ALL inputs, then tanh
    hidden       h = tanh(x @ W1 + b1)                         h        shape (B, H)
       |  output layer: one score ("logit") for every possible next character
    logits       z = h @ W2 + b2                               logits   shape (B, V)
       |  softmax turns the scores into probabilities that add up to 1
    probs        p = softmax(z)                                probs    shape (B, V)

THE PARAMETERS (the numbers that learning changes; they start random):

    C    (V, E)     embedding table: one row of E numbers per character
    W1   (N*E, H)   hidden-layer weights       b1   (H,)   hidden-layer biases
    W2   (H, V)     output-layer weights       b2   (V,)   output-layer biases

HOW IT LEARNS:
    loss = -log(probability the model gave to the real next character), averaged.
    backward() applies the chain rule of calculus from the loss back through every layer,
    producing a gradient for every parameter: "moving this number up or down by a tiny
    amount changes the loss by this much". The optimizer in brain/train.py then nudges
    every parameter a little in the direction that lowers the loss.
    tests/test_brain.py checks every gradient against a slow numerical estimate.

This is the classic neural language model of Bengio et al. (2003), implemented by us.
"""

import json

import numpy as np

PROJECT_TAG = "ai-homemade"
MODEL_TYPE = "CharMLP"


def softmax(logits):
    """Scores -> probabilities (each row sums to 1). Subtracting the max avoids overflow."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def cross_entropy(logits, targets):
    """Average loss over the batch, and its gradient with respect to the logits.

    loss      = mean over the batch of  -log p[true next character]
    gradient  = (p - one_hot(true character)) / B
    """
    batch = logits.shape[0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(batch), targets].mean()

    dlogits = np.exp(log_probs)                       # the probabilities p
    dlogits[np.arange(batch), targets] -= 1.0         # p - one_hot
    dlogits /= batch
    return float(loss), dlogits


class CharMLP:
    """Embeddings -> one tanh hidden layer -> next-character scores."""

    fixed_context = True      # always needs exactly `context` inputs (shorter ones get padded)

    def __init__(self, vocab_size, context=32, embed=16, hidden=512, seed=0, dtype=np.float32):
        self.config = {"vocab_size": int(vocab_size), "context": int(context),
                       "embed": int(embed), "hidden": int(hidden)}
        rng = np.random.default_rng(seed)
        fan_in = context * embed
        # Random starting values, scaled so signals neither explode nor vanish at the start.
        # W2 starts tiny, so at first every character looks almost equally likely.
        self.params = {
            "C": rng.standard_normal((vocab_size, embed)),
            "W1": rng.standard_normal((fan_in, hidden)) / np.sqrt(fan_in),
            "b1": np.zeros(hidden),
            "W2": rng.standard_normal((hidden, vocab_size)) * 0.01,
            "b2": np.zeros(vocab_size),
        }
        self.params = {name: value.astype(dtype) for name, value in self.params.items()}

    @property
    def context(self):
        return self.config["context"]

    def num_parameters(self):
        return int(sum(p.size for p in self.params.values()))

    # ------------------------------------------------------------------ forward pass
    def forward(self, X):
        """X: integer array (B, N) of character ids. Returns (logits (B, V), cache)."""
        P = self.params
        emb = P["C"][X]                               # (B, N, E)  look up each character
        x = emb.reshape(X.shape[0], -1)               # (B, N*E)   side by side
        h = np.tanh(x @ P["W1"] + P["b1"])            # (B, H)     hidden layer
        logits = h @ P["W2"] + P["b2"]                # (B, V)     one score per character
        cache = (X, x, h)                             # remembered for the backward pass
        return logits, cache

    # ----------------------------------------------------------------- backward pass
    def backward(self, dlogits, cache):
        """Chain rule, from the output back to every parameter. Returns a dict of gradients."""
        X, x, h = cache
        P = self.params
        grads = {}

        # output layer: logits = h @ W2 + b2
        grads["W2"] = h.T @ dlogits                   # (H, V)
        grads["b2"] = dlogits.sum(axis=0)             # (V,)
        dh = dlogits @ P["W2"].T                      # (B, H)  how the loss depends on h

        # tanh: h = tanh(pre), and the derivative of tanh is 1 - tanh^2
        dpre = dh * (1.0 - h * h)                     # (B, H)

        # hidden layer: pre = x @ W1 + b1
        grads["W1"] = x.T @ dpre                      # (N*E, H)
        grads["b1"] = dpre.sum(axis=0)                # (H,)
        dx = dpre @ P["W1"].T                         # (B, N*E)

        # embedding lookup: each use of a character sends its gradient back to that row of C.
        # A character used many times in the batch collects all of its gradients (np.add.at).
        demb = dx.reshape(X.shape[0] * X.shape[1], -1)
        dC = np.zeros_like(P["C"])
        np.add.at(dC, X.reshape(-1), demb)
        grads["C"] = dC
        return grads

    def next_char_probabilities(self, context_ids):
        """Probabilities for the next character after one context (a list of N ids)."""
        logits, _ = self.forward(np.asarray(context_ids, dtype=np.int64).reshape(1, -1))
        return softmax(logits[0].astype(np.float64))

    next_probabilities = next_char_probabilities    # the common name used by brain/generate.py

    # --------------------------------------------------------------- saving / loading
    def save(self, path, vocab, info):
        """Save weights + vocabulary + a provenance record into one .npz file.

        The provenance record ("created_by") is checked by the Originality Guard: model
        files that our own training did not create are rejected.
        """
        record = {"project": PROJECT_TAG, "created_by": "brain/train.py", "model": MODEL_TYPE,
                  "config": self.config, "vocab": vocab, **info}
        arrays = {f"param_{name}": value for name, value in self.params.items()}
        np.savez(path, meta_json=np.array(json.dumps(record)), **arrays)

    @classmethod
    def load(cls, path):
        """Load a model saved by save(). Returns (model, record)."""
        with np.load(path, allow_pickle=False) as data:
            if "meta_json" not in data.files:
                raise ValueError(f"{path} was not created by this project's training code")
            record = json.loads(str(data["meta_json"]))
            if record.get("project") != PROJECT_TAG or record.get("model") != MODEL_TYPE:
                raise ValueError(f"{path} is not a {MODEL_TYPE} model from this project")
            model = cls(**record["config"])
            for name in model.params:
                model.params[name] = data[f"param_{name}"]
        return model, record
