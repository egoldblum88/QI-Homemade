"""
The building blocks of the Stage 1.5 brain, written by hand in NumPy.

Every layer has:
    forward(x)     compute the output and remember what backward() will need
    backward(dy)   given "how the loss changes with my output" (dy), compute
                   (a) the gradient for each of my parameters  -> self.grads
                   (b) "how the loss changes with my input"     -> returned
    params, grads  dictionaries of NumPy arrays (same names)

tests/test_layers.py checks every backward() against slow numerical estimates.

Shapes: B = batch size, T = sequence length, D = width, H = heads, V = vocabulary.
"""

import numpy as np


def init_normal(rng, shape, std, dtype):
    return (rng.standard_normal(shape) * std).astype(dtype)


# ---------------------------------------------------------------------------- Linear
class Linear:
    """y = x @ W + b. Every output is a weighted sum of every input."""

    def __init__(self, n_in, n_out, rng, std=0.02, bias=True, dtype=np.float32):
        self.params = {"W": init_normal(rng, (n_in, n_out), std, dtype)}
        if bias:
            self.params["b"] = np.zeros(n_out, dtype=dtype)
        self.grads = {}

    def forward(self, x):
        self.x = x
        y = x @ self.params["W"]
        return y + self.params["b"] if "b" in self.params else y

    def backward(self, dy):
        x2 = self.x.reshape(-1, self.x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        self.grads["W"] = x2.T @ dy2                       # each weight: input x output, summed
        if "b" in self.params:
            self.grads["b"] = dy2.sum(axis=0)
        return dy @ self.params["W"].T


# ------------------------------------------------------------------------- Embedding
class Embedding:
    """Token id -> learned vector (a row of the table)."""

    def __init__(self, n_rows, dim, rng, std=0.02, dtype=np.float32):
        self.params = {"E": init_normal(rng, (n_rows, dim), std, dtype)}
        self.grads = {}

    def forward(self, ids):
        self.ids = ids
        return self.params["E"][ids]

    def backward(self, dy):
        dE = np.zeros_like(self.params["E"])
        np.add.at(dE, self.ids.reshape(-1), dy.reshape(-1, dy.shape[-1]))  # repeated ids add up
        self.grads["E"] = dE
        return None                                         # ids are not learnable


# ------------------------------------------------------------------------- LayerNorm
class LayerNorm:
    """Rescale each vector to mean 0 and spread 1, then apply a learned scale (g) and shift (b).

    Keeps the numbers flowing through a deep network in a healthy range (Ba et al., 2016).
    """

    def __init__(self, dim, eps=1e-5, dtype=np.float32):
        self.params = {"g": np.ones(dim, dtype=dtype), "b": np.zeros(dim, dtype=dtype)}
        self.grads = {}
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mean) * self.rstd
        return self.xhat * self.params["g"] + self.params["b"]

    def backward(self, dy):
        D = dy.shape[-1]
        self.grads["g"] = (dy * self.xhat).reshape(-1, D).sum(axis=0)
        self.grads["b"] = dy.reshape(-1, D).sum(axis=0)
        dxhat = dy * self.params["g"]
        # derivative of normalization: remove the parts that only shift or stretch the vector
        return self.rstd * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                            - self.xhat * (dxhat * self.xhat).mean(axis=-1, keepdims=True))


# ------------------------------------------------------------------------------ GELU
class GELU:
    """A smooth "let positive values through, damp negative ones" curve (Hendrycks & Gimpel)."""

    C = float(np.sqrt(2.0 / np.pi))     # a plain Python number keeps arrays in 32-bit

    def __init__(self):
        self.params, self.grads = {}, {}

    def forward(self, x):
        self.x = x
        self.t = np.tanh(self.C * (x + 0.044715 * x ** 3))
        return 0.5 * x * (1.0 + self.t)

    def backward(self, dy):
        x, t = self.x, self.t
        dt = (1.0 - t * t) * self.C * (1.0 + 3 * 0.044715 * x * x)
        return dy * (0.5 * (1.0 + t) + 0.5 * x * dt)


# ------------------------------------------------------- relative positions (rotary)
def rotary_tables(T, head_dim, base=10000.0, dtype=np.float32):
    """cos and sin tables (T, head_dim // 2) for rotary position encoding (Su et al., 2021).

    Each pair of numbers in a query or key is turned by an angle that grows with the position.
    The match between a query at position i and a key at position j then depends only on
    i - j (how far back), not on where they are: "the token right before me" means the same
    thing everywhere, which is exactly what copying needs.
    """
    half = head_dim // 2
    frequencies = base ** (-np.arange(half, dtype=np.float64) / half)
    angles = np.arange(T, dtype=np.float64)[:, None] * frequencies[None, :]
    return np.cos(angles).astype(dtype), np.sin(angles).astype(dtype)


def rotate(x, cos, sin):
    """Turn each pair (first half, second half) of the last axis by the position's angle."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def unrotate(dy, cos, sin):
    """Backward of rotate(): a rotation's gradient is the opposite rotation (its transpose)."""
    half = dy.shape[-1] // 2
    d1, d2 = dy[..., :half], dy[..., half:]
    return np.concatenate([d1 * cos + d2 * sin, d2 * cos - d1 * sin], axis=-1)


# ------------------------------------------------------------------ causal attention
class CausalSelfAttention:
    """Each position looks back at every earlier position and gathers what it needs.

    For every position, three vectors are computed from its current representation:
        query  "what am I looking for?"
        key    "what do I contain?"
        value  "what I hand over if chosen"
    scores  = query . key / sqrt(head size)       how well each earlier position matches
    (future positions are masked out: the model may never peek at what comes next)
    weights = softmax(scores)                     attention: each row sums to 1
    output  = weights @ values                    a weighted mix of earlier positions
    With H heads this happens H times in parallel on smaller slices, so each head can
    learn a different kind of lookup (Vaswani et al., 2017).

    rotary=True: queries and keys are rotated by their position first (see rotary_tables), so
    attention can look things up by RELATIVE position ("one step after that").
    """

    def __init__(self, dim, heads, rng, proj_std=0.02, dtype=np.float32, rotary=False):
        assert dim % heads == 0, "width must divide evenly into heads"
        self.heads, self.head_dim, self.rotary = heads, dim // heads, rotary
        self._tables = {}
        self.qkv = Linear(dim, 3 * dim, rng, dtype=dtype)                 # all three at once
        self.proj = Linear(dim, dim, rng, std=proj_std, dtype=dtype)      # mix the heads
        self.params = {**{f"qkv.{k}": v for k, v in self.qkv.params.items()},
                       **{f"proj.{k}": v for k, v in self.proj.params.items()}}
        self.grads = {}
        self.last_weights = None

    def _split(self, t):          # (B, T, D) -> (B, H, T, head_dim)
        B, T, _ = t.shape
        return t.reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge(self, t):          # (B, H, T, head_dim) -> (B, T, D)
        B, H, T, hd = t.shape
        return t.transpose(0, 2, 1, 3).reshape(B, T, H * hd)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = np.split(self.qkv.forward(x), 3, axis=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        if self.rotary:
            if (T, x.dtype) not in self._tables:
                self._tables[T, x.dtype] = rotary_tables(T, self.head_dim, dtype=x.dtype)
            cos, sin = self._tables[T, x.dtype]
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        scale = float(1.0 / np.sqrt(self.head_dim))        # plain Python number: stays 32-bit
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale                   # (B, H, T, T)
        future = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(future, np.float32(-1e9), scores)
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)                   # softmax per row
        out = self._merge(weights @ v)                                   # (B, T, D)
        self.cache = (q, k, v, weights, scale)
        self.last_weights = weights
        return self.proj.forward(out)

    def backward(self, dy):
        q, k, v, weights, scale = self.cache
        dout = self._split(self.proj.backward(dy))                       # (B, H, T, hd)
        dweights = dout @ v.transpose(0, 1, 3, 2)                        # (B, H, T, T)
        dv = weights.transpose(0, 1, 3, 2) @ dout
        # softmax backward: d score_j = w_j * (d w_j - sum_i w_i d w_i); masked w = 0 -> 0
        dscores = weights * (dweights - (dweights * weights).sum(axis=-1, keepdims=True)) * scale
        dq = dscores @ k
        dk = dscores.transpose(0, 1, 3, 2) @ q
        if self.rotary:                                   # q and k were rotated: turn the gradients back
            cos, sin = self._tables[q.shape[2], q.dtype]
            dq, dk = unrotate(dq, cos, sin), unrotate(dk, cos, sin)
        dqkv = np.concatenate([self._merge(dq), self._merge(dk), self._merge(dv)], axis=-1)
        dx = self.qkv.backward(dqkv)
        self.grads = {**{f"qkv.{k2}": g for k2, g in self.qkv.grads.items()},
                      **{f"proj.{k2}": g for k2, g in self.proj.grads.items()}}
        return dx


# --------------------------------------------------------------------- transformer block
class Block:
    """x = x + Attention(LayerNorm(x));  x = x + MLP(LayerNorm(x)).

    The "x +" (a residual connection, He et al. 2015) lets information and gradients pass
    straight through, which makes stacking several blocks trainable.
    """

    def __init__(self, dim, heads, rng, n_layers, dtype=np.float32, rotary=False):
        deep_std = 0.02 / np.sqrt(2 * n_layers)       # smaller start for the residual outputs
        self.ln1 = LayerNorm(dim, dtype=dtype)
        self.attn = CausalSelfAttention(dim, heads, rng, proj_std=deep_std, dtype=dtype, rotary=rotary)
        self.ln2 = LayerNorm(dim, dtype=dtype)
        self.fc1 = Linear(dim, 4 * dim, rng, dtype=dtype)
        self.gelu = GELU()
        self.fc2 = Linear(4 * dim, dim, rng, std=deep_std, dtype=dtype)
        self.parts = {"ln1": self.ln1, "attn": self.attn, "ln2": self.ln2,
                      "fc1": self.fc1, "fc2": self.fc2}
        self.params = {f"{p}.{k}": v for p, layer in self.parts.items() for k, v in layer.params.items()}
        self.grads = {}

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        return x + self.fc2.forward(self.gelu.forward(self.fc1.forward(self.ln2.forward(x))))

    def backward(self, dy):
        dx = dy + self.ln2.backward(self.fc1.backward(self.gelu.backward(self.fc2.backward(dy))))
        dx = dx + self.ln1.backward(self.attn.backward(dx))
        self.grads = {f"{p}.{k}": g for p, layer in self.parts.items() for k, g in layer.grads.items()}
        return dx
