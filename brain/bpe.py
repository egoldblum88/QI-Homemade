"""
Our own subword tokenizer (byte-pair encoding, BPE), learned from OUR training text.

Why: with one token per character, 256 positions hold only ~256 characters (about 50 words).
BPE learns frequent pieces of words (" the", "ing", " said") as single tokens, so the same
256 positions hold roughly 3-4 times more text: more of the conversation fits, and each
training step covers more language.

How it is learned (Gage 1994; Sennrich et al. 2016), written by us:
  1. split the text into chunks: a word with its leading space (" said"), a single digit,
     a run of punctuation, spaces, or a new line
  2. start with single characters as tokens
  3. repeatedly find the most frequent pair of neighbouring tokens inside chunks and merge it
     into one new token (e.g. "t"+"h" -> "th", then "th"+"e" -> "the", then " "+"the" -> " the")
  4. stop when the vocabulary is big enough

Two deliberate rules: DIGITS ARE NEVER MERGED ("10" is always "1","0", so arithmetic sees
every number the same way) and NEW LINES ARE NEVER MERGED (a reply ends at a new line).

It has the same interface as CharTokenizer (encode, decode, special, vocab_size, ...).
Ids: single characters first, then learned pieces, then the special markers.
"""

import re
from collections import Counter, defaultdict

CHUNK_RE = re.compile(r"\n|[0-9]| ?[A-Za-z]+| ?[^A-Za-z0-9\s]+| +|\s")


def learn_merges(text, vocab_size, n_specials=3, log=None):
    """Learn merge rules from text until the vocabulary reaches vocab_size."""
    base = sorted(set(text))
    chunk_counts = Counter(CHUNK_RE.findall(text))
    words = [(list(chunk), count) for chunk, count in chunk_counts.items() if len(chunk) > 1]
    pair_counts, where = Counter(), defaultdict(set)
    for i, (symbols, count) in enumerate(words):
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += count
            where[pair].add(i)
    merges = []
    while len(base) + len(merges) + n_specials < vocab_size and pair_counts:
        best, count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        if count < 2:
            break
        merges.append(best)
        merged = best[0] + best[1]
        for i in list(where[best]):
            symbols, c = words[i]
            for pair in zip(symbols, symbols[1:]):              # forget this word's old pairs
                pair_counts[pair] -= c
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                where[pair].discard(i)
            out, j = [], 0
            while j < len(symbols):
                if j + 1 < len(symbols) and symbols[j] == best[0] and symbols[j + 1] == best[1]:
                    out.append(merged)
                    j += 2
                else:
                    out.append(symbols[j])
                    j += 1
            words[i] = (out, c)
            for pair in zip(out, out[1:]):                      # count its new pairs
                pair_counts[pair] += c
                where[pair].add(i)
        if log and len(merges) % 500 == 0:
            log(f"    {len(merges)} merges, latest: {merged!r}")
    return "".join(base), merges


class BPETokenizer:

    def __init__(self, chars, merges, specials=()):
        self.chars = "".join(sorted(set(chars)))                  # the single characters
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.merges = [tuple(m) for m in merges]
        self.tokens = list(self.chars) + [a + b for a, b in self.merges]   # id -> text
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.rank = {pair: r for r, pair in enumerate(self.merges)}
        self.specials = list(specials)
        self.special_to_id = {name: len(self.tokens) + i for i, name in enumerate(self.specials)}
        self._cache = {}

    @classmethod
    def train(cls, text, vocab_size, specials=(), log=None):
        chars, merges = learn_merges(text, vocab_size, len(specials), log)
        return cls(chars, merges, specials)

    @property
    def vocab_size(self):
        return len(self.tokens) + len(self.specials)

    def special(self, name):
        return self.special_to_id[name]

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        symbols = list(chunk)
        while len(symbols) > 1:
            ranked = [(self.rank.get(p, len(self.rank)), k) for k, p in enumerate(zip(symbols, symbols[1:]))]
            best_rank, k = min(ranked)
            if best_rank == len(self.rank):
                break                                            # no learned merge applies
            symbols[k:k + 2] = [symbols[k] + symbols[k + 1]]
        ids = [self.token_to_id[s] for s in symbols]
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text):
        """Text -> ids. Characters outside the vocabulary are skipped."""
        text = "".join(ch for ch in text if ch in self.char_to_id)
        return [i for chunk in CHUNK_RE.findall(text) for i in self._encode_chunk(chunk)]

    def decode(self, ids, show_specials=False):
        out = []
        for i in ids:
            i = int(i)
            if i < len(self.tokens):
                out.append(self.tokens[i])
            elif show_specials:
                out.append(self.specials[i - len(self.tokens)])
        return "".join(out)

    def unknown_characters(self, text):
        return sorted({ch for ch in text if ch not in self.char_to_id})
