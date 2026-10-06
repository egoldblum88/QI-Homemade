"""
The tokenizer: turns text into numbers, and numbers back into text.

A neural network can only do arithmetic, so every character must become a number first.
We use one number per character.

    vocabulary        every distinct character in the training text, sorted
                      (newline, space, punctuation, digits, A-Z, a-z)
    encode("Hi")  ->  [34, 64]      each character's position in the vocabulary
    decode([34, 64]) -> "Hi"

Special tokens (Stage 1.5): a few extra numbers that are not characters at all.
They mark the structure of a conversation, and nobody can type them in the chat:

    <user>   a message from the person starts here
    <ai>     the reply starts here
    <end>    the reply is finished

They get the numbers right after the characters. For 83 characters: <user>=83, <ai>=84, <end>=85.
The Stage 1 brain has no special tokens.

The model can only receive characters that are in its vocabulary. encode() skips any
other character (an emoji, for example), and unknown_characters() says which ones.
"""


class CharTokenizer:

    def __init__(self, chars, specials=()):
        chars = "".join(sorted(set(chars)))
        if not chars:
            raise ValueError("the vocabulary is empty")
        self.chars = chars                                          # id -> character
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}     # character -> id
        self.specials = list(specials)                              # e.g. ["<user>", "<ai>", "<end>"]
        self.special_to_id = {name: len(chars) + i for i, name in enumerate(self.specials)}

    @classmethod
    def from_text(cls, text, specials=()):
        """Build the vocabulary from every character that appears in the text."""
        return cls(set(text), specials)

    @property
    def vocab_size(self):
        return len(self.chars) + len(self.specials)

    def special(self, name):
        """The id of a special token, e.g. special("<ai>")."""
        return self.special_to_id[name]

    def encode(self, text):
        """Text -> list of ids. Characters outside the vocabulary are skipped."""
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]

    def decode(self, ids, show_specials=False):
        """List of ids -> text. Special tokens are left out unless show_specials=True."""
        out = []
        for i in ids:
            i = int(i)
            if i < len(self.chars):
                out.append(self.chars[i])
            elif show_specials:
                out.append(self.specials[i - len(self.chars)])
        return "".join(out)

    def unknown_characters(self, text):
        """The characters in `text` that the model has never seen."""
        return sorted({ch for ch in text if ch not in self.char_to_id})
