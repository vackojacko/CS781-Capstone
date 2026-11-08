# phase 1 bpe tokenizer (from my A1), only the parts needed to load, encode and decode
import json
import re

import numpy as np

PIECE_RE = re.compile(r"[^\W\d_]+|\d+|_|[^\w\s]")


class BPETokenizer:

    def __init__(self, vocab_size=16384):
        self.target_vocab_size = vocab_size
        self.itos = ["<pad>", "<unk>", "<s>", "</s>"]
        self.itos.extend(range(256))
        self.eow_id = len(self.itos)
        self.itos.append("</w>")
        self.merges = []
        self.vocab_size = len(self.itos)
        self._merge_rank = {}
        self._token_bytes = None
        self._ends_word = None

    def _rebuild_caches(self):
        self._merge_rank = {}
        for rank, ((a, b), new_id) in enumerate(self.merges):
            self._merge_rank[(a, b)] = (rank, new_id)
        self._token_bytes = {}
        self._ends_word = {}
        for x in range(256):
            self._token_bytes[x + 4] = bytes([x])
            self._ends_word[x + 4] = False
        self._token_bytes[self.eow_id] = b""
        self._ends_word[self.eow_id] = True
        for (a, b), c in self.merges:
            self._token_bytes[c] = self._token_bytes[a] + self._token_bytes[b]
            self._ends_word[c] = self._ends_word[b]

    def decode(self, tokens):
        if self._token_bytes is None:
            self._rebuild_caches()
        parts = []
        for i, token in enumerate(tokens):
            b = self._token_bytes.get(token)
            if b is not None:
                parts.append(b)
                if self._ends_word.get(token, False) and i < len(tokens) - 1:
                    parts.append(b" ")
        return b"".join(parts).decode("utf-8", errors="replace")

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls(data["target_vocab_size"])
        for (a, b), c in data["merges"]:
            tok.merges.append(((a, b), c))
        for pair, new_id in tok.merges:
            tok.itos.append(pair)
        tok.vocab_size = len(tok.itos)
        tok._rebuild_caches()
        return tok


# bpe applied to the letter, digit and punctuation pieces of each word (phase 2)
class PieceTokenizer:

    def __init__(self, bpe):
        self.bpe = bpe
        bpe._rebuild_caches()
        self.cache = {}

    def encode_piece(self, s, eow):
        key = (s, eow)
        if key in self.cache:
            return self.cache[key]
        toks = []
        for b in s.encode("utf-8"):
            toks.append(b + 4)
        if eow:
            toks.append(self.bpe.eow_id)
        rank = self.bpe._merge_rank
        while len(toks) >= 2:
            best, bi = None, -1
            for i in range(len(toks) - 1):
                r = rank.get((toks[i], toks[i + 1]))
                if r is not None and (best is None or r[0] < best[0]):
                    best, bi = r, i
            if bi < 0:
                break
            toks = toks[:bi] + [best[1]] + toks[bi + 2:]
        out = []
        for t in toks:
            out.append((t, len(self.bpe._token_bytes[t])))
        self.cache[key] = out
        return out

    def encode(self, text):
        ids = []
        for m in re.finditer(r"\S+", text):
            pieces = PIECE_RE.findall(m.group())
            for k in range(len(pieces)):
                for t, nb in self.encode_piece(pieces[k], k == len(pieces) - 1):
                    ids.append(t)
        return np.array(ids, dtype=np.int64)
