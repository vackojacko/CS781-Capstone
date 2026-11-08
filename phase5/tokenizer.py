import json
import re

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

    def _rebuild_caches(self):
        self._merge_rank = {}
        for rank, merge in enumerate(self.merges):
            pair, new_id = merge
            self._merge_rank[pair] = (rank, new_id)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls(data["target_vocab_size"])
        for pair, c in data["merges"]:
            tok.merges.append(((pair[0], pair[1]), c))
        for pair, new_id in tok.merges:
            tok.itos.append(pair)
        tok.vocab_size = len(tok.itos)
        tok._rebuild_caches()
        return tok


# bpe on the letter, digit and punctuation pieces of each word, only the last piece gets </w>
class PieceTokenizer:

    def __init__(self, bpe):
        self.bpe = bpe
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
            best = None
            bi = -1
            for i in range(len(toks) - 1):
                r = rank.get((toks[i], toks[i + 1]))
                if r is not None and (best is None or r[0] < best[0]):
                    best = r
                    bi = i
            if bi < 0:
                break
            toks = toks[:bi] + [best[1]] + toks[bi + 2:]
        self.cache[key] = toks
        return toks

    def encode(self, text):
        ids = []
        for w in text.split():
            pieces = PIECE_RE.findall(w)
            for k in range(len(pieces)):
                ids += self.encode_piece(pieces[k], k == len(pieces) - 1)
        return ids
