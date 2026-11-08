# phase 1 bpe (my A1 tokenizer), applied to the letter, digit and punctuation pieces of every word
import json


class BPETokenizer:
    def __init__(self, vocab_size=16384):
        self.target_vocab_size = vocab_size
        self.eow_id = 4 + 256
        self.merges = []
        self.merge_rank = {}

    def rebuild_caches(self):
        self.merge_rank = {}
        for rank, (pair, new_id) in enumerate(self.merges):
            self.merge_rank[pair] = (rank, new_id)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls(data["target_vocab_size"])
        tok.merges = []
        for (a, b), c in data["merges"]:
            tok.merges.append(((a, b), c))
        tok.rebuild_caches()
        return tok


class PieceTokenizer:
    # only the last piece of a word gets the end-of-word marker, so tokens stay ones seen in pretraining
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
        rank = self.bpe.merge_rank
        while len(toks) >= 2:
            best = None
            best_i = -1
            for i in range(len(toks) - 1):
                r = rank.get((toks[i], toks[i + 1]))
                if r is not None and (best is None or r[0] < best[0]):
                    best = r
                    best_i = i
            if best_i < 0:
                break
            toks = toks[:best_i] + [best[1]] + toks[best_i + 2:]
        self.cache[key] = toks
        return toks
