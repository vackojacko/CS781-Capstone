import json
import re

TOKENIZER_PATH = "../phase1/legal_bpe_16384.json"
PIECE_RE = re.compile(r"[^\W\d_]+|\d+|_|[^\w\s]")
MAX_TOKENS = 20000
EOW_ID = 260


# phase 1 bpe, only the last piece of a word gets </w>
class PieceTokenizer:
    def __init__(self, path=TOKENIZER_PATH):
        merges = json.load(open(path))["merges"]
        self.merge_rank = {}
        for rank in range(len(merges)):
            pair, new_id = merges[rank]
            self.merge_rank[(pair[0], pair[1])] = (rank, new_id)
        self.cache = {}

    def encode_piece(self, s, eow):
        key = (s, eow)
        if key in self.cache:
            return self.cache[key]
        toks = []
        for b in s.encode("utf-8"):
            toks.append(b + 4)
        if eow:
            toks.append(EOW_ID)
        while len(toks) >= 2:
            best = None
            bi = -1
            for i in range(len(toks) - 1):
                r = self.merge_rank.get((toks[i], toks[i + 1]))
                if r is not None and (best is None or r[0] < best[0]):
                    best = r
                    bi = i
            if bi < 0:
                break
            toks = toks[:bi] + [best[1]] + toks[bi + 2:]
        self.cache[key] = toks
        return toks


def encode_ids(tok, text):
    ids = []
    for w in text.split():
        pieces = PIECE_RE.findall(w)
        for k in range(len(pieces)):
            ids += tok.encode_piece(pieces[k], k == len(pieces) - 1)
    if len(ids) > MAX_TOKENS:
        ids = ids[:MAX_TOKENS - 1024] + ids[-1024:]
    return ids
