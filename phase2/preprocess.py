# phase 2 preprocessing: piece tokenizer with character offsets, BIOES tags, windows
import hashlib
import json
import os
import re

import numpy as np
import torch

DATA_DIR = "data"
TOKENIZER_PATH = "legal_bpe_16384.json"

LABELS = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]
TAGS = ["O"]
for l in LABELS:
    for p in "BIES":
        TAGS.append(p + "-" + l)
TAG_ID = {}
for i, t in enumerate(TAGS):
    TAG_ID[t] = i
N_TAGS = len(TAGS)
RARE_LABELS = {"APP", "RESP", "A.COUNSEL", "R.COUNSEL", "WIT"}
PIECE_RE = re.compile(r"[^\W\d_]+|\d+|_|[^\w\s]")
IGNORE = -100


class BPE:
    # merges of my phase 1 BPE tokenizer (A1 code), only what encoding needs

    def __init__(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.eow_id = 260
        self.merge_rank = {}
        self.token_bytes = {}
        for x in range(256):
            self.token_bytes[x + 4] = bytes([x])
        self.token_bytes[self.eow_id] = b""
        rank = 0
        for pair, new_id in data["merges"]:
            a, b = pair
            self.merge_rank[(a, b)] = (rank, new_id)
            self.token_bytes[new_id] = self.token_bytes[a] + self.token_bytes[b]
            rank += 1


class PieceTokenizer:
    # each word is split into letter runs, digit runs and punctuation before BPE
    # only the last piece of a word gets the end-of-word marker

    def __init__(self, bpe):
        self.bpe = bpe
        self.cache = {}

    def encode_piece(self, s, eow):
        key = (s, eow)
        if key in self.cache:
            return self.cache[key]
        toks = [b + 4 for b in s.encode("utf-8")]
        if eow:
            toks.append(self.bpe.eow_id)
        while len(toks) >= 2:
            best = None
            best_i = -1
            for i in range(len(toks) - 1):
                r = self.bpe.merge_rank.get((toks[i], toks[i + 1]))
                if r is not None and (best is None or r[0] < best[0]):
                    best = r
                    best_i = i
            if best_i < 0:
                break
            toks = toks[:best_i] + [best[1]] + toks[best_i + 2:]
        out = []
        for t in toks:
            out.append((t, len(self.bpe.token_bytes[t])))
        self.cache[key] = out
        return out

    def encode(self, text):
        ids = []
        starts = []
        ends = []
        for m in re.finditer(r"\S+", text):
            word = m.group()
            w0 = m.start()
            pieces = []
            for p in PIECE_RE.finditer(word):
                pieces.append((p.start(), p.end()))
            for k in range(len(pieces)):
                a, b = pieces[k]
                seg = word[a:b]
                # byte position to character position inside the piece
                b2c = []
                for ci in range(len(seg)):
                    b2c += [ci] * len(seg[ci].encode("utf-8"))
                b2c.append(len(seg))
                pos = 0
                for t, nb in self.encode_piece(seg, k == len(pieces) - 1):
                    ids.append(t)
                    starts.append(w0 + a + b2c[pos])
                    pos += nb
                    ends.append(w0 + a + b2c[pos])
        ids = np.array(ids, dtype=np.int64)
        starts = np.array(starts, dtype=np.int64)
        ends = np.array(ends, dtype=np.int64)
        return ids, starts, ends


def load_tokenizer(path=TOKENIZER_PATH):
    return PieceTokenizer(BPE(path))


def spans_to_tags(starts, ends, spans):
    tags = np.zeros(len(starts), dtype=np.int64)
    exact = 0
    for sp in spans:
        s, e, lab = sp[0], sp[1], sp[2]
        lo = np.searchsorted(starts, s, side="left")
        hi = np.searchsorted(starts, e, side="left")
        idx = []
        for i in range(lo, hi):
            zero_width_edge = starts[i] == ends[i] and starts[i] in (s, e)
            if ends[i] <= e and not zero_width_edge:
                idx.append(i)
        pos = [i for i in idx if ends[i] > starts[i]]
        if len(pos) == 0:
            continue
        if len(pos) == 1:
            tags[pos[0]] = TAG_ID["S-" + lab]
        else:
            for i in idx:
                tags[i] = TAG_ID["I-" + lab]
            tags[pos[0]] = TAG_ID["B-" + lab]
            tags[pos[-1]] = TAG_ID["E-" + lab]
        if starts[pos[0]] == s and ends[pos[-1]] == e:
            exact += 1
    return tags, exact


def tags_to_spans(tags, starts, ends, probs):
    out = []
    i = 0
    n = len(tags)
    while i < n:
        t = TAGS[tags[i]]
        if t.startswith("S-"):
            out.append((int(starts[i]), int(ends[i]), t[2:], float(probs[i])))
        elif t.startswith("B-"):
            lab = t[2:]
            j = i + 1
            while j < n and TAGS[tags[j]] == "I-" + lab:
                j += 1
            if j < n and TAGS[tags[j]] == "E-" + lab:
                out.append((int(starts[i]), int(ends[j]), lab, float(np.mean(probs[i:j + 1]))))
                i = j
        i += 1
    return out


def load_split(data_dir, folds):
    texts = {}
    gold = {}
    for f in folds:
        with open(os.path.join(data_dir, f + "-text.json"), encoding="utf-8") as fh:
            texts.update(json.load(fh))
        gold_path = os.path.join(data_dir, f + "-gold.json")
        if os.path.exists(gold_path):
            with open(gold_path, encoding="utf-8") as fh:
                gold.update(json.load(fh))
    return texts, gold


def encode_docs(ptok, texts, gold=None):
    docs = {}
    for d, text in texts.items():
        ids, st, en = ptok.encode(text)
        doc = {"id": d, "text": text, "ids": ids, "starts": st, "ends": en}
        if gold is not None and d in gold:
            doc["tags"], doc["exact"] = spans_to_tags(st, en, gold[d])
            doc["gold"] = [(sp[0], sp[1], sp[2]) for sp in gold[d]]
        docs[d] = doc
    return docs


def window_starts(n, length, stride):
    if n <= length:
        return [0]
    s = list(range(0, n - length + 1, stride))
    if s[-1] != n - length:
        s.append(n - length)
    return s


def training_windows(docs, length, stride, rare_copies=1):
    out = []
    for doc in docs.values():
        n = len(doc["ids"])
        for s in window_starts(n, length, stride):
            w = (doc["id"], s, min(length, n))
            out.append(w)
            if rare_copies > 1:
                labs = set()
                for t in doc["tags"][s:s + length]:
                    if t:
                        labs.add(TAGS[t][2:])
                if labs & RARE_LABELS:
                    out += [w] * (rare_copies - 1)
    return out


def class_weights(docs, power=0.5, cap=10.0):
    cnt = np.ones(N_TAGS)
    for doc in docs.values():
        cnt += np.bincount(doc["tags"], minlength=N_TAGS)
    w = (cnt.max() / cnt) ** power
    w = np.minimum(w / w[0], cap)
    return torch.tensor(w, dtype=torch.float32)


def phase2_keys(texts):
    # used to skip phase 1 judgments that are also phase 2 train or test documents
    phase2 = []
    for t in texts:
        if t:
            phase2.append(" ".join(t.split()))
    keys = set()
    snips = []
    for t in phase2:
        keys.add(hashlib.md5(t[:200].lower().encode()).hexdigest())
        for i in (2000, 6000, 10000):
            if len(t) > i + 120:
                snips.append(t[i:i + 120])
    return keys, snips


def is_phase2(t, keys, snips):
    if hashlib.md5(t[:200].lower().encode()).hexdigest() in keys:
        return True
    for s in snips:
        if s in t:
            return True
    return False
