import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np

from tokenizer import BPETokenizer, PieceTokenizer

DATA_DIR = "data/lsi"
TOKEN_DIR = "data/lsi_tokens"
TOKENIZER_PATH = "tokenizer/legal_bpe_16384.json"

N_LABELS = 100
MAX_TOKENS = 20000
WINDOW = {"arch1": 1024, "arch2": 1024, "arch3": 704, "arch4": 1024}


def clean(t):
    # <ENTITY> costs 7 of my tokens, X keeps the slot cheaply
    t = t.replace("<ENTITY>", "X").replace("<SECTION>", "Section").replace("<ACT>", "Act")
    return re.sub(r"\s+", " ", t).strip()


def load_split(path, split):
    out = []
    for r in json.load(open(path)):
        labels = []
        if split != "test":
            labels = sorted(set(r["labels"]))
        out.append({"id": r["id"], "text": clean(r["text"]), "labels": labels})
    return out


def encode_ids(tok, text):
    ids = tok.encode(text)
    if len(ids) > MAX_TOKENS:
        ids = ids[:MAX_TOKENS - 1024] + ids[-1024:]
    return ids


def multi_hot(labels):
    y = np.zeros(N_LABELS, dtype=np.uint8)
    y[labels] = 1
    return y


def save_tokens(path, cases):
    lens = np.array([len(c["ids"]) for c in cases], dtype=np.int64)
    ids = np.concatenate([np.asarray(c["ids"], dtype=np.int16) for c in cases])
    y = np.stack([multi_hot(c["labels"]) for c in cases])
    case_ids = np.array([c["id"] for c in cases])
    np.savez(path, ids=ids, lens=lens, y=y, case_ids=case_ids)


def load_tokens(path):
    z = np.load(path)
    ids = z["ids"]
    lens = z["lens"]
    ends = np.cumsum(lens)
    cases = []
    for i in range(len(lens)):
        b = ends[i]
        cases.append({"id": str(z["case_ids"][i]), "ids": ids[b - lens[i]:b], "y": z["y"][i]})
    return cases


def window_starts(n, length, stride):
    if n <= length:
        return [0]
    s = list(range(0, n - length + 1, stride))
    if s[-1] != n - length:
        s.append(n - length)
    return s


def chunk_length(bb):
    return WINDOW[bb.arch] - len(bb.prefix)


def case_chunks(case, bb, k):
    # quarter overlap, a longer case keeps its first k-1 chunks and its last one
    L = chunk_length(bb)
    s = window_starts(len(case["ids"]), L, L * 3 // 4)
    if len(s) > k:
        s = s[:k - 1] + [s[-1]]
    chunks = []
    for a in s:
        chunks.append(list(bb.prefix) + case["ids"][a:a + L].tolist())
    return chunks


def encode_part(texts):
    tok = PieceTokenizer(BPETokenizer.load(TOKENIZER_PATH))
    ids = []
    for text in texts:
        ids.append(encode_ids(tok, text))
    return ids


if __name__ == "__main__":
    os.makedirs(TOKEN_DIR, exist_ok=True)
    for split in ["dev", "test", "train"]:
        t0 = time.time()
        cases = load_split(f"{DATA_DIR}/{split}.json", split)
        parts = []
        for i in range(0, len(cases), 1000):
            parts.append([c["text"] for c in cases[i:i + 1000]])
        ids = []
        with Pool(4) as pool:
            for part in pool.map(encode_part, parts):
                ids += part
        for i in range(len(cases)):
            cases[i]["ids"] = ids[i]
        save_tokens(f"{TOKEN_DIR}/{split}.npz", cases)
        lens = sorted([len(x) for x in ids])
        median = lens[len(lens) // 2]
        p90 = lens[int(0.9 * len(lens))]
        n_labels = sum([len(c["labels"]) for c in cases]) / len(cases)
        secs = time.time() - t0
        print(split, len(cases), "cases | median tokens", median, "p90", p90,
              f"| labels per case {n_labels:.2f} | {secs:.0f}s")
