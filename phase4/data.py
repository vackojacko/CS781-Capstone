# phase 4 data: cleaning, tokenizing and chunks counted back from the end of each judgment
import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np

from tokenizer import BPETokenizer, PieceTokenizer

DATA_DIR = "data/cjpe"
BPE_PATH = "data/legal_bpe_16384.json"
TOKEN_DIR = "tokens"

TRAIN_TAIL = 6400   # tokens kept from the end of a train case (8 chunks of 1024 with stride 768)
EVAL_TAIL = 50000   # dev/test: practically the whole case
TRAIN_WORDS = 5600
WINDOW = {"arch1": 1024, "arch2": 1024, "arch3": 704, "arch4": 1024}
PIECE_RE = re.compile(r"[^\W\d_]+|\d+|_|[^\w\s]")


def align_train(t):
    # make train look like dev/test (lowercased, the anonymiser also hit capitals and 'no' inside words)
    t = re.sub(r"\bCo[a-z]", "company", t)
    t = re.sub(r"\bNo\b\.?|\bNo[a-z]", "number", t)
    t = re.sub(r"(?<=[A-Za-z])no[a-z]", "number", t)
    return re.sub(r"\s+", " ", t.lower()).strip()


def clean_eval(t):
    return re.sub(r"\s+", " ", t).strip()


def load_split(path, split):
    rows = json.load(open(path))
    seen = set()
    out = []
    for r in rows:
        # train has 5,082 cases twice, dev has every case twice
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        if split == "train":
            text = align_train(r["text"])
        else:
            text = clean_eval(r["text"])
        # test labels are never read
        if split == "test":
            label = -1
        else:
            label = int(r["label"])
        out.append({"id": r["id"], "text": text, "label": label})
    return out


def encode_ids(tok, text, max_words=None):
    words = text.split()
    if max_words:
        words = words[-max_words:]
    ids = []
    for w in words:
        pieces = PIECE_RE.findall(w)
        for k in range(len(pieces)):
            ids += tok.encode_piece(pieces[k], k == len(pieces) - 1)
    return ids


def save_tokens(path, cases):
    lens = []
    labels = []
    case_ids = []
    arrays = []
    for c in cases:
        lens.append(len(c["ids"]))
        labels.append(c["label"])
        case_ids.append(c["id"])
        arrays.append(np.asarray(c["ids"], dtype=np.int16))
    np.savez(path, ids=np.concatenate(arrays), lens=np.array(lens, dtype=np.int64),
             labels=np.array(labels, dtype=np.int64), case_ids=np.array(case_ids))


def load_tokens(path):
    z = np.load(path)
    ids = z["ids"]
    lens = z["lens"]
    labels = z["labels"]
    case_ids = z["case_ids"]
    ends = np.cumsum(lens)
    cases = []
    for i in range(len(lens)):
        b = ends[i]
        cases.append({"id": str(case_ids[i]), "ids": ids[b - lens[i]:b], "label": int(labels[i])})
    return cases


def tail_starts(n, length, stride, k=None):
    # starts of chunks counted back from the end of the case, returned in reading order
    if n <= length:
        return [0]
    s = list(range(n - length, -1, -stride))
    if s[-1] != 0:
        s.append(0)
    if k:
        s = s[:k]
    return s[::-1]


def chunk_length(bb):
    return WINDOW[bb.arch] - len(bb.prefix)


def chunk_ids(case, bb, length, stride, k=None):
    pre = list(bb.prefix)
    n = len(case["ids"])
    chunks = []
    for s in tail_starts(n, length, stride, k):
        chunks.append(pre + case["ids"][s:s + length].tolist())
    return chunks


def make_batches(items, batch):
    # same-length chunks only, arch4 has no padding mask
    by_len = {}
    for it in items:
        n = len(it[1])
        if n not in by_len:
            by_len[n] = []
        by_len[n].append(it)
    out = []
    for group in by_len.values():
        for i in range(0, len(group), batch):
            out.append(group[i:i + batch])
    return out


def work(args):
    texts, max_words, keep = args
    tok = PieceTokenizer(BPETokenizer.load(BPE_PATH))
    out = []
    for text in texts:
        ids = encode_ids(tok, text, max_words)
        out.append(ids[-keep:])
    return out


if __name__ == "__main__":
    os.makedirs(TOKEN_DIR, exist_ok=True)
    splits = [("dev", None, EVAL_TAIL), ("test", None, EVAL_TAIL), ("train", TRAIN_WORDS, TRAIN_TAIL)]
    for split, max_words, keep in splits:
        t0 = time.time()
        cases = load_split(os.path.join(DATA_DIR, split + ".json"), split)
        jobs = []
        for i in range(0, len(cases), 500):
            texts = [c["text"] for c in cases[i:i + 500]]
            jobs.append((texts, max_words, keep))
        with Pool(4) as pool:
            parts = pool.map(work, jobs)
        ids = []
        for part in parts:
            ids += part
        for c, x in zip(cases, ids):
            c["ids"] = x
        save_tokens(os.path.join(TOKEN_DIR, split + ".npz"), cases)
        lens = sorted([len(x) for x in ids])
        print(split, len(cases), "cases | tokens median", lens[len(lens) // 2], "p90", lens[int(0.9 * len(lens))],
              "| time", int(time.time() - t0), "s")
