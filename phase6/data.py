# phase 6 data: sentence splitting, header removal, tokenizing and oracle labels for every judgment
import gzip
import json
import os
import pickle
import re
from multiprocessing import Pool

import numpy as np

from tokenizer import BPETokenizer, PieceTokenizer
import oracle

TRAIN_JSON = "data/Phase_6_Summ/Train/train.json"
TEST_JSON = "data/Phase_6_Summ/Test/test.json"
TOKENIZER_PATH = "data/legal_bpe_16384.json"
OUT_DIR = "work"

MAX_TOKS = 128
SEP_ID = 3
DEV_SIZE = 300

ABBREV = {"no.", "nos.", "mr.", "mrs.", "ms.", "dr.", "vs.", "sec.", "art.", "ltd.", "pvt.", "co.", "ors.", "anr.", "sr.", "jr.",
          "viz.", "i.e.", "e.g.", "para.", "cl.", "st.", "hon'ble", "rs.", "j.", "w.e.f.", "smt.", "shri.", "m/s.", "u/s."}


def split_sentences(text):
    # split on . ? ! followed by a capital letter, not after abbreviations like "No." "Mr." "v." "S."
    text = " ".join(text.split())
    sents, start = [], 0
    for m in re.finditer(r"[.?!]\s+(?=[A-Z(\"'])", text):
        words = text[start:m.start() + 1].split()
        word = ""
        if words:
            word = words[-1].lower().strip("(")
        if len(word) <= 2 or word in ABBREV:
            continue
        sents.append(text[start:m.end()].strip())
        start = m.end()
    if text[start:].strip():
        sents.append(text[start:].strip())
    return [x for x in sents if len(x) > 3]


def header_end(sents):
    # the case header ends at "... delivered by ..."
    for i in range(min(40, len(sents))):
        if "delivered by" in sents[i]:
            return i + 1
    return 0


def prepare(doc, tokenizer):
    sents = split_sentences(doc)
    toks = []
    for s in sents:
        ids = tokenizer.encode(s)[:MAX_TOKS].tolist()
        if len(ids) == 0:
            ids = [SEP_ID]
        toks.append(ids)
    return sents, toks


def work(job):
    case, tokenizer = job
    sents, toks = prepare(case["document"], tokenizer)
    out = {"id": case["id"], "sents": sents, "toks": toks}
    if "summary" in case:
        out["summary"] = case["summary"]
        out["labels"] = oracle.oracle_labels(sents, case["summary"])
    return out


def load_docs(split):
    with gzip.open(os.path.join(OUT_DIR, "p6_" + split + ".pkl.gz"), "rb") as f:
        return pickle.load(f)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = PieceTokenizer(BPETokenizer.load(TOKENIZER_PATH))
    train = json.load(open(TRAIN_JSON))
    test = json.load(open(TEST_JSON))
    # same 300 dev judgments for the whole phase
    dev_idx = set(np.random.RandomState(0).choice(len(train), DEV_SIZE, replace=False).tolist())
    splits = {"train": [], "dev": [], "test": test}
    for i in range(len(train)):
        if i in dev_idx:
            splits["dev"].append(train[i])
        else:
            splits["train"].append(train[i])
    for split in ("dev", "test", "train"):
        jobs = []
        for case in splits[split]:
            jobs.append((case, tokenizer))
        with Pool(4) as pool:
            docs = pool.map(work, jobs, chunksize=16)
        n_sents = []
        for d in docs:
            n_sents.append(len(d["sents"]))
        print(split, len(docs), "docs | sentences median", int(np.median(n_sents)))
        with gzip.open(os.path.join(OUT_DIR, "p6_" + split + ".pkl.gz"), "wb") as f:
            pickle.dump(docs, f)


if __name__ == "__main__":
    main()
