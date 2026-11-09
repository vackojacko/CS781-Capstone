import gzip
import pickle

import numpy as np

import data
import lexical
from tokenizer import PieceTokenizer, encode_ids

DATA_DIR = "data"
TOKENIZER_PATH = "../phase1/legal_bpe_16384.json"
PREP_FILE = "p7_data.pkl.gz"


if __name__ == "__main__":
    tok = PieceTokenizer(TOKENIZER_PATH)
    out = {}
    for split in ("train", "dev", "test"):
        q, c = data.load_raw(split, DATA_DIR)
        ids = {}
        for k in q:
            ids[k] = encode_ids(tok, data.model_text(q[k]["text"]))
        for k in c:
            if k not in q:
                ids[k] = encode_ids(tok, data.model_text(c[k]["text"]))

        # tf-idf top 50 per query for the hard negatives
        queries, cands = data.load(split, DATA_DIR)
        S, qids, cids = lexical.candidate_tfidf(queries, cands, ngram=(4, 6))
        top = {}
        for a in range(len(qids)):
            order = np.argsort(-S[a])[:50]
            pairs = []
            for j in order:
                pairs.append((cids[j], float(S[a, j])))
            top[qids[a]] = pairs
        relevant = {}
        for k in qids:
            relevant[k] = queries[k]["relevant"]
        out[split] = {"ids": ids, "queries": relevant, "cands": cids, "tfidf_top50": top,
                      "tfidf": S.astype(np.float16), "qids": qids}
        lens = []
        for v in ids.values():
            lens.append(len(v))
        print(split, len(qids), "queries,", len(cids), "candidates, median tokens", int(np.median(lens)))
    with gzip.open(PREP_FILE, "wb") as f:
        pickle.dump(out, f)
    print("saved", PREP_FILE)
