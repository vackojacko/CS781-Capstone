import json
import os
import zipfile

import numpy as np

import data
import lexical
from citation_graph import cocitation_scores, neighbour_votes
from rerank import final_score

DATA_DIR = "data"
OUT_DIR = "submissions"
VOTE_K = 5
DENSE_FILE = "dense_runs/d2b_arch4_qlora/scores.npz"
W_DENSE = 0.25


def build_signals(split, graph):
    queries, cands = data.load(split, DATA_DIR)
    raw, _ = data.load_raw(split, DATA_DIR)
    views, qids, cids = lexical.lexical_views(queries, cands, raw)
    co = cocitation_scores(qids, cids, graph)
    texts = {}
    cites = {}
    for k in graph:
        texts[k] = graph[k]["text"]
        cites[k] = graph[k]["relevant"]
    qtexts = []
    for k in qids:
        qtexts.append(queries[k]["text"])
    votes = neighbour_votes(qtexts, qids, cids, texts, cites, k=VOTE_K)
    return queries, cands, views, co, votes, qids, cids


def load_dense(split, qids, cids):
    if DENSE_FILE is None:
        return None
    z = np.load(DENSE_FILE)
    assert list(z[split + "_qids"]) == qids and list(z[split + "_cands"]) == cids
    return z[split]


def check(pred, queries, cands):
    assert set(pred) == set(queries)
    for k in pred:
        assert len(pred[k]) == 7 and len(set(pred[k])) == 7 and k not in pred[k]
        for c in pred[k]:
            assert c in cands


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    train_q, _ = data.load("train", DATA_DIR)

    # dev: graph from train labels only
    dq, dc, views, co, votes, qids, cids = build_signals("dev", train_q)
    S = final_score(views, co, votes, dense=load_dense("dev", qids, cids), w_dense=W_DENSE)
    f, p, r = lexical.micro_f1(lexical.top7(S, qids, cids), dq)
    print(f"dev F1@7 {f:.4f} | P {p:.4f} | R {r:.4f}")

    # test: graph from train and dev labels
    graph = dict(train_q)
    graph.update(dq)
    eq, ec, views, co, votes, qids, cids = build_signals("test", graph)
    S = final_score(views, co, votes, dense=load_dense("test", qids, cids), w_dense=W_DENSE)
    pred = lexical.top7(S, qids, cids)
    check(pred, eq, ec)
    name = f"p7_final_dev{f:.4f}.zip"
    with zipfile.ZipFile(os.path.join(OUT_DIR, name), "w") as z:
        z.writestr("prediction.json", json.dumps(pred))
    print("wrote", name)
