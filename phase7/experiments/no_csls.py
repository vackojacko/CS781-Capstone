import json
import os
import sys
import zipfile

import numpy as np

import data
import lexical
from experiments.views import DATA_DIR, load_dense, load_split
from lexical import zscore
from predict import check

OUT_DIR = "submissions"

# submission number: early recipes with z-scores only, no csls and no votes
SETTINGS = {
    3: {"w_best": 0.5, "w_merged": 0.5, "w_dense": 0.5,
        "dense": ["dense_runs/d3_arch3_partial/scores.npz", "dense_runs/d4_arch2_ia3/scores.npz"]},
    4: {"w_best": 1.0, "w_merged": 0.0, "w_dense": 0.0, "dense": []},
}


def score(cfg, split, graph):
    extra = []
    if cfg["w_merged"] > 0:
        extra.append("merged")
    queries, cands, views, co, qids, cids = load_split(split, graph, extra)
    S = zscore(views["full"]) + np.log1p(co) + cfg["w_best"] * zscore(views["best"])
    if cfg["w_merged"] > 0:
        S = S + cfg["w_merged"] * zscore(views["merged"])
    if len(cfg["dense"]) > 0:
        avg = np.zeros_like(S)
        for path in cfg["dense"]:
            avg = avg + zscore(load_dense(path, split, qids, cids))
        S = S + cfg["w_dense"] * avg / len(cfg["dense"])
    return S, queries, cands, qids, cids


if __name__ == "__main__":
    sub = int(sys.argv[1])
    cfg = SETTINGS[sub]
    print("submission", sub, cfg)
    os.makedirs(OUT_DIR, exist_ok=True)
    train_q, _ = data.load("train", DATA_DIR)

    S, dq, dc, qids, cids = score(cfg, "dev", train_q)
    f, p, r = lexical.micro_f1(lexical.top7(S, qids, cids), dq)
    print(f"dev F1@7 {f:.4f} | P {p:.4f} | R {r:.4f}")

    graph = dict(train_q)
    graph.update(dq)
    S, eq, ec, qids, cids = score(cfg, "test", graph)
    pred = lexical.top7(S, qids, cids)
    check(pred, eq, ec)
    name = f"p7_sub{sub}_dev{f:.4f}.zip"
    with zipfile.ZipFile(os.path.join(OUT_DIR, name), "w") as z:
        z.writestr("prediction.json", json.dumps(pred))
    print("wrote", name)
