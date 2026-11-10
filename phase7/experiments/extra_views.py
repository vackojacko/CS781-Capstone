import json
import os
import sys
import zipfile

import data
import lexical
from experiments.views import DATA_DIR, load_split, votes_for
from predict import check
from rerank import final_score, view_score

OUT_DIR = "submissions"

# submission number: weights of the extra views and the votes
SETTINGS = {
    5: {"w_words": 0.5, "w_merged": 0.5, "w_words23": 0.0, "w_votes": 4.0, "vote_k": 5},
    7: {"w_words": 0.5, "w_merged": 1.0, "w_words23": 0.0, "w_votes": 6.0, "vote_k": 10},
    12: {"w_words": 0.5, "w_merged": 0.0, "w_words23": 0.5, "w_votes": 4.0, "vote_k": 5},
}


def score(cfg, split, graph):
    extra = []
    if cfg["w_merged"] > 0:
        extra.append("merged")
    if cfg["w_words23"] > 0:
        extra.append("words23")
    queries, cands, views, co, qids, cids = load_split(split, graph, extra)
    votes = votes_for(queries, qids, cids, graph, cfg["vote_k"])
    S = final_score(views, co, votes, w_words=cfg["w_words"], w_votes=cfg["w_votes"])
    for name in extra:
        S = S + cfg["w_" + name] * view_score(views[name])
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
