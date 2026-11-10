# submission 28: python -m experiments.greedy
# greedy ensemble selection with replacement (Caruana et al. 2004) over 17 runs, logit average
import os

import numpy as np

from ensemble import load_runs
from predict import sigmoid, write_submission
from thresholds import apply, crossfit, macro_f1, per_label

RUNS_DIR = "runs"
OUT_PATH = "submissions/28_greedy_selection.zip"
ROUNDS = 25

POOL = ["f1_arch4_lorma_attn_asl", "f1b_arch4_lorma_attn_bce", "f1c_arch4_lorma_pool_bce",
        "f1b_s1_arch4_lorma_attn_bce", "f3_arch4_qlora_attn_bce", "f4_arch2_partial_attn_bce",
        "f4s1_arch2_partial_attn_bce_s1", "f4s2_arch2_partial_attn_bce_s2", "f8_arch2_full_attn_bce",
        "f7_arch2_kd_from_f1", "f7s1_arch2_kd_from_f1_s1", "g1_arch2_partial_e6", "g2_arch2_partial_e6_lr1e4",
        "g3_arch2_full_e6", "g4_arch2_kd_e6", "g5_arch2_partial_e6_lr1e4_s1", "g6_arch2_partial_e6_k8"]


def mean_logits(runs, members, split):
    z = []
    for m in members:
        z.append(runs[m][f"{split}_k4"])
    return np.mean(z, 0)


def half_score(p, y, fit, held):
    # thresholds from one half of dev, macro-F1 on the other
    thr = per_label(p[fit], y[fit])
    return macro_f1(y[held], apply(p[held], thr))


def select(runs, y):
    a = np.arange(len(y) // 2)
    b = np.arange(len(y) // 2, len(y))
    chosen = []
    best = -1.0
    for r in range(ROUNDS):
        pick = None
        pick_s = -1.0
        for m in POOL:
            p = sigmoid(mean_logits(runs, chosen + [m], "dev"))
            s = half_score(p, y, a, b) + half_score(p, y, b, a)
            if s > pick_s:
                pick = m
                pick_s = s
        if pick_s <= best and len(chosen) >= 3:
            break
        chosen.append(pick)
        best = max(best, pick_s)
        print(f"round {r + 1}: add {pick}, score {pick_s / 2:.4f}")
    return chosen


if __name__ == "__main__":
    runs = load_runs(RUNS_DIR)
    ref = runs[POOL[0]]
    y = ref["dev_y"].astype(bool)
    members = select(runs, y)
    p_dev = sigmoid(mean_logits(runs, members, "dev"))
    p_test = sigmoid(mean_logits(runs, members, "test"))
    pred = apply(p_test, per_label(p_dev, y))
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    write_submission(OUT_PATH, list(ref["test_ids"]), pred)
    score = crossfit(p_dev, y)["per_label"]
    print(f"{len(members)} picks, dev crossfit {score:.4f}, wrote {OUT_PATH}")
