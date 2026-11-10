# submission 27: python -m experiments.group_weights
# logits of the arch 4 group and of the arch 2 group averaged with a weight tuned on dev
import os

import numpy as np

from ensemble import load_runs
from predict import sigmoid, write_submission
from thresholds import apply, crossfit, macro_f1, per_label

RUNS_DIR = "runs"
OUT_PATH = "submissions/27_group_weights.zip"

A4 = ["f1_arch4_lorma_attn_asl", "f1b_arch4_lorma_attn_bce", "f1c_arch4_lorma_pool_bce",
      "f1b_s1_arch4_lorma_attn_bce", "f3_arch4_qlora_attn_bce"]
A2 = ["f4_arch2_partial_attn_bce", "f4s1_arch2_partial_attn_bce_s1", "f4s2_arch2_partial_attn_bce_s2",
      "f8_arch2_full_attn_bce", "f7_arch2_kd_from_f1", "f7s1_arch2_kd_from_f1_s1", "g1_arch2_partial_e6",
      "g2_arch2_partial_e6_lr1e4", "g3_arch2_full_e6", "g4_arch2_kd_e6", "g5_arch2_partial_e6_lr1e4_s1",
      "g6_arch2_partial_e6_k8"]
WEIGHTS = np.linspace(0.2, 0.8, 13)


def mean_logits(runs, members, split):
    z = []
    for m in members:
        z.append(runs[m][f"{split}_k4"])
    return np.mean(z, 0)


def mix(runs, w, split):
    return w * mean_logits(runs, A4, split) + (1 - w) * mean_logits(runs, A2, split)


if __name__ == "__main__":
    runs = load_runs(RUNS_DIR)
    ref = runs[A2[0]]
    y = ref["dev_y"].astype(bool)
    best = -1.0
    best_w = None
    for w in WEIGHTS:
        p = sigmoid(mix(runs, w, "dev"))
        f = macro_f1(y, apply(p, per_label(p, y)))
        print(f"arch 4 weight {w:.2f}: dev macro-F1 {f:.4f}")
        if f > best:
            best = f
            best_w = w
    p_dev = sigmoid(mix(runs, best_w, "dev"))
    p_test = sigmoid(mix(runs, best_w, "test"))
    pred = apply(p_test, per_label(p_dev, y))
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    write_submission(OUT_PATH, list(ref["test_ids"]), pred)
    score = crossfit(p_dev, y)["per_label"]
    print(f"arch 4 weight {best_w:.2f}, dev crossfit {score:.4f}, wrote {OUT_PATH}")
