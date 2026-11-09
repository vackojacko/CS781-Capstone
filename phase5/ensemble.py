import glob
import os

import numpy as np

from predict import sigmoid, write_submission
from thresholds import apply, crossfit, per_label

RUNS_DIR = "runs"
OUT_PATH = "submissions/ens9_a4x5_g4_logit.zip"

A4 = ["f1_arch4_lorma_attn_asl", "f1b_arch4_lorma_attn_bce", "f1c_arch4_lorma_pool_bce",
      "f1b_s1_arch4_lorma_attn_bce", "f3_arch4_qlora_attn_bce"]
A2 = ["g1_arch2_partial_e6", "g2_arch2_partial_e6_lr1e4", "g4_arch2_kd_e6", "g5_arch2_partial_e6_lr1e4_s1"]
MEMBERS = A4 + A2


def load_runs(runs_dir):
    runs = {}
    for f in glob.glob(f"{runs_dir}/**/scores.npz", recursive=True):
        runs[os.path.basename(os.path.dirname(f))] = np.load(f)
    return runs


def average_logits(runs, members, split):
    z = []
    for m in members:
        z.append(runs[m][f"{split}_k4"])
    return sigmoid(np.mean(z, 0))


def ensemble(runs):
    ref = runs[MEMBERS[0]]
    y_dev = ref["dev_y"].astype(bool)
    p_dev = average_logits(runs, MEMBERS, "dev")
    p_test = average_logits(runs, MEMBERS, "test")
    print(f"{len(MEMBERS)}-run ensemble crossfit {crossfit(p_dev, y_dev)['per_label']:.4f}")
    pred = apply(p_test, per_label(p_dev, y_dev))
    return list(ref["test_ids"]), pred


if __name__ == "__main__":
    runs = load_runs(RUNS_DIR)
    ids, pred = ensemble(runs)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    write_submission(OUT_PATH, ids, pred)
    per_case = pred.sum(1).mean()
    print(f"wrote {OUT_PATH}, {per_case:.2f} labels per case")
