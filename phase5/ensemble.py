import glob
import os

import numpy as np

from data import TOKEN_DIR
from knn_votes import gated_blend, knn_votes
from predict import sigmoid, write_submission
from thresholds import apply, crossfit, per_label

RUNS_DIR = "runs"
KNN_FILE = "runs/knn_bpe.npz"
OUT_PATH = "submissions/ens12_knn_bpe_k50_p8_a0.2_g0.8.zip"

A4 = ["f1_arch4_lorma_attn_asl", "f1b_arch4_lorma_attn_bce", "f1b_s1_arch4_lorma_attn_bce", "f3_arch4_qlora_attn_bce"]
A2 = ["g1_arch2_partial_e6", "g2_arch2_partial_e6_lr1e4", "g5_arch2_partial_e6_lr1e4_s1"]
KD = ["g4_arch2_kd_e6", "g7_arch2_kd_e6_s1", "g8_arch2_kd_e6_s2", "g9_arch2_kd_e6_s3", "g10_arch2_kd_e6_s4"]
MEMBERS = A4 + A2 + KD

K_NEIGHBOURS = 50
POWER = 8
BLEND = 0.2
GATE = 0.8
BLEND_GATE = 0.7


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


def fuse(runs, neighbours, y_train):
    ref = runs[MEMBERS[0]]
    y_dev = ref["dev_y"].astype(bool)
    ypool = np.vstack([y_train, y_dev])
    p_dev = average_logits(runs, MEMBERS, "dev")
    p_test = average_logits(runs, MEMBERS, "test")
    v_dev = knn_votes(neighbours["di"], neighbours["ds"], ypool, K_NEIGHBOURS, POWER)
    v_test = knn_votes(neighbours["ti"], neighbours["ts"], ypool, K_NEIGHBOURS, POWER)
    f_dev = gated_blend(p_dev, v_dev, neighbours["ds"][:, 0], BLEND, GATE, BLEND_GATE)
    f_test = gated_blend(p_test, v_test, neighbours["ts"][:, 0], BLEND, GATE, BLEND_GATE)
    base_f1 = crossfit(p_dev, y_dev)["per_label"]
    blend_f1 = crossfit(f_dev, y_dev)["per_label"]
    print(f"12-run ensemble crossfit {base_f1:.4f}")
    print(f"+ gated kNN votes crossfit {blend_f1:.4f}")
    pred = apply(f_test, per_label(f_dev, y_dev))
    return list(ref["test_ids"]), pred


if __name__ == "__main__":
    runs = load_runs(RUNS_DIR)
    neighbours = np.load(KNN_FILE)
    y_train = np.load(f"{TOKEN_DIR}/train.npz")["y"].astype(bool)
    ids, pred = fuse(runs, neighbours, y_train)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    write_submission(OUT_PATH, ids, pred)
    per_case = pred.sum(1).mean()
    print(f"wrote {OUT_PATH}, {per_case:.2f} labels per case")
