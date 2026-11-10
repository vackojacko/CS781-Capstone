# the other phase 5 ensembles: python -m experiments.ensembles <name>, or all
# each group is averaged (logits or probabilities), the groups are averaged with equal weight,
# and the per-section thresholds are tuned on dev
import os
import sys

import numpy as np

from ensemble import load_runs
from predict import sigmoid, write_submission
from thresholds import apply, crossfit, per_label

RUNS_DIR = "runs"
OUT_DIR = "submissions"

F1 = "f1_arch4_lorma_attn_asl"
F1B = "f1b_arch4_lorma_attn_bce"
F1B_S1 = "f1b_s1_arch4_lorma_attn_bce"
F1C = "f1c_arch4_lorma_pool_bce"
F3 = "f3_arch4_qlora_attn_bce"
F4 = "f4_arch2_partial_attn_bce"
F4S1 = "f4s1_arch2_partial_attn_bce_s1"
F4S2 = "f4s2_arch2_partial_attn_bce_s2"
F5 = "f5_arch3_adapter_attn_bce"
F7 = "f7_arch2_kd_from_f1"
F7S1 = "f7s1_arch2_kd_from_f1_s1"
F8 = "f8_arch2_full_attn_bce"
G1 = "g1_arch2_partial_e6"
G2 = "g2_arch2_partial_e6_lr1e4"
G3 = "g3_arch2_full_e6"
G4 = "g4_arch2_kd_e6"
G5 = "g5_arch2_partial_e6_lr1e4_s1"
G6 = "g6_arch2_partial_e6_k8"

A4 = [F1, F1B, F1C, F1B_S1, F3]
A4_LORMA = [F1, F1B, F1C, F1B_S1]
A2 = [F4, F4S1, F8, F7]
A2_SIX = [F4, F4S1, F8, F7, F4S2, F7S1]
G_FOUR = [G1, G2, G4, G5]
G_SIX = [G1, G2, G3, G4, G5, G6]
KD3 = [F7, F7S1, G4]

RECIPES = {
    "09_ens_a2x4": ([A2], "prob", 50),
    "10_ens_a4x5": ([A4], "prob", 50),
    "11_ens_a4qlora_a2": ([[F3, F4]], "prob", 50),
    "12_ens_bce_attn6": ([[F1B, F1B_S1, F3, F4, F4S1, F7]], "prob", 50),
    "13_ens_all9_a3": ([A4 + A2 + [F5]], "prob", 50),
    "14_ens_top4": ([[F7, F4, F1B_S1, F1B]], "prob", 50),
    "15_ens_all9": ([A4 + A2], "prob", 50),
    "16_ens_all11": ([A4 + A2_SIX], "prob", 50),
    "17_all9_logit": ([A4 + A2], "logit", 50),
    "18_all9_groups": ([A4, A2], "prob", 50),
    "19_all9_prob_shrink20": ([A4 + A2], "prob", 20),
    "20_all9_prob_shrink0": ([A4 + A2], "prob", 0),
    "21_all11_groups": ([A4, A2_SIX], "prob", 50),
    "23_all13_prob": ([A4 + A2 + G_FOUR], "prob", 50),
    "24_all13_logit": ([A4 + A2 + G_FOUR], "logit", 50),
    "25_a4x5_g_prob": ([A4 + G_FOUR], "prob", 50),
    "39_ens_kd3_logit": ([KD3], "logit", 50),
    "40_ens_a4x5_logit": ([A4], "logit", 50),
    "41_ens_a4_lorma_only_logit": ([A4_LORMA], "logit", 50),
    "42_ens_f2x6_logit": ([[F4, F4S1, F4S2, F8, F7, F7S1]], "logit", 50),
    "43_ens_a4_kd3_prob": ([A4 + KD3], "prob", 50),
    "44_ens_a4_kd3_logit": ([A4 + KD3], "logit", 50),
    "45_ens_g4x_logit": ([G_FOUR], "logit", 50),
    "46_ens_g6x_logit": ([G_SIX], "logit", 50),
    "47_ens_a2_all12_logit": ([A2_SIX + G_SIX], "logit", 50),
    "48_ens_a4_kd_g5_logit": ([A4 + [G4, G5]], "logit", 50),
    "49_ens_a4_all_logit": ([A4 + A2_SIX + G_SIX], "logit", 50),
    "50_ens_a4_g4_kd_prob": ([A4 + G_FOUR + [F7]], "prob", 50),
    "51_ens_a4_g6_prob": ([A4 + G_SIX], "prob", 50),
    "52_ens_a4_g4_kd_logit": ([A4 + G_FOUR + [F7]], "logit", 50),
    "53_ens_a4_g6_logit": ([A4 + G_SIX], "logit", 50),
    "54_ens_a4_g4_minus_asl_logit": ([[F1B, F1C, F1B_S1, F3] + G_FOUR], "logit", 50),
    "55_ens_a4_g4_double_kd_logit": ([A4 + G_FOUR + [G4]], "logit", 50),
    "56_ens_a4_g4_minus_pool_logit": ([[F1, F1B, F1B_S1, F3] + G_FOUR], "logit", 50),
    "57_ens_a4_g4_f7s_logit": ([A4 + G_FOUR + [F7, F7S1]], "logit", 50),
    "58_ens_a4_g4_minus_qlora_logit": ([A4_LORMA + G_FOUR], "logit", 50),
    "59_ens_attn_bce_a4_g4_logit": ([[F1B, F1B_S1, F3] + G_FOUR], "logit", 50),
}


def average(runs, members, how, split):
    if how == "logit":
        z = []
        for m in members:
            z.append(runs[m][f"{split}_k4"])
        return sigmoid(np.mean(z, 0))
    p = []
    for m in members:
        p.append(sigmoid(runs[m][f"{split}_k4"]))
    return np.mean(p, 0)


def combine(runs, groups, how, split):
    parts = []
    for g in groups:
        parts.append(average(runs, g, how, split))
    return np.mean(parts, 0)


def build(runs, name):
    groups, how, shrink = RECIPES[name]
    ref = runs[groups[0][0]]
    y_dev = ref["dev_y"].astype(bool)
    p_dev = combine(runs, groups, how, "dev")
    p_test = combine(runs, groups, how, "test")
    score = crossfit(p_dev, y_dev, shrink=shrink)["per_label"]
    pred = apply(p_test, per_label(p_dev, y_dev, shrink))
    os.makedirs(OUT_DIR, exist_ok=True)
    write_submission(os.path.join(OUT_DIR, name + ".zip"), list(ref["test_ids"]), pred)
    per_case = pred.sum(1).mean()
    print(f"{name}: dev crossfit {score:.4f} | {per_case:.2f} labels per case")


if __name__ == "__main__":
    runs = load_runs(RUNS_DIR)
    if sys.argv[1] == "all":
        for key in RECIPES:
            build(runs, key)
    else:
        build(runs, sys.argv[1])
