# the other phase 4 ensembles: python -m experiments.ensembles <name>, or all
# every member score is standardised with its dev mean and std, then averaged (or the median is taken),
# and the threshold is the one with the best dev macro-F1
import os
import sys

import numpy as np

from ensemble import standardise
from predict import report, write_submission

RUNS_DIR = "runs"
OUT_DIR = "submissions"

E1 = "e1_arch4_lorma"
E2 = "e2_arch2_unfreeze"
E3 = "e3_arch3_adapter"
E4 = "e4_arch1_full"
E5 = "e5_arch2_kd"
E6 = "e6_arch4_lorma_tail3_s1"
E7 = "e7_arch2_unfreeze_tail3_s1"
E10 = "e10_arch4_lorma_r32_e3_s2"
E11 = "e11_arch4_lorma_s3"
E12 = "e12_arch4_lorma_kd"
E13 = "e13_arch4_lorma_tail3_s4"
E14 = "e14_arch4_lorma_tail3_s5"
NEW_SEEDS = ["e15_arch4_lorma_s6", "e16_arch4_lorma_s7", "e17_arch4_lorma_s8", "e18_arch4_lorma_s9",
             "e19_arch4_lorma_s10", "e20_arch4_lorma_s11"]

A4_SIX = [E1, E6, E10, E11, E13, E14]
A4_TOP4_R2 = [E1, E6, E10, E14]
A4_TAIL3 = [E6, E13, E14]
A4_TAIL2 = [E1, E10, E11] + NEW_SEEDS
A4_ALL = A4_TAIL2 + A4_TAIL3
# arch 4 runs sorted by dev macro-F1 of the hier input
A4_BY_DEV = ["e15_arch4_lorma_s6", "e19_arch4_lorma_s10", E10, E14, E6, "e17_arch4_lorma_s8", E1,
             "e16_arch4_lorma_s7", E11, E13, "e18_arch4_lorma_s9", "e20_arch4_lorma_s11"]


def inputs(runs, variants):
    parts = []
    for r in runs:
        for v in variants:
            parts.append((r, v))
    return parts


def hier(runs):
    return inputs(runs, ["hier"])


def tail2_hier(runs):
    return inputs(runs, ["tail2", "hier"])


RECIPES = {
    "r1_e1_e6_tail2_hier": (tail2_hier([E1, E6]), "mean"),
    "r1_e1_e2_e3_e4_e6_hier": (hier([E1, E2, E3, E4, E6]), "mean"),
    "r1_e2_e6_e7_hier": (hier([E2, E6, E7]), "mean"),
    "r1_e1_e4_e6_hier": (hier([E1, E4, E6]), "mean"),
    "r1_e1_e2_e6_tail2_hier": (tail2_hier([E1, E2, E6]), "mean"),
    "r1_e1_e2_hier": (hier([E1, E2]), "mean"),
    "r1_e1_e6_e7_tail2_hier": (tail2_hier([E1, E6, E7]), "mean"),
    "r1_e1_e2_e6_e7_tail2_hier": (tail2_hier([E1, E2, E6, E7]), "mean"),
    "r1_e6_e7_hier": (hier([E6, E7]), "mean"),
    "r1_e1_e7_hier": (hier([E1, E7]), "mean"),
    "r1_e2_e6_tail2_hier": (tail2_hier([E2, E6]), "mean"),
    "r1_e2_e6_hier": (hier([E2, E6]), "mean"),
    "r1_e1_e2_e6_e7_hier": (hier([E1, E2, E6, E7]), "mean"),
    "r1_e1_e6_e7_hier": (hier([E1, E6, E7]), "mean"),
    "r1_e1_e6_hier": (hier([E1, E6]), "mean"),
    "r1_e1_e2_e6_hier": (hier([E1, E2, E6]), "mean"),
    "r2_a4_tail3_hier": (hier(A4_TAIL3), "mean"),
    "r2_a4x6_kd_hier": (hier(A4_SIX + [E12]), "mean"),
    "r2_a4x6_kd_a2_hier": (hier(A4_SIX + [E12, E2]), "mean"),
    "r2_a4x6_a2_hier": (hier(A4_SIX + [E2]), "mean"),
    "r2_a4x6_hier": (hier(A4_SIX), "mean"),
    "r2_a4x6_hier_whole": (inputs(A4_SIX, ["hier", "whole"]), "mean"),
    "r2_a4_top4_hier": (hier(A4_TOP4_R2), "mean"),
    "r2_a4_top4_hier_whole": (inputs(A4_TOP4_R2, ["hier", "whole"]), "mean"),
    "r3_all_a4_hier_tail2": (inputs(A4_ALL, ["hier", "tail2"]), "mean"),
    "r3_new_seeds_hier": (hier(NEW_SEEDS), "mean"),
    "r3_all_a4_a2x2_kd_hier": (hier(A4_ALL + [E2, E7, E5, E12]), "mean"),
    "r3_all_a4_allinputs": (inputs(A4_ALL, ["hier", "whole", "tail2", "last"]), "mean"),
    "r3_tail2_hier_tail2": (inputs(A4_TAIL2, ["hier", "tail2"]), "mean"),
    "r3_all_a4_hier_x2_tail2": (hier(A4_ALL) + hier(A4_ALL) + inputs(A4_ALL, ["tail2"]), "mean"),
    "r3_all_a4_a2_hier": (hier(A4_ALL + [E2]), "mean"),
    "r3_tail2_hier": (hier(A4_TAIL2), "mean"),
    "r3_top10_hier": (hier(A4_BY_DEV[:10]), "mean"),
    "r3_all_a4_hier_whole": (inputs(A4_ALL, ["hier", "whole"]), "mean"),
    "r3_all_a4_hier": (hier(A4_ALL), "mean"),
    "r3_top8_hier": (hier(A4_BY_DEV[:8]), "mean"),
    "r3_all_a4_kd_hier": (hier(A4_ALL + [E12]), "mean"),
    "r3_all_a4_hier_whole_median": (inputs(A4_ALL, ["hier", "whole"]), "median"),
    "r3_all_a4_hier_median": (hier(A4_ALL), "median"),
    "r3_new_seeds_hier_whole": (inputs(NEW_SEEDS, ["hier", "whole"]), "mean"),
    "r3_tail2_hier_whole": (inputs(A4_TAIL2, ["hier", "whole"]), "mean"),
    "r3_tail2_hier_median": (hier(A4_TAIL2), "median"),
    "r3_top6_hier": (hier(A4_BY_DEV[:6]), "mean"),
    "r3_top4_hier": (hier(A4_BY_DEV[:4]), "mean"),
    # members of the 0.8154 ensemble and of r3_top6_hier together
    "final_union_hier": (hier([E1, E6, E10, E11, E14, "e15_arch4_lorma_s6", "e17_arch4_lorma_s8",
                               "e19_arch4_lorma_s10"]), "mean"),
}


def load_runs(parts):
    runs = {}
    for name, v in parts:
        if name not in runs:
            runs[name] = np.load(os.path.join(RUNS_DIR, name, "scores.npz"))
    return runs


def build(name):
    parts, how = RECIPES[name]
    runs = load_runs(parts)
    ref = runs[parts[0][0]]
    test_ids = list(ref["test_ids"])
    for r in runs:
        assert list(runs[r]["dev_ids"]) == list(ref["dev_ids"])
        assert list(runs[r]["test_ids"]) == test_ids
    dev_parts = []
    test_parts = []
    for r, v in parts:
        d, t = standardise(runs[r], v)
        dev_parts.append(d)
        test_parts.append(t)
    if how == "median":
        d = np.median(dev_parts, axis=0)
        t = np.median(test_parts, axis=0)
    else:
        d = np.mean(dev_parts, axis=0)
        t = np.mean(test_parts, axis=0)
    r = report(d, ref["dev_y"])
    pred = t > r["thr"]
    os.makedirs(OUT_DIR, exist_ok=True)
    n = write_submission(os.path.join(OUT_DIR, name + ".zip"), test_ids, pred)
    f1_tuned = r["f1_tuned"]
    share = float(pred.mean())
    print(f"{name}: {len(parts)} scores, dev macro-F1 {f1_tuned:.4f} | {n} test cases, {share:.3f} predicted 1")


if __name__ == "__main__":
    if sys.argv[1] == "all":
        for key in RECIPES:
            build(key)
    else:
        build(sys.argv[1])
