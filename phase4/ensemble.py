# best phase 4 submission: average of the dev-standardised hierarchical scores of three arch 4 lorma runs
# (e1, e10, e11, all trained on the last 2 chunks), threshold tuned on dev
import os

import numpy as np

from predict import report, write_submission

RUNS_DIR = "runs"
OUT_PATH = "submission_ens_a4_tail2_hier.zip"
MEMBERS = ["e1_arch4_lorma", "e10_arch4_lorma_r32_e3_s2", "e11_arch4_lorma_s3"]
VARIANT = "hier"


def standardise(scores, variant):
    # dev mean and std of the run, used for both dev and test
    d = scores["dev_" + variant]
    t = scores["test_" + variant]
    mu = d.mean()
    sd = d.std()
    return (d - mu) / sd, (t - mu) / sd


def ensemble(runs, members, variant):
    dev_parts = []
    test_parts = []
    for name in members:
        d, t = standardise(runs[name], variant)
        dev_parts.append(d)
        test_parts.append(t)
    return np.mean(dev_parts, axis=0), np.mean(test_parts, axis=0)


if __name__ == "__main__":
    runs = {}
    for name in MEMBERS:
        runs[name] = np.load(os.path.join(RUNS_DIR, name, "scores.npz"))
    ref = runs[MEMBERS[0]]
    dev_y = ref["dev_y"]
    test_ids = list(ref["test_ids"])
    for name in MEMBERS:
        assert list(runs[name]["dev_ids"]) == list(ref["dev_ids"])
        assert list(runs[name]["test_ids"]) == test_ids
    d, t = ensemble(runs, MEMBERS, VARIANT)
    r = report(d, dev_y)
    pred = t > r["thr"]
    n = write_submission(OUT_PATH, test_ids, pred)
    f1_tuned = r["f1_tuned"]
    thr = r["thr"]
    share = float(pred.mean())
    print(f"dev macro-F1 {f1_tuned:.4f} (thr {thr:.3f}) | {n} test cases, {share:.3f} predicted 1")
