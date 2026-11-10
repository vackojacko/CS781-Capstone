# one run with one input as a submission: python -m experiments.single_variant <run> <variant>
# variant is first, last, tail2, whole, selected or hier, threshold tuned on dev
import os
import sys

import numpy as np

from predict import report, write_submission

RUNS_DIR = "runs"
OUT_DIR = "submissions"


if __name__ == "__main__":
    name = sys.argv[1]
    variant = sys.argv[2]
    z = np.load(os.path.join(RUNS_DIR, name, "scores.npz"))
    r = report(z["dev_" + variant], z["dev_y"])
    pred = z["test_" + variant] > r["thr"]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + "__" + variant + ".zip")
    n = write_submission(path, list(z["test_ids"]), pred)
    f1_tuned = r["f1_tuned"]
    share = float(pred.mean())
    print(f"{path}: dev macro-F1 {f1_tuned:.4f} | {n} test cases, {share:.3f} predicted 1")
