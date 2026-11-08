import json
import sys
import zipfile

import numpy as np

from thresholds import apply, per_label


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def write_submission(path, ids, pred):
    sub = []
    for i in range(len(ids)):
        labels = [int(j) for j in np.flatnonzero(pred[i])]
        assert len(labels) > 0
        sub.append({"id": ids[i], "pred": labels})
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("prediction.json", json.dumps(sub))
    with zipfile.ZipFile(path) as z:
        assert z.namelist() == ["prediction.json"]
        assert len(json.loads(z.read("prediction.json"))) == len(set(ids))
    return len(sub)


def submission_from_scores(scores_path, out_path, k=4):
    # thresholds tuned per label on dev, applied to test
    z = np.load(scores_path)
    dev_y = z["dev_y"].astype(bool)
    thr = per_label(sigmoid(z[f"dev_k{k}"]), dev_y)
    pred = apply(sigmoid(z[f"test_k{k}"]), thr)
    return write_submission(out_path, list(z["test_ids"]), pred)


if __name__ == "__main__":
    n = submission_from_scores(sys.argv[1], sys.argv[2])
    print("wrote", n, "cases to", sys.argv[2])
