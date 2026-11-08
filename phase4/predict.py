# scores from chunk logits, decision threshold tuned on dev, and the submission writer (numpy and scikit-learn)
import json
import math
import zipfile

import numpy as np
from sklearn.metrics import f1_score

PRIOR_SHIFT = math.log(0.59 / 0.41)   # train is 59/41, dev and test are balanced


def margin(logits):
    return logits[..., 1] - logits[..., 0]


def best_threshold(z, y):
    best = -1.0
    thr = 0.0
    for t in np.unique(np.quantile(z, np.linspace(0.05, 0.95, 181))):
        f = f1_score(y, (z > t).astype(int), average="macro")
        if f > best:
            best = f
            thr = float(t)
    return best, thr


def report(z, y):
    f_tuned, thr = best_threshold(z, y)
    return {"f1_0": f1_score(y, (z > 0).astype(int), average="macro"),
            "f1_prior": f1_score(y, (z + PRIOR_SHIFT > 0).astype(int), average="macro"),
            "f1_tuned": f_tuned, "thr": thr}


def first_score(logits):
    return np.array([margin(l[0]) for l in logits])


def tail_score(logits, k=2):
    return np.array([margin(l[-k:]).mean() for l in logits])


def whole_score(logits):
    return np.array([margin(l).mean() for l in logits])


def selected_score(logits, counts, top=2):
    # the two chunks with the most reasoning cue words
    z = []
    for l, c in zip(logits, counts):
        best = np.argsort(-np.asarray(c), kind="stable")[:top]
        z.append(margin(l[best]).mean())
    return np.array(z)


def write_submission(path, ids, preds):
    sub = []
    for i, p in zip(ids, preds):
        sub.append({"id": i, "pred": int(p)})
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("prediction.json", json.dumps(sub))
    with zipfile.ZipFile(path) as z:
        assert z.namelist() == ["prediction.json"]
        assert len(json.loads(z.read("prediction.json"))) == len(set(ids))
    return len(sub)
