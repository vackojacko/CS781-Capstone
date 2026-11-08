# best submission (Codabench 0.6345): weighted average of three runs, decoded with counted label transitions,
# then the dissent rule
import json
import zipfile

import numpy as np

from data import DATA_DIR, LABELS, load_docs, transition_matrix
from decode import transition_viterbi
from rules import dissent_rule

RUNS_DIR = "runs"
SUBMISSION_PATH = "submission_phase3.zip"

# sentence probabilities of every member, {} is the split (dev / test)
RUN_FILES = {
    "rr1": RUNS_DIR + "/rr1_{}_probs.npz",  # arch4 + lora
    "x2": RUNS_DIR + "/x2_{}_probs.npz",  # arch4 (longer adaptation) + lora
    "kdov": RUNS_DIR + "/arch2_kd_{}_probs_centered.npz",  # distilled arch2, overlapping windows at prediction
}
WEIGHTS = {"kdov": 0.6, "rr1": 0.2, "x2": 0.2}
BETA = 0.3


def save_probs(path, docs, probs):
    arrays = {}
    for i in range(len(docs)):
        arrays["doc" + str(i)] = probs[i].astype(np.float32)
    np.savez_compressed(path, ids=json.dumps([d["id"] for d in docs]), **arrays)


def load_probs(path):
    z = np.load(path)
    ids = json.loads(str(z["ids"]))
    out = {}
    for i in range(len(ids)):
        out[ids[i]] = z["doc" + str(i)].astype(np.float32)
    return out


def write_submission(docs, preds, path):
    out = {}
    for i in range(len(docs)):
        assert len(preds[i]) == len(docs[i]["sents"])
        names = []
        for y in preds[i]:
            names.append(LABELS[y])
        out[docs[i]["id"]] = names
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("prediction.json", json.dumps(out))


def load_member(name, split, docs):
    p = load_probs(RUN_FILES[name].format(split))
    out = []
    for d in docs:
        out.append(np.asarray(p[d["id"]], dtype=np.float64))
    return out


def ensemble(P, weights, i):
    # weighted average of the members' (row-normalised) probabilities for document i
    x = 0
    for k in weights:
        x = x + weights[k] * P[k][i] / P[k][i].sum(axis=1, keepdims=True)
    return x / x.sum(axis=1, keepdims=True)


def best_submission(train_docs, docs, split="test"):
    T = transition_matrix(train_docs)
    P = {}
    for name in RUN_FILES:
        P[name] = load_member(name, split, docs)
    preds = []
    for i in range(len(docs)):
        x = ensemble(P, WEIGHTS, i)
        preds.append(dissent_rule(docs[i], transition_viterbi(x, T, BETA)))
    return preds


if __name__ == "__main__":
    train_docs = load_docs(DATA_DIR, "train")
    test_docs = load_docs(DATA_DIR, "test")
    preds = best_submission(train_docs, test_docs)
    write_submission(test_docs, preds, SUBMISSION_PATH)
    print("wrote", SUBMISSION_PATH)
