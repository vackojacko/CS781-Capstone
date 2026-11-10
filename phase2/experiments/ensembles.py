# phase 2 ensembles: python -m experiments.ensembles <name>, or all
# the saved token probabilities of the runs are averaged with weights, then viterbi, spreading in the document
# and boundary repair
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from decode import decode_doc, propagate, repair_spans
from preprocess import DATA_DIR, N_TAGS, TAG_ID, encode_docs, load_split, load_tokenizer
from train import OUT_DIR, write_submission

SUB_DIR = "submissions"
MIN_LEN = 6
MIN_CONF = 0.5

A4 = "arch4_full_h128"
A4_S1 = "arch4_full_h128_s1"
A2_H256 = "arch2_full_h256_s1"

# per-label additions to the B, I, E and S tag scores, chosen on cross validation
BIAS_2 = {"APP": 0.5, "RESP": 0.5, "WIT": 0.5, "AUTH": 0.5, "COURT": -1.0, "DATE": -2.0}
BIAS_3 = {"APP": 0.5, "RESP": 0.5, "A.COUNSEL": -0.5, "WIT": 0.5, "AUTH": 0.5, "COURT": -0.5, "PREC": 0.5,
          "DATE": -1.5}

TWO = [(2, A4), (1, A2_H256)]
THREE = [(2, A4), (1, A4_S1), (1, A2_H256)]


def recipe(groups, bias=None, ci_app=False, repair=True):
    if bias is None:
        bias = {}
    return dict(groups=groups, bias=bias, ci_app=ci_app, repair=repair)


# groups: (group weight, [(run weight, run name), ...]); each group is averaged first
RECIPES = {
    "ens_a4_a2_a2h256": recipe([(1, TWO)], repair=False),
    "ens_a4x2_a2h256_repair": recipe([(1, TWO)]),
    "ens_bias": recipe([(1, TWO)], BIAS_2),
    "ens_bias_ciapp": recipe([(1, TWO)], BIAS_2, ci_app=True),
    "ens3_bias": recipe([(1, THREE)], BIAS_3, ci_app=True),
}


def load_logprobs(run):
    z = np.load(os.path.join(OUT_DIR, run, "test_logprobs.npz"))
    out = {}
    for k in z.files:
        out[k] = z[k].astype(np.float32)
    return out


def average(parts):
    # weighted mean of the probabilities, back to log probabilities
    total = {}
    wsum = 0.0
    for w, logp in parts:
        wsum += w
        for d in logp:
            total[d] = total.get(d, 0) + w * np.exp(logp[d])
    out = {}
    for d in total:
        out[d] = np.log(np.clip(total[d] / wsum, 1e-9, 1))
    return out


def mix(groups, cache):
    group_logp = []
    for gw, members in groups:
        parts = []
        for w, run in members:
            if run not in cache:
                cache[run] = load_logprobs(run)
            parts.append((float(w), cache[run]))
        group_logp.append((float(gw), average(parts)))
    if len(group_logp) == 1:
        return group_logp[0][1]
    return average(group_logp)


def propagate_ci(text, spans, min_len, min_conf, ci_labels):
    # decode.propagate, with case-insensitive matching for the labels in ci_labels
    votes = defaultdict(Counter)
    count = Counter()
    for s, e, lab, c in spans:
        votes[text[s:e]][lab] += c
        count[text[s:e]] += 1
    label_of = {}
    for t, v in votes.items():
        label_of[t] = v.most_common(1)[0][0]
    out = []
    for s, e, lab, c in spans:
        out.append((s, e, label_of[text[s:e]], c))
    taken = np.zeros(len(text) + 1, dtype=bool)
    for sp in out:
        taken[sp[0]:sp[1]] = True
    for t in sorted(label_of, key=len, reverse=True):
        if len(t) < min_len or sum(votes[t].values()) / count[t] < min_conf:
            continue
        flags = 0
        if label_of[t] in ci_labels:
            flags = re.IGNORECASE
        for m in re.finditer(r"(?<!\w)" + re.escape(t) + r"(?!\w)", text, flags):
            if not taken[m.start():m.end()].any():
                out.append((m.start(), m.end(), label_of[t], 0.0))
                taken[m.start():m.end()] = True
    return sorted(out)


def bias_vector(bias):
    b = np.zeros(N_TAGS, dtype=np.float32)
    for lab, v in bias.items():
        for p in "BIES":
            b[TAG_ID[p + "-" + lab]] = v
    return b


def predict(r, docs, logp):
    b = bias_vector(r["bias"])
    pred = {}
    for d, doc in docs.items():
        text = doc["text"]
        spans = decode_doc(doc, logp[d] + b)
        if r["ci_app"]:
            spans = propagate_ci(text, spans, MIN_LEN, MIN_CONF, ("APP",))
        else:
            spans = propagate(text, spans, MIN_LEN, MIN_CONF, True)
        if r["repair"]:
            spans = repair_spans(text, spans)
        pred[d] = spans
    return pred


def main(names):
    texts, _ = load_split(DATA_DIR, ["fold3"])
    docs = encode_docs(load_tokenizer(), texts)
    os.makedirs(SUB_DIR, exist_ok=True)
    cache = {}
    for name in names:
        r = RECIPES[name]
        pred = predict(r, docs, mix(r["groups"], cache))
        path = os.path.join(SUB_DIR, "submission_" + name + ".zip")
        n = write_submission(pred, texts, path)
        print(name, "|", n, "spans |", path, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "all":
        main(list(RECIPES))
    else:
        main(sys.argv[1:])
