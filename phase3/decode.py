# prediction and decoding of sentence labels (crf marginals, viterbi with counted label transitions, macro F1)
import numpy as np
import torch
from sklearn.metrics import f1_score

from data import LABELS
from rules import dissent_rule

@torch.no_grad()
def predict(model, doc):
    model.eval()
    E, _ = model(doc, 0, len(doc["sents"]))
    E = E.float()
    # float64, in float32 the rows of very long docs dont sum to 1
    probs = model.crf.marginals(E.double()).cpu().numpy()
    probs = probs / probs.sum(axis=1, keepdims=True)
    return dissent_rule(doc, model.crf.viterbi(E)), probs


def transition_viterbi(probs, T, beta):
    # viterbi over sentence probabilities with label transitions counted on the training docs
    lp = np.log(np.clip(probs, 1e-9, 1.0))
    lt = beta * np.log(T)
    score = lp[0].copy()
    back = []
    for i in range(1, len(probs)):
        c = score[:, None] + lt
        back.append(c.argmax(axis=0))
        score = c.max(axis=0) + lp[i]
    y = [int(score.argmax())]
    for b in reversed(back):
        y.append(int(b[y[-1]]))
    return y[::-1]


def macro_f1(gold, pred):
    # same as evaluate_phase3.py
    y_true = []
    y_pred = []
    for g in gold:
        for y in g:
            y_true.append(LABELS[y])
    for p in pred:
        for y in p:
            y_pred.append(LABELS[y])
    return f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
