# final sentence score: centrality + headnote propensity + mean of the phase 1 salience models
import numpy as np

import centrality as C
from data import header_end

W_MODEL = 0.52
W_PROPENSITY = 0.65
WORDS = 525
MIN_WORDS = 6
MAX_WORDS = 90


def select(sents, scores, max_words):
    # highest scores first until max_words, then back in document order
    order = np.argsort(-np.asarray(scores))
    keep = []
    used = 0
    for i in order:
        keep.append(int(i))
        used += len(sents[i].split())
        if used >= max_words:
            break
    keep.sort()
    out = []
    for i in keep:
        out.append(sents[i])
    return " ".join(out)


def select_filtered(sents, scores):
    # same as select but skips fragments and very long sentences
    order = np.argsort(-np.asarray(scores))
    keep, used = [], 0
    for i in order:
        n = len(sents[i].split())
        if n < MIN_WORDS or n > MAX_WORDS:
            continue
        keep.append(int(i))
        used += n
        if used >= WORDS:
            break
    keep.sort()
    out = []
    for i in keep:
        out.append(sents[i])
    return " ".join(out)


def model_mean(model_scores, k, n):
    m = np.zeros(n)
    for name in model_scores:
        m = m + np.asarray(model_scores[name][k])
    return m / len(model_scores)


def blend_scores(sents, model_avg, glob_vec, propensity, prior, idf):
    c = C.global_centrality(sents, glob_vec)
    if W_PROPENSITY > 0:
        c = (1 - W_PROPENSITY) * c + W_PROPENSITY * C.propensity_scores(sents, propensity, prior, idf)
    sc = (1 - W_MODEL) * c + W_MODEL * model_avg
    # never pick the case header
    sc[:header_end(sents)] = -1.0
    return sc
