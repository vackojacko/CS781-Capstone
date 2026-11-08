# tf-idf centrality and headnote propensity scores for each sentence of a judgment
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

WORD = re.compile(r"[a-z]+")
SMOOTH = 10


def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def fit_global_idf(train_docs):
    all_sents = []
    for d in train_docs:
        all_sents += d["sents"]
    return TfidfVectorizer(min_df=2, sublinear_tf=True).fit(all_sents)


def centrality(m):
    # similarity of every sentence to the mean tf-idf vector of the judgment
    c = np.asarray(m.mean(0)).ravel()
    return norm(np.asarray(m @ c).ravel())


def global_centrality(sents, glob_vec):
    return centrality(glob_vec.transform(sents))


def fit_propensity(train_docs):
    # share of judgments with the word whose headnote also has it, smoothed toward the average
    in_doc = Counter()
    in_both = Counter()
    for d in train_docs:
        doc_words = set(WORD.findall(" ".join(d["sents"]).lower()))
        sum_words = set(WORD.findall(d["summary"].lower()))
        for w in doc_words:
            in_doc[w] += 1
            if w in sum_words:
                in_both[w] += 1
    prior = sum(in_both.values()) / max(sum(in_doc.values()), 1)
    propensity = {}
    for w in in_doc:
        propensity[w] = (in_both[w] + SMOOTH * prior) / (in_doc[w] + SMOOTH)
    return propensity, prior


def propensity_scores(sents, propensity, prior, idf):
    # idf weighted mean propensity of the words of each sentence
    out = []
    for s in sents:
        ws = WORD.findall(s.lower())
        if not ws:
            out.append(0.0)
            continue
        total, weight = 0.0, 0.0
        for w in ws:
            v = idf.get(w, 1.0)
            total += v * propensity.get(w, prior)
            weight += v
        out.append(total / weight)
    return norm(np.array(out))


def idf_table(glob_vec):
    return dict(zip(glob_vec.get_feature_names_out(), glob_vec.idf_))
