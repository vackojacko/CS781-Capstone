# the first centrality submissions (#01, #13 to #16): sentences taken straight from the raw judgment text
import json
import os
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import blend as B
import centrality as C
from data import TEST_JSON, TRAIN_JSON
from predict import write_prediction

OUT_DIR = "work/experiments"
SAMPLE_DOCS = 1000
SAMPLE_SEED = 1

# submission: (centrality, words)
RECIPES = {
    "01": ("local", 600),
    "13": ("local", 500),
    "14": ("local", 700),
    "15": ("global", 600),
    "16": ("local+global", 600),
}


def body(doc):
    # drop the case header when "delivered by ..." comes early in the judgment
    m = re.search(r"delivered by[^.]{0,80}\.", doc)
    if m and m.end() < len(doc) * 0.3:
        return doc[m.end():].strip()
    return doc


def split(text):
    out = []
    for s in re.split(r"(?<=[.?!])\s+", text):
        if len(s.split()) > 3:
            out.append(s)
    return out


def fit_sample_idf(train):
    rng = np.random.RandomState(SAMPLE_SEED)
    all_sents = []
    for i in rng.choice(len(train), SAMPLE_DOCS, replace=False):
        all_sents += split(body(train[i]["document"]))
    return TfidfVectorizer(min_df=2, sublinear_tf=True).fit(all_sents)


def summary(doc, kind, words, glob_vec):
    sents = split(body(doc))
    if kind == "local":
        sc = C.centrality(TfidfVectorizer().fit_transform(sents))
    elif kind == "global":
        sc = C.centrality(glob_vec.transform(sents))
    else:
        sc = C.centrality(TfidfVectorizer().fit_transform(sents)) + C.centrality(glob_vec.transform(sents))
    return B.select(sents, sc, words)


def main():
    names = sys.argv[1:]
    if not names or names == ["all"]:
        names = list(RECIPES)
    os.makedirs(OUT_DIR, exist_ok=True)
    train = json.load(open(TRAIN_JSON))
    test = json.load(open(TEST_JSON))
    glob_vec = fit_sample_idf(train)
    for name in names:
        kind, words = RECIPES[name]
        preds = []
        for case in test:
            preds.append({"id": case["id"], "summary": summary(case["document"], kind, words, glob_vec)})
        path = os.path.join(OUT_DIR, "submission_" + name + ".zip")
        write_prediction(preds, path)
        print(name, kind, words, "->", path)


if __name__ == "__main__":
    main()
