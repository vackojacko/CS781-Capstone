# phase 3 data: CL + IT documents loaded together, each doc keeps its domain
import json

import numpy as np

DATA_DIR = "data"
SEP_ID = 3
MAX_SENT_TOKENS = 128

LABELS = ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute", "Dissent", "PrecedentReliedUpon",
          "PrecedentNotReliedUpon", "PrecedentOverruled", "RulingByLowerCourt", "RatioOfTheDecision",
          "RulingByPresentCourt", "None"]


def load_docs(data_dir, split):
    docs = []
    for domain in ["CL", "IT"]:
        data = json.load(open(data_dir + "/" + domain + "_" + split + ".json"))
        if split == "test":
            # test files are {id: sentences}
            for doc_id in data:
                docs.append({"id": doc_id, "domain": domain, "sents": data[doc_id], "labels": None})
        else:
            for d in data:
                labels = []
                for l in d["labels"]:
                    labels.append(LABELS.index(l))
                docs.append({"id": d["id"], "domain": domain, "sents": d["text"], "labels": labels})
    return docs


def tokenize(tokenizer, docs):
    for d in docs:
        d["toks"] = []
        for s in d["sents"]:
            ids = tokenizer.encode(s)[:MAX_SENT_TOKENS].tolist()
            if len(ids) == 0:
                ids = [SEP_ID]
            d["toks"].append(ids)
    return docs


def make_windows(toks, core, total, a=0, b=None, left=False):
    # sentences i..j get their vectors from this window, lo..i and j..hi are only context
    # left=True fills the context before the core first
    if b is None:
        b = len(toks)
    n = len(toks)
    windows = []
    i = a
    while i < b:
        j = i
        used = 0
        while j < b and (j == i or used + len(toks[j]) <= core):
            used += len(toks[j])
            j += 1
        lo = i
        hi = j
        budget = total - used
        while left and lo > 0 and len(toks[lo - 1]) <= budget:
            lo -= 1
            budget -= len(toks[lo])
        grew = True
        while budget > 0 and grew:
            grew = False
            if lo > 0 and len(toks[lo - 1]) <= budget:
                lo -= 1
                budget -= len(toks[lo])
                grew = True
            if hi < n and len(toks[hi]) <= budget:
                budget -= len(toks[hi])
                hi += 1
                grew = True
        windows.append((lo, i, j, hi))
        i = j
    return windows


def transition_matrix(docs):
    # label bigram counts with add-one smoothing, rows sum to 1
    T = np.ones((len(LABELS), len(LABELS)))
    for d in docs:
        for k in range(len(d["labels"]) - 1):
            T[d["labels"][k], d["labels"][k + 1]] += 1
    return T / T.sum(axis=1, keepdims=True)
