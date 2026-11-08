import json
import os
import re

DATA_DIR = "data"


def clean(t):
    t = t.replace("<NAME>", " ").replace("<ORG>", " ").replace("<CITATION>", " ")
    return re.sub(r"\s+", " ", t).strip().lower()


def load_raw(split, folder=DATA_DIR):
    q = json.load(open(os.path.join(folder, split + ".json")))
    c = json.load(open(os.path.join(folder, split + "_candidates.json")))
    return q, c


def load(split, folder=DATA_DIR):
    q, c = load_raw(split, folder)
    queries = {}
    for k in q:
        relevant = q[k].get("relevant_candidates", [])
        if split == "test":
            relevant = []
        queries[k] = {"text": clean(" ".join(q[k]["text"])), "relevant": relevant}
    cands = {}
    for k in c:
        cands[k] = clean(" ".join(c[k]["text"]))
    return queries, cands


def context_list(sentences, window=1):
    # the sentence with a citation and its neighbours
    out = []
    for i in range(len(sentences)):
        if "<CITATION>" in sentences[i]:
            out.append(" ".join(sentences[max(0, i - window):i + window + 1]))
    return out


def model_text(sentences):
    t = " ".join(sentences).replace("<NAME>", "X").replace("<ORG>", "X").replace("<CITATION>", "citation")
    w = t.split()
    if len(w) > 4000:
        w = w[:3000] + w[-1000:]
    return " ".join(w)
