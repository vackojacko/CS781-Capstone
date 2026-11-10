import numpy as np

import data
import lexical
from citation_graph import cocitation_scores, neighbour_votes

DATA_DIR = "data"


def merged_contexts(sentences, window=1):
    # all sentences near a citation joined into one text
    keep = set()
    for i in range(len(sentences)):
        if "<CITATION>" in sentences[i]:
            for j in range(max(0, i - window), min(len(sentences), i + window + 1)):
                keep.add(j)
    if len(keep) == 0:
        return " ".join(sentences)
    out = []
    for j in sorted(keep):
        out.append(sentences[j])
    return " ".join(out)


def extra_views(queries, cands, raw, names):
    qids = list(queries)
    cids = list(cands)
    docs = []
    for i in cids:
        docs.append(cands[i])
    qs = []
    for k in qids:
        qs.append(queries[k]["text"])

    views = {}
    if "merged" in names:
        vec = lexical.make_vectorizer((4, 6))
        vec.fit(docs + qs)
        texts = []
        for k in qids:
            texts.append(data.clean(merged_contexts(raw[k]["text"])))
        views["merged"] = (vec.transform(texts) @ vec.transform(docs).T).toarray()
    if "words23" in names:
        vec = lexical.make_vectorizer((2, 3), max_features=500000)
        vec.fit(docs + qs)
        views["words23"] = (vec.transform(qs) @ vec.transform(docs).T).toarray()
    for name in views:
        lexical.remove_self(views[name], qids, cids)
    return views


def load_split(split, graph, extra):
    queries, cands = data.load(split, DATA_DIR)
    raw, _ = data.load_raw(split, DATA_DIR)
    views, qids, cids = lexical.lexical_views(queries, cands, raw)
    views.update(extra_views(queries, cands, raw, extra))
    co = cocitation_scores(qids, cids, graph)
    return queries, cands, views, co, qids, cids


def votes_for(queries, qids, cids, graph, k):
    texts = {}
    cites = {}
    for g in graph:
        texts[g] = graph[g]["text"]
        cites[g] = graph[g]["relevant"]
    qtexts = []
    for q in qids:
        qtexts.append(queries[q]["text"])
    return neighbour_votes(qtexts, qids, cids, texts, cites, k=k)


def load_dense(path, split, qids, cids):
    z = np.load(path)
    assert list(z[split + "_qids"]) == qids and list(z[split + "_cands"]) == cids
    return z[split]
