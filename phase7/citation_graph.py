import numpy as np

from lexical import make_vectorizer


def cocitation_scores(qids, cids, graph):
    # number of graph judgments that cite both q and c
    cited_by = {}
    for k in graph:
        for r in graph[k]["relevant"]:
            if r not in cited_by:
                cited_by[r] = set()
            cited_by[r].add(k)
    S = np.zeros((len(qids), len(cids)), np.float32)
    for a in range(len(qids)):
        k = qids[a]
        mine = cited_by.get(k, set())
        if len(mine) == 0:
            continue
        for j in range(len(cids)):
            c = cids[j]
            if c != k and c in cited_by:
                S[a, j] = len(mine & cited_by[c])
    return S


def neighbour_votes(query_texts, qids, cids, graph_texts, graph_cites, k=5):
    # the k most similar graph judgments vote for the cases they cite
    tids = list(graph_texts)
    texts = []
    for t in tids:
        texts.append(graph_texts[t])
    vec = make_vectorizer((1, 2), max_features=300000)
    T = vec.fit_transform(texts)
    Q = vec.transform(query_texts)
    sim = (Q @ T.T).toarray()
    col = {}
    for j in range(len(cids)):
        col[cids[j]] = j
    S = np.zeros((len(qids), len(cids)), np.float32)
    for a in range(len(qids)):
        q = qids[a]
        order = np.argsort(-sim[a])
        used = 0
        for b in order:
            if tids[b] == q:
                continue
            for c in graph_cites[tids[b]]:
                if c in col and c != q:
                    S[a, col[c]] += sim[a, b]
            used += 1
            if used >= k:
                break
    return S
