import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import data

TOKEN_PATTERN = r"[a-z0-9]+"
NEG = -1e9


def make_vectorizer(ngram, max_features=None):
    return TfidfVectorizer(ngram_range=ngram, min_df=2, sublinear_tf=True, max_features=max_features,
                           token_pattern=TOKEN_PATTERN, dtype=np.float32)


def remove_self(S, qids, cids):
    # every query is also in its own pool, so it must never be picked
    for a in range(len(qids)):
        if qids[a] in cids:
            S[a, cids.index(qids[a])] = NEG
    return S


def candidate_tfidf(queries, cands, ngram=(4, 6)):
    # fitted on the candidates only, used for hard negatives
    qids = list(queries)
    cids = list(cands)
    docs = []
    for i in cids:
        docs.append(cands[i])
    qs = []
    for k in qids:
        qs.append(queries[k]["text"])
    vec = make_vectorizer(ngram)
    D = vec.fit_transform(docs)
    Q = vec.transform(qs)
    S = (Q @ D.T).toarray()
    remove_self(S, qids, cids)
    return S, qids, cids


def lexical_views(queries, cands, raw):
    # full query, best citation context and word 1-2 grams
    qids = list(queries)
    cids = list(cands)
    docs = []
    for i in cids:
        docs.append(cands[i])
    qs = []
    for k in qids:
        qs.append(queries[k]["text"])

    vec = make_vectorizer((4, 6))
    vec.fit(docs + qs)
    D = vec.transform(docs)
    full = (vec.transform(qs) @ D.T).toarray()
    best = np.zeros_like(full)
    for a in range(len(qids)):
        ctx = []
        for t in data.context_list(raw[qids[a]]["text"], 1):
            ctx.append(data.clean(t))
        if len(ctx) > 0:
            best[a] = (vec.transform(ctx) @ D.T).toarray().max(0)

    vec2 = make_vectorizer((1, 2), max_features=300000)
    vec2.fit(docs + qs)
    D2 = vec2.transform(docs)
    words = (vec2.transform(qs) @ D2.T).toarray()

    views = {"full": full, "best": best, "words": words}
    for name in views:
        remove_self(views[name], qids, cids)
    return views, qids, cids


def zscore(S):
    M = S.copy()
    M[M < -1e8] = np.nan
    mu = np.nanmean(M, 1, keepdims=True)
    sd = np.nanstd(M, 1, keepdims=True) + 1e-9
    out = (S - mu) / sd
    out[S < -1e8] = NEG
    return out


def csls(S, k=4):
    # 2 sim(q, c) - r_c - r_q
    M = S.copy()
    M[M < -1e8] = NEG
    col_top = np.sort(M, axis=0)[::-1][:k]
    r_c = col_top.mean(0, keepdims=True)
    row_top = np.sort(M, axis=1)[:, ::-1][:, :k]
    r_q = row_top.mean(1, keepdims=True)
    out = 2 * S - r_c - r_q
    out[S < -1e8] = NEG
    return out


def top7(S, qids, cids):
    out = {}
    for a in range(len(qids)):
        order = np.argsort(-S[a])[:7]
        picks = []
        for j in order:
            picks.append(cids[j])
        out[qids[a]] = picks
    return out


def micro_f1(pred, queries):
    hits = 0
    n_pred = 0
    n_gold = 0
    for k in queries:
        rel = set(queries[k]["relevant"])
        hits += len(rel & set(pred[k]))
        n_pred += len(pred[k])
        n_gold += len(rel)
    p = hits / n_pred
    r = hits / n_gold
    return 2 * p * r / (p + r), p, r
