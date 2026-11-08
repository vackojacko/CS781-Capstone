import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN_DIR = "data/lsi_tokens"
OUT_FILE = "runs/knn_bpe.npz"
K = 50


def load_docs(path):
    # each case becomes a string of its token ids, so tf-idf works on my bpe tokens
    z = np.load(path)
    ends = np.cumsum(z["lens"])
    docs = []
    for i in range(len(ends)):
        ids = z["ids"][ends[i] - z["lens"][i]:ends[i]]
        docs.append(" ".join([str(t) for t in ids]))
    return docs


def top_k(Xq, Xpool, self_offset=None):
    idx = np.zeros((Xq.shape[0], K), np.int64)
    sim = np.zeros((Xq.shape[0], K), np.float32)
    for s in range(0, Xq.shape[0], 1000):
        S = (Xq[s:s + 1000] @ Xpool.T).toarray()
        if self_offset is not None:
            rows = np.arange(S.shape[0])
            S[rows, self_offset + s + rows] = -1
        p = np.argpartition(-S, K, 1)[:, :K]
        v = np.take_along_axis(S, p, 1)
        o = np.argsort(-v, 1)
        idx[s:s + 1000] = np.take_along_axis(p, o, 1)
        sim[s:s + 1000] = np.take_along_axis(v, o, 1)
    return idx, sim


# tf-idf fitted on train only, a dev case never retrieves itself
def build_neighbours(train_docs, dev_docs, test_docs):
    vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, max_features=400000,
                          sublinear_tf=True, dtype=np.float32)
    Xtr = vec.fit_transform(train_docs)
    Xdv = vec.transform(dev_docs)
    Xte = vec.transform(test_docs)
    Xpool = sp.vstack([Xtr, Xdv]).tocsr()
    di, ds = top_k(Xdv, Xpool, self_offset=Xtr.shape[0])
    ti, ts = top_k(Xte, Xpool)
    return di, ds, ti, ts


def knn_votes(idx, sim, ypool, k, p):
    # label vector of the k nearest cases, weighted by similarity^p
    w = np.maximum(sim[:, :k], 0) ** p
    votes = (w[:, :, None] * ypool[idx[:, :k]]).sum(1)
    return votes / np.maximum(w.sum(1, keepdims=True), 1e-9)


def gated_blend(p_model, votes, top1, a, gate, a_gate):
    # blend weight a, raised to a_gate when the nearest case is very similar
    near = (1 - a_gate) * p_model + a_gate * votes
    far = (1 - a) * p_model + a * votes
    return np.where((top1 >= gate)[:, None], near, far)


if __name__ == "__main__":
    train_docs = load_docs(f"{TOKEN_DIR}/train.npz")
    dev_docs = load_docs(f"{TOKEN_DIR}/dev.npz")
    test_docs = load_docs(f"{TOKEN_DIR}/test.npz")
    di, ds, ti, ts = build_neighbours(train_docs, dev_docs, test_docs)
    np.savez_compressed(OUT_FILE, di=di, ds=ds, ti=ti, ts=ts)
    close = (ds[:, 0] >= 0.9).mean()
    print(f"dev cases with top-1 similarity >= 0.9: {close:.4f}")
