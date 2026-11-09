import numpy as np

from lexical import csls, zscore

K_CSLS = 4
W_WORDS = 0.5
W_VOTES = 4.0


def view_score(S, k=K_CSLS):
    return zscore(csls(S, k))


def final_score(views, co, votes, k=K_CSLS, w_words=W_WORDS, w_votes=W_VOTES, dense=None, w_dense=0.0):
    S = view_score(views["full"], k) + view_score(views["best"], k) + w_words * view_score(views["words"], k)
    S = S + np.log1p(co) + w_votes * votes
    if dense is not None and w_dense > 0:
        S = S + w_dense * view_score(dense, k)
    return S
