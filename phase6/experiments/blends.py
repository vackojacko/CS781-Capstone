# the other blends of centrality and salience models: python -m experiments.blends <submission>, or all
# score = (1 - w_model) * centrality + w_model * mean of the model probabilities, then the top sentences up to the word budget
import os
import sys

import numpy as np
from bert_score import score as bert_score
from sklearn.feature_extraction.text import TfidfVectorizer

import blend as B
import centrality as C
from data import header_end, load_docs
from predict import load_model_scores, write_prediction

OUT_DIR = "work/experiments"

S1 = "s1_arch2_lorma"
S2 = "s2_arch4_lora"
S3 = "s3_arch3_pfeiffer"
S4 = "s4_arch1_houlsby"
S5 = "s5_arch2_lorma_s1"
K1 = "k1_arch3_pfeiffer_kd"
K2 = "k2_arch1_houlsby_kd"
K3 = "k3_arch2_lorma_kd_a4"
K4 = "k4_arch4_lora_kd_a2"
K5 = "k5_arch3_pfeiffer_kd_a4"
K6 = "k6_arch1_houlsby_kd_a4"
P2 = "p2_arch4_lora_fullref_nopos"

FOUR = [S1, S2, S3, S4]
THREE = [S1, S2, K1]
FIVE = [S1, S2, S5, K1, K2]
SEVEN = [S1, S2, S5, K1, K3, K4, K5]
NINE = [S1, S2, S5, K1, K2, K3, K4, K5, K6]

L = "local"
G = "global"
LG = "local+global"


def recipe(models, kind, w_model, words, skip=True, norm_model=False, mmr=0.0):
    return dict(models=models, kind=kind, w_model=w_model, words=words, skip=skip, norm_model=norm_model, mmr=mmr)


RECIPES = {
    # one or a few models, the judgment's own idf, no header skip
    "05": recipe([S1], L, 1.0, 600, skip=False),
    "06": recipe([S1], L, 1.0, 750, skip=False),
    "07": recipe([S1], L, 0.5, 600, skip=False),
    "09": recipe([S1], L, 0.8, 500, skip=False),
    "10": recipe([S2], L, 0.8, 500, skip=False),
    "11": recipe([S2], L, 0.6, 600, skip=False),
    "12": recipe(FOUR, L, 0.8, 500, skip=False),
    # global idf, alone or averaged with the local one
    "18": recipe([S1, S2], G, 0.0, 600, skip=False),
    "19": recipe([S1, S2], LG, 0.0, 600, skip=False),
    "20": recipe([S1, S2], LG, 0.1, 600, skip=False),
    "21": recipe([S1, S2], LG, 0.3, 600, skip=False),
    "23": recipe(THREE, G, 0.5, 600),
    "24": recipe(THREE, G, 0.4, 600, skip=False),
    "25": recipe(THREE, LG, 0.5, 600),
    "26": recipe(THREE, LG, 0.3, 600),
    "27": recipe(THREE, G, 0.3, 600, skip=False),
    "28": recipe(FIVE, G, 0.5, 600),
    "29": recipe(FIVE, G, 0.6, 600),
    "30": recipe(FIVE, G, 0.7, 600),
    "31": recipe(FIVE + [P2], G, 0.6, 600),
    "32": recipe(FIVE + [P2], G, 0.5, 600),
    "33": recipe(FIVE + [P2], LG, 0.7, 600),
    # 34 to 36: model mean rescaled to 0..1 in each judgment
    "34": recipe(FIVE, G, 0.5, 500, norm_model=True),
    "35": recipe(FIVE, G, 0.4, 600, norm_model=True),
    "36": recipe(FIVE, G, 0.5, 600, norm_model=True),
    "37": recipe(FIVE, G, 0.55, 600),
    "38": recipe(FIVE, G, 0.55, 550),
    "39": recipe(FIVE, G, 0.5, 650),
    "40": recipe(FIVE, G, 0.55, 650),
    "41": recipe(FIVE, G, 0.5, 550),
    "42": recipe(NINE, G, 0.6, 600),
    "43": recipe(NINE, G, 0.6, 550),
    "44": recipe(NINE, G, 0.55, 600),
    "45": recipe(SEVEN, G, 0.55, 550),
    "46": recipe(SEVEN, G, 0.6, 550),
    "47": recipe(SEVEN, G, 0.6, 600),
    "48": recipe(NINE, G, 0.55, 550),
    "49": recipe(SEVEN, G, 0.55, 600),
    "50": recipe(SEVEN, G, 0.5, 600),
    "51": recipe(NINE, G, 0.5, 600),
    "52": recipe(SEVEN, G, 0.5, 550),
    "53": recipe(NINE, G, 0.5, 550),
    "54": recipe(NINE, G, 0.45, 600),
    "55": recipe(SEVEN, G, 0.45, 600),
    "56": recipe(SEVEN, G, 0.45, 550),
    "57": recipe(NINE, G, 0.45, 550),
    # 58 to 63: redundancy penalty (mmr)
    "58": recipe(SEVEN, G, 0.5, 550, mmr=0.2),
    "59": recipe(SEVEN, G, 0.5, 600, mmr=0.3),
    "60": recipe(SEVEN, G, 0.5, 550, mmr=0.3),
    "61": recipe(SEVEN, G, 0.5, 600, mmr=0.2),
    "62": recipe(SEVEN, G, 0.5, 550, mmr=0.1),
    "63": recipe(SEVEN, G, 0.5, 600, mmr=0.1),
    "64": recipe(SEVEN, G, 0.52, 575),
    "65": recipe(NINE, G, 0.52, 575),
    "66": recipe(NINE, LG, 0.52, 525),
    "67": recipe(NINE, LG, 0.52, 575),
    "69": recipe(SEVEN, LG, 0.52, 525),
    "70": recipe(SEVEN, G, 0.48, 575),
    "71": recipe(NINE, G, 0.48, 575),
    "72": recipe(SEVEN, LG, 0.52, 575),
    "73": recipe(NINE, G, 0.52, 525),
    "74": recipe(NINE, LG, 0.48, 525),
    "75": recipe(NINE, LG, 0.48, 575),
    "76": recipe(SEVEN, G, 0.48, 525),
    "77": recipe(SEVEN, LG, 0.48, 525),
    "78": recipe(SEVEN, LG, 0.48, 575),
    "79": recipe(NINE, G, 0.48, 525),
    "81": recipe(SEVEN, G, 0.55, 525),
}


def local_centrality(sents):
    return C.centrality(TfidfVectorizer().fit_transform(sents))


def sentence_scores(sents, model_avg, cfg, glob_vec):
    if cfg["kind"] == L:
        c = local_centrality(sents)
    elif cfg["kind"] == G:
        c = C.global_centrality(sents, glob_vec)
    else:
        c = (local_centrality(sents) + C.global_centrality(sents, glob_vec)) / 2
    m = model_avg
    if cfg["norm_model"]:
        m = C.norm(m)
    sc = (1 - cfg["w_model"]) * c + cfg["w_model"] * m
    if cfg["skip"]:
        sc[:header_end(sents)] = -1.0
    return sc


def select_mmr(sents, scores, max_words, lam, glob_vec):
    # each next sentence: score minus lam times its highest similarity to a sentence already picked
    if lam == 0:
        return B.select(sents, scores, max_words)
    m = glob_vec.transform(sents)
    sim = (m @ m.T).toarray()
    keep = []
    left = list(range(len(sents)))
    used = 0
    while left and used < max_words:
        best, best_val = None, None
        for i in left:
            penalty = 0.0
            for j in keep:
                penalty = max(penalty, sim[i][j])
            val = scores[i] - lam * penalty
            if best_val is None or val > best_val:
                best, best_val = i, val
        keep.append(best)
        left.remove(best)
        used += len(sents[best].split())
    keep.sort()
    out = []
    for i in keep:
        out.append(sents[i])
    return " ".join(out)


def summaries(docs, split, cfg, glob_vec):
    model_scores = load_model_scores(cfg["models"], split)
    out = []
    for k in range(len(docs)):
        sents = docs[k]["sents"]
        avg = B.model_mean(model_scores, k, len(sents))
        sc = sentence_scores(sents, avg, cfg, glob_vec)
        out.append(select_mmr(sents, sc, cfg["words"], cfg["mmr"], glob_vec))
    return out


def main():
    names = sys.argv[1:]
    if not names or names == ["all"]:
        names = list(RECIPES)
    os.makedirs(OUT_DIR, exist_ok=True)
    train, dev, test = load_docs("train"), load_docs("dev"), load_docs("test")
    glob_vec = C.fit_global_idf(train)
    refs = []
    for d in dev:
        refs.append(d["summary"])
    for name in names:
        cfg = RECIPES[name]
        cands = summaries(dev, "dev", cfg, glob_vec)
        F1 = bert_score(cands, refs, lang="en", batch_size=16, verbose=False)[2]
        test_summ = summaries(test, "test", cfg, glob_vec)
        preds = []
        for k in range(len(test)):
            preds.append({"id": test[k]["id"], "summary": test_summ[k]})
        path = os.path.join(OUT_DIR, "submission_" + name + ".zip")
        write_prediction(preds, path)
        words = np.mean([len(s.split()) for s in test_summ])
        print(f"{name} | dev bertscore f1 {float(F1.mean()):.4f} | test words {words:.0f} | {path}")


if __name__ == "__main__":
    main()
