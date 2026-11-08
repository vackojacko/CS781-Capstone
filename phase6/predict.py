# final submission (#82): blend of seven salience models with centrality and headnote propensity, 525 words
import json
import os
import zipfile

import numpy as np
from bert_score import score as bert_score

import blend as B
import centrality as C
from data import load_docs

RUNS_DIR = "work/runs"
OUT_ZIP = "work/submission_final.zip"

# arch 2 lorma (2 seeds), arch 4 lora, and the four distilled students
SEVEN = ["s1_arch2_lorma", "s2_arch4_lora", "s5_arch2_lorma_s1", "k1_arch3_pfeiffer_kd", "k3_arch2_lorma_kd_a4",
         "k4_arch4_lora_kd_a2", "k5_arch3_pfeiffer_kd_a4"]


def write_prediction(preds, path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("prediction.json", json.dumps(preds))


def load_model_scores(names, split):
    scores = {}
    for name in names:
        z = np.load(os.path.join(RUNS_DIR, name, "scores.npz"), allow_pickle=True)
        scores[name] = list(z[split])
    return scores


def summaries(docs, split, glob_vec, propensity, prior, idf):
    model_scores = load_model_scores(SEVEN, split)
    out = []
    for k in range(len(docs)):
        sents = docs[k]["sents"]
        avg = B.model_mean(model_scores, k, len(sents))
        sc = B.blend_scores(sents, avg, glob_vec, propensity, prior, idf)
        out.append(B.select_filtered(sents, sc))
    return out


def main():
    train, dev, test = load_docs("train"), load_docs("dev"), load_docs("test")
    glob_vec = C.fit_global_idf(train)
    propensity, prior = C.fit_propensity(train)
    idf = C.idf_table(glob_vec)

    cands = summaries(dev, "dev", glob_vec, propensity, prior, idf)
    refs = []
    for d in dev:
        refs.append(d["summary"])
    F1 = bert_score(cands, refs, lang="en", batch_size=16, verbose=False)[2]
    print(f"dev bertscore f1: {float(F1.mean()):.4f}")

    test_summ = summaries(test, "test", glob_vec, propensity, prior, idf)
    preds = []
    for k in range(len(test)):
        preds.append({"id": test[k]["id"], "summary": test_summ[k]})
    write_prediction(preds, OUT_ZIP)
    print("wrote", OUT_ZIP)


if __name__ == "__main__":
    main()
