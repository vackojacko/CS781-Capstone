# p2: arch 4 + lora trained on labels from the whole headnote, without the sentence position features
import os
from multiprocessing import Pool

import numpy as np
import torch

import model as M
import oracle
import train as T
from data import load_docs

RUNS_DIR = "work/runs"
TAG = "p2_arch4_lora_fullref_nopos"
CFG = dict(arch="arch4", bidirectional=True, method="lora", epochs=2, lr_body=3e-4, max_minutes=300, seed=0)
LABEL_SENTS = 200
LABEL_WORDS = 900
WHOLE_REF = 100000


class NoPositionModel(M.SalienceModel):
    def forward(self, toks):
        v = self.drop(self.encode(toks))
        extra = torch.zeros(len(toks), 2, device=v.device)
        h, _ = self.lstm(torch.cat([v, extra], dim=1).unsqueeze(0))
        return self.out(self.drop(h[0])).squeeze(1)


def whole_labels(doc):
    # greedy oracle against the whole headnote, up to 900 words, on the first 200 sentences
    oracle.REF_WORDS = WHOLE_REF
    labels = oracle.oracle_labels(doc["sents"][:LABEL_SENTS], doc["summary"], max_words=LABEL_WORDS)
    return labels + [0] * max(0, len(doc["sents"]) - LABEL_SENTS)


def main():
    out = os.path.join(RUNS_DIR, TAG)
    os.makedirs(out, exist_ok=True)
    train_docs, dev_docs, test_docs = load_docs("train"), load_docs("dev"), load_docs("test")
    with Pool(4) as pool:
        labels = pool.map(whole_labels, train_docs, chunksize=16)
    for d, lab in zip(train_docs, labels):
        d["labels"] = lab

    torch.manual_seed(CFG["seed"])
    base = M.build(CFG["arch"], CFG["method"], CFG["bidirectional"])
    model = NoPositionModel(base.backbone).to(T.DEVICE)
    best, state = T.train(model, train_docs, dev_docs, epochs=CFG["epochs"], lr_body=CFG["lr_body"],
                          max_minutes=CFG["max_minutes"], seed=CFG["seed"])
    model.load_state_dict(state, strict=False)
    print(f"best dev rouge {best:.4f}")

    dev_ids, test_ids = [], []
    for d in dev_docs:
        dev_ids.append(d["id"])
    for d in test_docs:
        test_ids.append(d["id"])
    dev_arr = np.array(T.predict(model, dev_docs), dtype=object)
    test_arr = np.array(T.predict(model, test_docs), dtype=object)
    np.savez(os.path.join(out, "scores.npz"), dev_ids=np.array(dev_ids, dtype=object), dev=dev_arr,
             test_ids=np.array(test_ids, dtype=object), test=test_arr)
    print("wrote", os.path.join(out, "scores.npz"))


if __name__ == "__main__":
    main()
