import gzip
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import lexical
from dense_model import case_chunks, make_classifier
from distill import distillation_loss
from peft_methods import attach_method

PREP_FILE = "p7_data.pkl.gz"
MODELS_DIR = "models"
OUT_DIR = "dense_runs"
DEVICE = "cuda"
SCALE = 20.0
ACCUM = 4

BASE = dict(epochs=2, lr=3e-4, max_minutes=150, chunks=2, negatives=4, seed=0)
LORMA = {"rank": 16, "alpha": 32}
CONFIGS = {
    "d3_arch3_partial": dict(BASE, arch="arch3", model="arch3", method="partial", method_cfg={"top_blocks": 4}, lr=1e-4),
    "d1_arch2_lorma": dict(BASE, arch="arch2", model="arch2", method="lorma", method_cfg=LORMA),
    "d4_arch2_ia3": dict(BASE, arch="arch2", model="arch2", method="ia3", method_cfg={}, lr=3e-3),
    "d5_arch2_lorma_kd": dict(BASE, arch="arch2", model="arch2", method="lorma", method_cfg=LORMA, kd=True, teacher_scale=30.0),
    "d2b_arch4_qlora": dict(BASE, arch="arch4", model="arch4_mntp", bidirectional=True, method="qlora", method_cfg=LORMA,
                            chunks=1, negatives=2),
    "d6b_arch4_lorma": dict(BASE, arch="arch4", model="arch4_mntp", bidirectional=True, method="lorma", method_cfg=LORMA,
                            chunks=1, negatives=2),
    "d7_arch1_lorma": dict(BASE, arch="arch1", model="arch1", method="lorma", method_cfg=LORMA, chunks=1, negatives=2,
                           epochs=1, max_minutes=45),
}


def embed(clf, data, doc_id, split, n_chunks, fp16):
    # mean of the chunk vectors, unit length
    ids = np.asarray(data[split]["ids"][doc_id])
    chunks = case_chunks(ids, clf.backbone, n_chunks)
    x = torch.tensor(chunks, device=DEVICE)
    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        v = clf.features(x).float()
    return F.normalize(v.mean(0), dim=0)


def score_split(clf, data, split, n_chunks, fp16):
    clf.eval()
    qids = data[split]["qids"]
    cids = data[split]["cands"]
    with torch.no_grad():
        cand_vecs = []
        for c in cids:
            cand_vecs.append(embed(clf, data, c, split, n_chunks, fp16))
        C = torch.stack(cand_vecs)
        S = np.zeros((len(qids), len(cids)), np.float32)
        for a in range(len(qids)):
            k = qids[a]
            S[a] = (C @ embed(clf, data, k, split, n_chunks, fp16)).cpu().numpy()
            if k in cids:
                S[a, cids.index(k)] = -1e9
    clf.train()
    return S


def dev_f1(data, S):
    queries = {}
    for k in data["dev"]["qids"]:
        queries[k] = {"relevant": data["dev"]["queries"][k]}
    f, p, r = lexical.micro_f1(lexical.top7(S, data["dev"]["qids"], data["dev"]["cands"]), queries)
    return f


def main(tag):
    cfg = CONFIGS[tag]
    out = os.path.join(OUT_DIR, tag)
    os.makedirs(out, exist_ok=True)
    with gzip.open(PREP_FILE, "rb") as f:
        data = pickle.load(f)
    print(tag, cfg)

    torch.manual_seed(cfg["seed"])
    clf = make_classifier(cfg["arch"], os.path.join(MODELS_DIR, cfg["model"]), cfg.get("bidirectional", False), num_classes=2)
    clf = attach_method(clf, cfg["method"], cfg["method_cfg"])
    clf = clf.to(DEVICE)
    params = []
    n_train = 0
    for p in clf.parameters():
        if p.requires_grad:
            params.append(p)
            n_train += p.numel()
    print(f"{cfg['arch']} {cfg['method']} | trainable {n_train / 1e6:.2f}M")
    n_chunks = cfg["chunks"]
    fp16 = cfg["arch"] != "arch1"

    pairs = []
    for k in data["train"]["qids"]:
        for pos in data["train"]["queries"][k]:
            pairs.append((k, pos))
    print("training pairs", len(pairs))
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01)
    random.seed(cfg["seed"])
    t0 = time.time()
    best = -1.0
    best_state = None
    step = 0
    for ep in range(cfg["epochs"]):
        random.shuffle(pairs)
        for n in range(len(pairs)):
            k, pos = pairs[n]
            # hard negatives from the tf-idf top 20
            rel = set(data["train"]["queries"][k])
            hard = []
            for c, s in data["train"]["tfidf_top50"][k]:
                if c not in rel and c != k:
                    hard.append((c, s))
            negs = random.sample(hard[:20], min(cfg["negatives"], len(hard[:20])))
            group = [pos]
            teacher = [0.0]
            for c, s in negs:
                group.append(c)
                teacher.append(s)
            q = embed(clf, data, k, "train", n_chunks, fp16)
            doc_vecs = []
            for c in group:
                doc_vecs.append(embed(clf, data, c, "train", n_chunks, fp16))
            docs = torch.stack(doc_vecs)
            logits = (SCALE * (docs @ q)).unsqueeze(0)
            label = torch.zeros(1, dtype=torch.long, device=DEVICE)
            if cfg.get("kd", False):
                teacher[0] = dict(data["train"]["tfidf_top50"][k]).get(pos, 0.0)
                t_logits = torch.tensor([teacher], device=DEVICE) * cfg["teacher_scale"]
                loss = distillation_loss(logits, t_logits, label, tau=2.0, kd_weight=0.5)
            else:
                loss = F.cross_entropy(logits, label)
            (loss / ACCUM).backward()
            if (n + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 100 == 0:
                    minutes = (time.time() - t0) / 60
                    print(f"epoch {ep} | step {step} | loss {loss.item():.4f} | minutes {minutes:.1f}")
            if (time.time() - t0) / 60 > cfg["max_minutes"]:
                break
        S = score_split(clf, data, "dev", n_chunks, fp16)
        f = dev_f1(data, S)
        print(f"epoch {ep} done | dev micro-F1@7 dense only {f:.4f}")
        if f > best:
            best = f
            best_state = {}
            for name, p in clf.named_parameters():
                if p.requires_grad:
                    best_state[name] = p.detach().cpu().clone()
        if (time.time() - t0) / 60 > cfg["max_minutes"]:
            break

    clf.load_state_dict(best_state, strict=False)
    dev_S = score_split(clf, data, "dev", n_chunks, fp16)
    test_S = score_split(clf, data, "test", n_chunks, fp16)
    lex = data["dev"]["tfidf"].astype(np.float32)
    fused = {}
    for w in [0.0, 0.25, 0.5, 1.0]:
        fused[w] = round(dev_f1(data, lexical.zscore(lex) + w * lexical.zscore(dev_S)), 4)
    print(f"dev micro-F1@7: dense {dev_f1(data, dev_S):.4f} | tf-idf + w x dense", fused)
    np.savez_compressed(os.path.join(out, "scores.npz"), dev=dev_S, test=test_S,
                        dev_qids=np.array(data["dev"]["qids"]), dev_cands=np.array(data["dev"]["cands"]),
                        test_qids=np.array(data["test"]["qids"]), test_cands=np.array(data["test"]["cands"]))
    metrics = {"cfg": cfg, "dev_dense": dev_f1(data, dev_S), "dev_fused": fused, "minutes": (time.time() - t0) / 60}
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
