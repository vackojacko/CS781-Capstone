# training one salience model (phase 1 backbone + lorma / lora / adapters) on the oracle labels
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from bert_score import score as bert_score

import model as M
import oracle
from blend import select
from data import load_docs
from distill import load_teacher
from predict import write_prediction

OUT_DIR = "work/runs"
DEVICE = "cuda"
LENGTHS = (450, 600, 750)

BASE = dict(epochs=2, lr_body=3e-4, max_minutes=330)
CONFIGS = {
    "s1_arch2_lorma": dict(BASE, arch="arch2", method="lorma", score_train=True),
    "s2_arch4_lora": dict(BASE, arch="arch4", bidirectional=True, method="lora", score_train=True),
    "s3_arch3_pfeiffer": dict(BASE, arch="arch3", method="pfeiffer", lr_body=1e-3),
    "s4_arch1_houlsby": dict(BASE, arch="arch1", method="houlsby", lr_body=1e-3),
    "s5_arch2_lorma_s1": dict(BASE, arch="arch2", method="lorma", seed=1),
}


def doc_loss(model, doc, teacher=None, kd_weight=0.5, pos_weight=4.0):
    logits = model(doc["toks"])
    gold = torch.tensor(doc["labels"], dtype=torch.float32, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, gold, pos_weight=torch.tensor(pos_weight, device=logits.device))
    if teacher is not None:
        # distillation: half gold labels, half teacher probabilities
        soft = torch.tensor(teacher, dtype=torch.float32, device=logits.device)
        loss = (1 - kd_weight) * loss + kd_weight * F.binary_cross_entropy_with_logits(logits, soft)
    return loss


@torch.no_grad()
def predict(model, docs):
    model.eval()
    out = []
    for d in docs:
        out.append(torch.sigmoid(model(d["toks"])).cpu().numpy())
    model.train()
    return out


def dev_overlap(model, docs, max_words=600):
    # fast dev check: rouge-1 f1 of the top 600 words against the reference start
    scores = predict(model, docs)
    vals = []
    for k in range(len(docs)):
        d = docs[k]
        summ = select(d["sents"], scores[k], max_words).lower().split()
        ref = d["summary"].lower().split()[:oracle.REF_WORDS]
        vals.append(oracle.overlap_f1(oracle.ngrams(summ, 1), oracle.ngrams(ref, 1)))
    return float(np.mean(vals))


def train(model, train_docs, dev_docs, epochs=1, lr_body=3e-4, lr_head=1e-3, accum=8, teacher=None,
          max_minutes=500, seed=0, max_sents=200):
    random.seed(seed)
    torch.manual_seed(seed)
    body, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            body.append(p)
        else:
            head.append(p)
    opt = torch.optim.AdamW([{"params": body, "lr": lr_body}, {"params": head, "lr": lr_head}], weight_decay=0.01)
    steps = epochs * len(train_docs) // accum

    def lr_factor(s):
        # 50 warmup steps, then linear decay to 5%
        return min(1.0, s / 50) * max(0.05, 1 - s / max(steps, 1))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    t0 = time.time()
    step = 0
    best, best_state = -1.0, None
    order = list(range(len(train_docs)))
    for ep in range(epochs):
        random.shuffle(order)
        total = 0.0
        for k in range(len(order)):
            idx = order[k]
            d = train_docs[idx]
            if len(d["toks"]) > max_sents:
                d = {"toks": d["toks"][:max_sents], "labels": d["labels"][:max_sents]}
            t = None
            if teacher is not None:
                t = teacher[idx][:len(d["toks"])]
            loss = doc_loss(model, d, t)
            (loss / accum).backward()
            total += loss.item()
            if (k + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(body + head, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 100 == 0:
                    minutes = (time.time() - t0) / 60
                    avg = total / (k + 1)
                    print(f"epoch {ep} | step {step}/{steps} | loss {avg:.4f} | minutes {minutes:.1f}", flush=True)
            if (time.time() - t0) / 60 > max_minutes:
                break
        score = dev_overlap(model, dev_docs)
        minutes = (time.time() - t0) / 60
        print(f"epoch {ep} done | dev rouge (top 600 words vs ref start) {score:.4f} | minutes {minutes:.1f}", flush=True)
        if score > best:
            best = score
            best_state = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    best_state[n] = p.detach().cpu().clone()
        if (time.time() - t0) / 60 > max_minutes:
            break
    return best, best_state


def run(tag, cfg, teacher_path=None):
    out = os.path.join(OUT_DIR, tag)
    os.makedirs(out, exist_ok=True)
    train_docs, dev_docs, test_docs = load_docs("train"), load_docs("dev"), load_docs("test")
    print(tag, "| train", len(train_docs), "dev", len(dev_docs), "test", len(test_docs), "|", cfg)

    torch.manual_seed(cfg.get("seed", 0))
    model = M.build(cfg["arch"], cfg["method"], cfg.get("bidirectional", False)).to(DEVICE)
    n_train = 0
    for p in model.parameters():
        if p.requires_grad:
            n_train += p.numel()
    print(f"{cfg['arch']} {cfg['method']} | trainable {n_train / 1e6:.2f}M")

    teacher = None
    if teacher_path is not None:
        teacher = load_teacher(teacher_path, train_docs)

    best, state = train(model, train_docs, dev_docs, epochs=cfg["epochs"], lr_body=cfg["lr_body"], teacher=teacher,
                        max_minutes=cfg["max_minutes"], seed=cfg.get("seed", 0))
    model.load_state_dict(state, strict=False)
    print(f"best dev rouge {best:.4f}")

    dev_scores = predict(model, dev_docs)
    test_scores = predict(model, test_docs)
    dev_ids, test_ids = [], []
    for d in dev_docs:
        dev_ids.append(d["id"])
    for d in test_docs:
        test_ids.append(d["id"])
    dev_ids = np.array(dev_ids, dtype=object)
    test_ids = np.array(test_ids, dtype=object)
    dev_arr = np.array(dev_scores, dtype=object)
    test_arr = np.array(test_scores, dtype=object)
    path = os.path.join(out, "scores.npz")
    if cfg.get("score_train"):
        # teacher targets for distillation
        train_ids = []
        for d in train_docs:
            train_ids.append(d["id"])
        train_ids = np.array(train_ids, dtype=object)
        train_arr = np.array(predict(model, train_docs), dtype=object)
        np.savez(path, dev_ids=dev_ids, dev=dev_arr, test_ids=test_ids, test=test_arr, ids=train_ids, train=train_arr)
    else:
        np.savez(path, dev_ids=dev_ids, dev=dev_arr, test_ids=test_ids, test=test_arr)

    # pick the summary length by real bertscore on dev
    refs = []
    for d in dev_docs:
        refs.append(d["summary"])
    results = {}
    for words in LENGTHS:
        cands = []
        for k in range(len(dev_docs)):
            cands.append(select(dev_docs[k]["sents"], dev_scores[k], words))
        F1 = bert_score(cands, refs, lang="en", batch_size=16, verbose=False)[2]
        results[words] = round(float(F1.mean()), 4)
        print(f"dev bertscore f1 at {words} words: {results[words]:.4f}")
    pick = max(results, key=results.get)
    preds = []
    for k in range(len(test_docs)):
        preds.append({"id": test_docs[k]["id"], "summary": select(test_docs[k]["sents"], test_scores[k], pick)})
    write_prediction(preds, os.path.join(out, "submission_" + tag + ".zip"))


if __name__ == "__main__":
    tag = sys.argv[1]
    run(tag, CONFIGS[tag])
