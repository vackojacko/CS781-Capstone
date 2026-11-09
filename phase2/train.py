# NER fine-tuning of the phase 1 models: cross-validation (train fold1 eval fold2, and the reverse), then a final model on both folds
# python train.py <run name>
import json
import math
import os
import random
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

import model as M
from decode import decode_doc, propagate, repair_spans, strict_scores
from lorma import apply_method
from preprocess import (DATA_DIR, IGNORE, LABELS, N_TAGS, class_weights, encode_docs, load_split, load_tokenizer,
                        training_windows)

OUT_DIR = "runs"
MNTP_DIRS = {0: "runs/arch4_mntp_s0", 1: "runs/arch4_mntp_s1"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR_HEAD = 1e-3
ACCUM = 1
RARE_COPIES = 2
DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
CW_POWER = 0.5
RANK, ALPHA = 16, 16
DEFAULT_PROP = dict(min_len=6, min_conf=0.5, relabel=True)

PROP_GRID = []
for m in (3, 4, 6):
    for c in (0.0, 0.5, 0.8):
        for r in (True, False):
            PROP_GRID.append(dict(min_len=m, min_conf=c, relabel=r))


def run_config(arch, method, epochs, lr_body, batch, header=128, seed=0, bidir=False, model_dir=None, run_cv=True,
               repair=True):
    if model_dir is None:
        model_dir = M.MODEL_DIRS[arch]
    return dict(arch=arch, method=method, epochs=epochs, lr_body=lr_body, batch=batch, header=header, seed=seed,
                bidir=bidir, model_dir=model_dir, run_cv=run_cv, repair=repair)


# the runs behind the phase 2 submissions
RUNS = {
    "arch1_lorma_h128": run_config("arch1", "lorma", 10, 5e-4, 4),
    "arch2_full_h128": run_config("arch2", "full", 8, 1e-4, 8, repair=False),
    "arch3_full_h128": run_config("arch3", "full", 8, 1e-4, 8, repair=False),
    "arch4_full_h128": run_config("arch4", "full", 8, 1e-4, 4, repair=False),
    "arch4_mntp_lora_h128_bidir": run_config("arch4", "lora", 10, 5e-4, 4, bidir=True, model_dir=MNTP_DIRS[0]),
    "arch4_mntp_lora_h128_s1_bidir": run_config("arch4", "lora", 6, 5e-4, 4, seed=1, bidir=True,
                                                model_dir=MNTP_DIRS[1], run_cv=False),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_from_windows(docs, wins, bb, device):
    ids, tags = [], []
    for d, s, n in wins:
        doc = docs[d]
        pre = M.window_prefix(bb, doc)
        ids.append(pre + doc["ids"][s:s + n].tolist())
        tags.append([IGNORE] * len(pre) + doc["tags"][s:s + n].tolist())
    ids = torch.tensor(ids, device=device)
    tags = torch.tensor(tags, device=device)
    return ids, tags


def kd_loss(logits, t_logits, tags, temp):
    # T^2 * KL(teacher || student) at temperature T, on the tokens with gold tags
    keep = tags.view(-1) != IGNORE
    s_lp = F.log_softmax(logits.float().view(-1, N_TAGS)[keep] / temp, -1)
    t_p = F.softmax(t_logits.float().view(-1, N_TAGS)[keep] / temp, -1)
    return F.kl_div(s_lp, t_p, reduction="batchmean") * temp ** 2


def evaluate(model, docs, device):
    pred = {}
    for d, doc in docs.items():
        pred[d] = decode_doc(doc, M.doc_logprobs(model, doc, device))
    gold = {}
    for d, doc in docs.items():
        gold[d] = doc["gold"]
    return strict_scores(gold, pred), pred


def train(model, train_docs, eval_docs, device, epochs=10, lr_body=5e-5, lr_head=1e-3, batch=4, accum=2,
          warmup=0.06, weight_decay=0.01, rare_copies=2, cw_power=0.5, clip=1.0, label_smoothing=0.0, seed=0,
          teacher=None, kd_alpha=0.5, kd_temp=2.0):
    set_seed(seed)
    bb = model.backbone
    length = M.content_length(bb)
    if teacher is not None:
        assert M.content_length(teacher.backbone) == length and teacher.backbone.prefix == bb.prefix
        teacher.eval()
    wins = training_windows(train_docs, length, length // 2, rare_copies)
    weights = class_weights(train_docs, cw_power).to(device)

    head_params = []
    body_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            body_params.append(p)
        else:
            head_params.append(p)
    groups = [{"params": head_params, "lr": lr_head}]
    if body_params:
        groups.append({"params": body_params, "lr": lr_body})
    all_params = head_params + body_params
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)

    # linear warmup then linear decay
    steps = math.ceil(len(wins) / (batch * accum)) * epochs
    warm = int(warmup * steps)

    def lr_factor(u):
        return min(1.0, (u + 1) / max(1, warm)) * max(0.0, (steps - u) / max(1, steps - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist = []
    best = -1.0
    best_state = None
    t0 = time.time()
    u = 0
    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(wins)
        tot = 0.0
        nb = 0
        for k in range(0, len(wins), batch):
            ids, tags = batch_from_windows(train_docs, wins[k:k + batch], bb, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(ids)
            loss = F.cross_entropy(logits.float().view(-1, N_TAGS), tags.view(-1), weight=weights,
                                   ignore_index=IGNORE, label_smoothing=label_smoothing)
            if teacher is not None:
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        t_logits = teacher(ids)
                kd = kd_loss(logits, t_logits, tags, kd_temp)
                loss = (1 - kd_alpha) * loss + kd_alpha * kd
            scaler.scale(loss / accum).backward()
            tot += loss.item()
            nb += 1
            if nb % accum == 0 or k + batch >= len(wins):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(all_params, clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                u += 1
        rec = {"epoch": ep, "loss": tot / max(nb, 1), "updates": u, "minutes": (time.time() - t0) / 60}
        if eval_docs:
            (macro, per), _ = evaluate(model, eval_docs, device)
            rec["F1"] = macro["F1"]
            if macro["F1"] > best:
                best = macro["F1"]
                best_state = {}
                for key, v in model.state_dict().items():
                    best_state[key] = v.detach().cpu().clone()
        hist.append(rec)
        line = f"epoch {ep} | loss {rec['loss']:.4f} | updates {u} | minutes {rec['minutes']:.1f}"
        if "F1" in rec:
            line += f" | F1 {rec['F1']:.4f}"
        print(line, flush=True)
    return hist, best, best_state


def new_model(cfg):
    set_seed(cfg["seed"])
    bb = M.Backbone(cfg["arch"], cfg["model_dir"], bidirectional=cfg["bidir"], dropout=DROPOUT)
    model = M.TokenClassifier(bb)
    model = apply_method(model, cfg["method"], r=RANK, alpha=ALPHA)
    return model.to(DEVICE)


def save_logprobs(model, docs, path):
    arrs = {}
    for d, doc in docs.items():
        arrs[d] = M.doc_logprobs(model, doc, DEVICE).astype(np.float16)
    np.savez_compressed(path, **arrs)
    out = {}
    for d in arrs:
        out[d] = arrs[d].astype(np.float32)
    return out


def tune_propagation(folds, cv_logp):
    gold = {}
    alldocs = {}
    for f in folds.values():
        for d, doc in f.items():
            gold[d] = doc["gold"]
            alldocs[d] = doc
    raw = {}
    for d in alldocs:
        raw[d] = decode_doc(alldocs[d], cv_logp[d])
    base, base_per = strict_scores(gold, raw)
    best_f1 = -1.0
    best_kw = None
    for kw in PROP_GRID:
        pred = {}
        for d in alldocs:
            pred[d] = propagate(alldocs[d]["text"], raw[d], kw["min_len"], kw["min_conf"], kw["relabel"])
        f1 = strict_scores(gold, pred)[0]["F1"]
        if f1 > best_f1:
            best_f1 = f1
            best_kw = kw
    if best_f1 <= base["F1"]:
        best_kw = None
    best_pred = raw
    if best_kw:
        best_pred = {}
        for d in alldocs:
            best_pred[d] = propagate(alldocs[d]["text"], raw[d], best_kw["min_len"], best_kw["min_conf"],
                                     best_kw["relabel"])
    final, final_per = strict_scores(gold, best_pred)
    print(f"CV strict macro-F1: raw {base['F1']:.4f} | with propagation {final['F1']:.4f}", best_kw)
    for l in LABELS:
        print(f"{l} {base_per[l]['F1']:.4f} {final_per[l]['F1']:.4f}")
    return best_kw


def predict_test(test_docs, test_logp, prop_kw, repair):
    pred = {}
    for d, doc in test_docs.items():
        spans = decode_doc(doc, test_logp[d])
        if prop_kw:
            spans = propagate(doc["text"], spans, prop_kw["min_len"], prop_kw["min_conf"], prop_kw["relabel"])
        if repair:
            spans = repair_spans(doc["text"], spans)
        pred[d] = spans
    return pred


def write_submission(pred, texts, path):
    sub = {}
    for d, text in texts.items():
        rows = []
        for sp in pred.get(d, []):
            s, e, lab = sp[0], sp[1], sp[2]
            assert lab in LABELS and 0 <= s < e <= len(text), (d, s, e, lab)
            rows.append([int(s), int(e), lab])
        sub[d] = sorted(rows)
    assert set(sub) == set(texts)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(sub, ensure_ascii=False))
    n = 0
    for v in sub.values():
        n += len(v)
    return n


def main(tag, cfg, teacher=None, kd_alpha=0.5, kd_temp=2.0):
    out = os.path.join(OUT_DIR, tag)
    os.makedirs(out, exist_ok=True)
    M.HEADER = cfg["header"]
    pt = load_tokenizer()
    folds = {}
    for f in ("fold1", "fold2"):
        texts, gold = load_split(DATA_DIR, [f])
        folds[f] = encode_docs(pt, texts, gold)
    test_texts, _ = load_split(DATA_DIR, ["fold3"])
    test_docs = encode_docs(pt, test_texts)

    best_epochs = []
    cv_logp = {}
    prop_kw = DEFAULT_PROP
    if cfg["run_cv"]:
        for tr, ev in (("fold1", "fold2"), ("fold2", "fold1")):
            print("train", tr, "eval", ev)
            model = new_model(cfg)
            hist, best, state = train(model, folds[tr], folds[ev], DEVICE, epochs=cfg["epochs"], lr_body=cfg["lr_body"],
                                      lr_head=LR_HEAD, batch=cfg["batch"], accum=ACCUM, rare_copies=RARE_COPIES,
                                      cw_power=CW_POWER, label_smoothing=LABEL_SMOOTHING, seed=cfg["seed"],
                                      teacher=teacher, kd_alpha=kd_alpha, kd_temp=kd_temp)
            model.load_state_dict(state)
            cv_logp.update(save_logprobs(model, folds[ev], os.path.join(out, "cv_logprobs_" + ev + ".npz")))
            best_h = hist[0]
            for h in hist:
                if h.get("F1", -1) > best_h.get("F1", -1):
                    best_h = h
            best_epochs.append(best_h["epoch"])
            del model, state
            torch.cuda.empty_cache()
        prop_kw = tune_propagation(folds, cv_logp)
        final_epochs = round(np.mean(best_epochs))
    else:
        final_epochs = cfg["epochs"]

    print("final model on fold1 + fold2 for", final_epochs, "epochs | propagation", prop_kw)
    train_all = dict(folds["fold1"])
    train_all.update(folds["fold2"])
    model = new_model(cfg)
    hist, _, _ = train(model, train_all, None, DEVICE, epochs=final_epochs, lr_body=cfg["lr_body"], lr_head=LR_HEAD,
                       batch=cfg["batch"], accum=ACCUM, rare_copies=RARE_COPIES, cw_power=CW_POWER,
                       label_smoothing=LABEL_SMOOTHING, seed=cfg["seed"], teacher=teacher, kd_alpha=kd_alpha,
                       kd_temp=kd_temp)
    test_logp = save_logprobs(model, test_docs, os.path.join(out, "test_logprobs.npz"))
    half = {}
    for k, v in model.state_dict().items():
        half[k] = v.half()
    torch.save(half, os.path.join(out, "model_final.pt"))
    pred = predict_test(test_docs, test_logp, prop_kw, cfg["repair"])
    n = write_submission(pred, test_texts, os.path.join(out, "submission_" + tag + ".zip"))
    print("wrote", n, "spans for", len(pred), "test docs")
    with open(os.path.join(out, "final_results.json"), "w") as f:
        json.dump({"final_epochs": final_epochs, "propagation": prop_kw, "history": hist}, f, indent=1, default=float)


if __name__ == "__main__":
    name = sys.argv[1]
    main(name, RUNS[name])
