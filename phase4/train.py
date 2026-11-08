# one phase 4 run: python train.py <run>
# stage 1 fine-tunes a chunk classifier on the end of each case, stage 2 trains a bilstm over the last 8 chunk
# vectors, then every input variant is scored on dev, its threshold tuned on dev, and submissions are written
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from backbone import set_seed
from data import BPE_PATH, TOKEN_DIR, chunk_ids, chunk_length, load_tokens, make_batches
from distill import TEACHER_FILES, distillation_loss, load_teacher_cache, save_teacher_cache
from model import Aggregator, make_classifier
from peft_methods import attach_peft, unfreeze_top
from predict import first_score, margin, report, selected_score, tail_score, whole_score, write_submission
from tokenizer import BPETokenizer, PieceTokenizer

OUT_DIR = "runs"
MODEL_DIRS = {"arch1": "models/arch1", "arch2": "models/arch2", "arch3": "models/arch3", "arch4": "models/arch4_mntp"}

# defaults: batch 4, accum 8, lr_head 1e-3, eval every 400 updates, 480 minutes, last_k 2, seed 0
E1 = {"tag": "e1_arch4_lorma", "arch": "arch4", "bidirectional": True, "method": "lorma",
      "method_cfg": {"rank": 16, "alpha": 32}, "epochs": 2, "lr_body": 3e-4, "n_agg": 6000, "teacher_n": 16000}
E2 = {"tag": "e2_arch2_unfreeze", "arch": "arch2", "method": "unfreeze", "method_cfg": {"start_blocks": 4},
      "unfreeze_at": {0.25: 8, 0.5: 99}, "epochs": 2, "lr_body": 5e-5, "n_agg": 6000, "teacher_n": 16000}
E3 = {"tag": "e3_arch3_adapter", "arch": "arch3", "method": "adapter", "method_cfg": {"bottleneck": 64},
      "epochs": 3, "lr_body": 5e-4, "n_agg": 6000, "teacher_n": 16000}
E4 = {"tag": "e4_arch1_full", "arch": "arch1", "method": "full", "epochs": 1, "lr_body": 3e-5, "batch": 2,
      "accum": 16, "n_agg": 3000, "max_minutes": 200}
E5 = {"tag": "e5_arch2_kd", "arch": "arch2", "method": "unfreeze", "method_cfg": {"start_blocks": 4},
      "unfreeze_at": {0.25: 8, 0.5: 99}, "epochs": 2, "lr_body": 5e-5, "n_agg": 6000, "teachers": TEACHER_FILES}
E6 = dict(E1, tag="e6_arch4_lorma_tail3_s1", last_k=3, seed=1)
E10 = dict(E1, tag="e10_arch4_lorma_r32_e3_s2", method_cfg={"rank": 32, "alpha": 64}, epochs=3, seed=2, teacher_n=None)
E11 = dict(E1, tag="e11_arch4_lorma_s3", seed=3, teacher_n=None)
E12 = dict(E1, tag="e12_arch4_lorma_kd", teacher_n=None, teachers=TEACHER_FILES)
CONFIGS = {}
for c in [E1, E2, E3, E4, E5, E6, E10, E11, E12]:
    CONFIGS[c["tag"]] = c

CUES = ["opinion", "hold", "held", "conclusion", "satisfied", "erred", "accordingly", "therefore", "evidence", "reasons"]


@torch.no_grad()
def encode_cases(model, cases, k=None, batch=8, keep_vecs=True, fp16=True, max_chunks=48):
    # logits (and pooled vectors) of the last k chunks of every case, in reading order
    model.eval()
    bb = model.backbone
    L = chunk_length(bb)
    device = model.classifier.weight.device
    if not k:
        k = max_chunks
    items = []
    n_chunks = []
    for ci in range(len(cases)):
        chunks = chunk_ids(cases[ci], bb, L, L * 3 // 4, k)
        for j in range(len(chunks)):
            items.append(((ci, j), chunks[j]))
        n_chunks.append(len(chunks))
    logits = []
    vecs = []
    for n in n_chunks:
        logits.append(np.zeros((n, 2), np.float32))
        if keep_vecs:
            vecs.append(np.zeros((n, bb.d_model), np.float16))
    for group in make_batches(items, batch):
        x = torch.tensor([g[1] for g in group], device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
            v = model.features(x)
        lg = model.classifier(v.float()).float().cpu().numpy()
        v = v.float().cpu().numpy()
        for i in range(len(group)):
            ci, j = group[i][0]
            logits[ci][j] = lg[i]
            if keep_vecs:
                vecs[ci][j] = v[i]
    if not keep_vecs:
        vecs = None
    return logits, vecs


def evaluate_tail(model, dev, last_k, fp16):
    lg, _ = encode_cases(model, dev, k=last_k, keep_vecs=False, fp16=fp16)
    model.train()
    dev_y = np.array([c["label"] for c in dev])
    return report(tail_score(lg, last_k), dev_y)


def trainable_state(model):
    state = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            state[n] = p.detach().cpu().clone()
    return state


def format_record(rec):
    parts = []
    for k, v in rec.items():
        if isinstance(v, float):
            parts.append(k + " " + format(v, ".4f"))
        else:
            parts.append(k + " " + str(v))
    return " | ".join(parts)


def train_chunks(model, train, dev, device, epochs=2, batch=4, accum=8, lr_body=5e-5, lr_head=1e-3, warmup=0.06,
                 weight_decay=0.01, clip=1.0, fp16=True, last_k=2, teacher=None, tau=2.0, kd_weight=0.5,
                 unfreeze_at=None, eval_every=400, max_minutes=600, log=print, seed=0):
    # each epoch every case gives one random chunk from its last last_k chunks
    set_seed(seed)
    bb = model.backbone
    L = chunk_length(bb)
    head = []
    body = []
    for n, p in model.named_parameters():
        if n.startswith("classifier."):
            head.append(p)
        else:
            body.append(p)
    opt = torch.optim.AdamW([{"params": head, "lr": lr_head}, {"params": body, "lr": lr_body}],
                            weight_decay=weight_decay)
    per_epoch = math.ceil(len(train) / (batch * accum))
    total = per_epoch * epochs
    n_warm = int(warmup * total)

    def lr_factor(u):
        return min(1.0, (u + 1) / max(1, n_warm)) * max(0.0, (total - u) / max(1, total - n_warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    use_amp = fp16 and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pending = []
    if unfreeze_at:
        pending = sorted(unfreeze_at.items())
    hist = []
    best = -1.0
    best_state = None
    t0 = time.time()
    u = 0
    for ep in range(epochs):
        model.train()
        items = []
        for ci in range(len(train)):
            chunks = chunk_ids(train[ci], bb, L, L * 3 // 4, last_k)
            j = random.randrange(len(chunks))
            # key: case index and how many chunks back from the end
            items.append(((ci, len(chunks) - 1 - j), chunks[j]))
        batches = make_batches(items, batch)
        random.shuffle(batches)
        tot = 0.0
        nb = 0
        ce_sum = 0.0
        kd_sum = 0.0
        for bi in range(len(batches)):
            group = batches[bi]
            x = torch.tensor([g[1] for g in group], device=device)
            y = torch.tensor([train[g[0][0]]["label"] for g in group], device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            if teacher is not None:
                t = []
                for g in group:
                    ci, back = g[0]
                    t.append(teacher[ci, back])
                t = torch.tensor(np.stack(t), device=device)
                loss, parts = distillation_loss(logits, t, y, tau, kd_weight)
                ce_sum += parts["ce"]
                kd_sum += parts["kd"]
            else:
                loss = F.cross_entropy(logits.float(), y)
            loss = loss * len(group) / batch
            scaler.scale(loss / accum).backward()
            tot += loss.item()
            nb += 1
            last = bi == len(batches) - 1
            if nb % accum == 0 or last:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in head + body if p.requires_grad], clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                u += 1
                while pending and u >= pending[0][0] * total:
                    unfreeze_top(model, pending.pop(0)[1])
                    n_body = sum([p.numel() for p in body if p.requires_grad])
                    log(f"update {u}: unfroze, {n_body / 1e6:.1f}M body params trainable")
                if u % eval_every == 0 or last:
                    rec = {"epoch": ep + 1, "update": u, "loss": tot / nb, "minutes": (time.time() - t0) / 60}
                    if teacher is not None:
                        rec["ce"] = ce_sum / nb
                        rec["kd"] = kd_sum / nb
                    rec.update(evaluate_tail(model, dev, last_k, fp16))
                    if rec["f1_tuned"] > best:
                        best = rec["f1_tuned"]
                        best_state = trainable_state(model)
                    hist.append(rec)
                    log(format_record(rec))
                if (time.time() - t0) / 60 > max_minutes:
                    log("time budget reached")
                    return hist, best, best_state
    return hist, best, best_state


def agg_batches(vecs, k, batch, shuffle):
    # cases grouped by number of chunks (at most k)
    idx = {}
    for i in range(len(vecs)):
        n = min(len(vecs[i]), k)
        if n not in idx:
            idx[n] = []
        idx[n].append(i)
    out = []
    for n, ii in idx.items():
        if shuffle:
            random.shuffle(ii)
        for j in range(0, len(ii), batch):
            out.append((n, ii[j:j + batch]))
    if shuffle:
        random.shuffle(out)
    return out


def stack_last(vecs, ii, n, device):
    rows = []
    for i in ii:
        rows.append(vecs[i][-n:])
    return torch.tensor(np.stack(rows), dtype=torch.float32, device=device)


def agg_scores(agg, vecs, k, device):
    agg.eval()
    z = np.zeros(len(vecs))
    with torch.no_grad():
        for n, ii in agg_batches(vecs, k, 256, False):
            z[ii] = margin(agg(stack_last(vecs, ii, n, device)).cpu().numpy())
    return z


def train_aggregator(train_vecs, train_y, dev_vecs, dev_y, device, k=8, epochs=15, lr=1e-3, batch=64, seed=0,
                     log=print):
    set_seed(seed)
    agg = Aggregator(train_vecs[0].shape[1]).to(device)
    opt = torch.optim.AdamW(agg.parameters(), lr=lr, weight_decay=0.01)
    best = -1.0
    best_state = None
    for ep in range(epochs):
        agg.train()
        for n, ii in agg_batches(train_vecs, k, batch, True):
            x = stack_last(train_vecs, ii, n, device)
            y = torch.tensor(train_y[ii], device=device)
            loss = F.cross_entropy(agg(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        r = report(agg_scores(agg, dev_vecs, k, device), dev_y)
        f1_tuned = r["f1_tuned"]
        f1_prior = r["f1_prior"]
        log(f"aggregator epoch {ep + 1} | dev f1_tuned {f1_tuned:.4f} f1_prior {f1_prior:.4f}")
        if r["f1_tuned"] > best:
            best = r["f1_tuned"]
            best_state = {}
            for kk, v in agg.state_dict().items():
                best_state[kk] = v.clone()
    agg.load_state_dict(best_state)
    return agg, best


def cue_ids(tok):
    out = []
    for w in CUES:
        out.append(tuple(tok.encode_piece(w, True)))
    return out


def cue_counts(case, bb, cues, k=48):
    L = chunk_length(bb)
    counts = []
    for ids in chunk_ids(case, bb, L, L * 3 // 4, k):
        n = 0
        for c in cues:
            for i in range(len(ids) - len(c) + 1):
                if ids[i] == c[0] and tuple(ids[i:i + len(c)]) == c:
                    n += 1
        counts.append(n)
    return counts


def run(cfg):
    tag = cfg["tag"]
    out = os.path.join(OUT_DIR, tag)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a")

    def log(*a):
        s = " ".join([str(x) for x in a])
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = load_tokens(os.path.join(TOKEN_DIR, "train.npz"))
    dev = load_tokens(os.path.join(TOKEN_DIR, "dev.npz"))
    test = load_tokens(os.path.join(TOKEN_DIR, "test.npz"))
    log(tag, "| train", len(train), "dev", len(dev), "test", len(test), "| device", device)

    torch.manual_seed(cfg.get("seed", 0))
    clf = make_classifier(cfg["arch"], MODEL_DIRS[cfg["arch"]], cfg.get("bidirectional", False))
    clf = attach_peft(clf, cfg["method"], cfg.get("method_cfg"))
    clf.to(device)
    n_train = 0
    n_all = 0
    for p in clf.parameters():
        n_all += p.numel()
        if p.requires_grad:
            n_train += p.numel()
    log(cfg["arch"], cfg["method"], "| trainable", n_train, "of", n_all)

    teacher = None
    if cfg.get("teachers"):
        by_id = {}
        for c in train:
            by_id[c["id"]] = c
        ids = [str(i) for i in np.load(cfg["teachers"][0])["case_ids"]]
        # the student trains on the cases the teachers scored
        train = [by_id[i] for i in ids]
        teacher = load_teacher_cache(cfg["teachers"], ids)
        log("teacher cache from", cfg["teachers"], "|", len(train), "cases")

    # arch1 did not learn in fp16 in phase 3
    fp16 = cfg["arch"] != "arch1"
    t0 = time.time()
    hist, best, state = train_chunks(clf, train, dev, device, epochs=cfg["epochs"], batch=cfg.get("batch", 4),
                                     accum=cfg.get("accum", 8), lr_body=cfg["lr_body"],
                                     lr_head=cfg.get("lr_head", 1e-3), fp16=fp16, teacher=teacher,
                                     unfreeze_at=cfg.get("unfreeze_at", {}), eval_every=cfg.get("eval_every", 400),
                                     max_minutes=cfg.get("max_minutes", 480), log=log, seed=cfg.get("seed", 0),
                                     last_k=cfg.get("last_k", 2))
    clf.load_state_dict(state, strict=False)
    minutes = (time.time() - t0) / 60
    log(f"stage 1 done in {minutes:.1f} min, best dev f1 {best:.4f}")
    if n_train < 30e6:
        torch.save(state, os.path.join(out, "trainable_state.pt"))

    dev_lg, dev_vec = encode_cases(clf, dev, fp16=fp16)
    test_lg, test_vec = encode_cases(clf, test, fp16=fp16)

    # stage 2 trains on the last 8 chunk vectors of a random subset of train cases
    n_agg = cfg.get("n_agg", 8000)
    rng = np.random.RandomState(0)
    agg_idx = sorted(rng.choice(len(train), min(n_agg, len(train)), replace=False))
    agg_cases = [train[i] for i in agg_idx]
    agg_lg, agg_vec = encode_cases(clf, agg_cases, k=8, fp16=fp16)
    log("encoded", len(agg_idx), "train cases (last 8 chunks)")

    if cfg.get("teacher_n"):
        t_idx = sorted(np.random.RandomState(1).choice(len(train), cfg["teacher_n"], replace=False))
        t_cases = [train[i] for i in t_idx]
        tr_lg, _ = encode_cases(clf, t_cases, k=2, keep_vecs=False, fp16=fp16)
        save_teacher_cache(os.path.join(out, "teacher_train.npz"), [c["id"] for c in t_cases], tr_lg)
        log("teacher cache for", len(t_idx), "train cases")

    dev_y = np.array([c["label"] for c in dev])
    train_y = np.array([c["label"] for c in agg_cases])
    agg, agg_best = train_aggregator(agg_vec, train_y, dev_vec, dev_y, device, log=log)
    torch.save(agg.state_dict(), os.path.join(out, "aggregator.pt"))

    tok = PieceTokenizer(BPETokenizer.load(BPE_PATH))
    cues = cue_ids(tok)
    variants = {}
    for name, lg, vec, cases in [("dev", dev_lg, dev_vec, dev), ("test", test_lg, test_vec, test)]:
        cc = [cue_counts(c, clf.backbone, cues) for c in cases]
        variants[name] = {
            "first": first_score(lg),
            "last": tail_score(lg, 1),
            "tail2": tail_score(lg, 2),
            "whole": whole_score(lg),
            "selected": selected_score(lg, cc),
            "hier": agg_scores(agg, vec, 8, device),
        }
    results = {}
    for v in variants["dev"]:
        r = report(variants["dev"][v], dev_y)
        results[v] = r
        f1_0 = r["f1_0"]
        f1_prior = r["f1_prior"]
        f1_tuned = r["f1_tuned"]
        thr = r["thr"]
        log(f"dev {v} f1@0 {f1_0:.4f} | prior-shift {f1_prior:.4f} | tuned {f1_tuned:.4f} (thr {thr:.3f})")

    scores = {"dev_ids": np.array([c["id"] for c in dev]), "test_ids": np.array([c["id"] for c in test]), "dev_y": dev_y}
    for s in variants:
        for v in variants[s]:
            scores[s + "_" + v] = variants[s][v]
    np.savez(os.path.join(out, "scores.npz"), **scores)

    pick = "tail2"
    for v in ["whole", "hier"]:
        if results[v]["f1_tuned"] > results[pick]["f1_tuned"]:
            pick = v
    test_ids = [c["id"] for c in test]
    names = ["hier", "tail2"]
    if pick not in names:
        names.append(pick)
    for v in sorted(names):
        pred = variants["test"][v] > results[v]["thr"]
        n = write_submission(os.path.join(out, "submission_" + tag + "_" + v + ".zip"), test_ids, pred)
        share = float(pred.mean())
        log(f"submission {v}: {n} cases, {share:.3f} predicted 1")
    minutes = (time.time() - t0) / 60
    metrics = {"cfg": cfg, "trainable": n_train, "total": n_all, "hist": hist, "dev": results, "pick": pick,
               "minutes": minutes}
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    best_dev = results[pick]["f1_tuned"]
    log(f"done: pick {pick} dev {best_dev:.4f}, total {minutes:.1f} min")


if __name__ == "__main__":
    run(CONFIGS[sys.argv[1]])
