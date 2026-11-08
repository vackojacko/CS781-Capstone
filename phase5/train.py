import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

from data import N_LABELS, TOKEN_DIR, case_chunks, load_tokens
from distill import check_distillation_loss, kd_case_loss, load_teacher_cache
from model import CaseModel, case_loss, make_classifier
from peft_methods import attach_peft, check_lorma, check_nf4
from predict import sigmoid, write_submission
from thresholds import apply, best_global, per_label, report

MODEL_DIRS = {"arch1": "models/arch1", "arch2": "models/arch2", "arch3": "models/arch3", "arch4": "models/arch4_mntp"}
TEACHER_PATH = "runs/f1_arch4_lorma_attn_asl/teacher_train.npz"
OUT_DIR = "runs"

# f1 is the arch4 teacher, it also saves its logits on every training case
F1 = dict(arch="arch4", bidirectional=True, method="lorma", method_cfg={"rank": 16, "alpha": 32}, head="attn",
          loss="asl", epochs=2, lr_body=3e-4, chunks=4, ks=[1, 4, 8], eval_every=500, eval_n=2000, max_minutes=480,
          teacher_cache=True)
F1B = dict(F1, loss="bce", epochs=3, ks=[4], eval_every=1000, eval_n=1500, max_minutes=600, teacher_cache=False)
F1C = dict(F1B, head="pool")
F3 = dict(F1B, method="qlora", epochs=2)
F4 = dict(F1B, arch="arch2", bidirectional=False, method="partial", method_cfg={"top_blocks": 8}, lr_body=5e-5)
F8 = dict(F4, method="full", method_cfg=None, lr_body=3e-5)
F7 = dict(F4, teachers=[TEACHER_PATH])
F5 = dict(F1B, arch="arch3", bidirectional=False, method="adapter", method_cfg={"bottleneck": 64}, lr_body=5e-4)
F6 = dict(F1, arch="arch1", bidirectional=False, method="full", method_cfg=None, head="pool", loss="bce",
          lr_body=3e-5, epochs=1, ks=[4], eval_every=1000, eval_n=1500, max_minutes=240, teacher_cache=False)
G1 = dict(F4, epochs=6, max_minutes=640, eval_every=1500)
G2 = dict(G1, lr_body=1e-4)
G3 = dict(F8, epochs=6, max_minutes=640, eval_every=1500)
G4 = dict(G1, teachers=[TEACHER_PATH])
G6 = dict(G1, chunks=8)

RUNS = {
    "f1_arch4_lorma_attn_asl": F1,
    "f1b_arch4_lorma_attn_bce": F1B,
    "f1b_s1_arch4_lorma_attn_bce": dict(F1B, seed=1),
    "f1c_arch4_lorma_pool_bce": F1C,
    "f3_arch4_qlora_attn_bce": F3,
    "f4_arch2_partial_attn_bce": F4,
    "f4s1_arch2_partial_attn_bce_s1": dict(F4, seed=1),
    "f4s2_arch2_partial_attn_bce_s2": dict(F4, seed=2),
    "f8_arch2_full_attn_bce": F8,
    "f5_arch3_adapter_attn_bce": F5,
    "f6_arch1_full_pool_bce": F6,
    "f7_arch2_kd_from_f1": F7,
    "f7s1_arch2_kd_from_f1_s1": dict(F7, seed=1),
    "g1_arch2_partial_e6": G1,
    "g2_arch2_partial_e6_lr1e4": G2,
    "g3_arch2_full_e6": G3,
    "g5_arch2_partial_e6_lr1e4_s1": dict(G2, seed=1),
    "g6_arch2_partial_e6_k8": G6,
    "g4_arch2_kd_e6": G4,
    "g7_arch2_kd_e6_s1": dict(G4, seed=1),
    "g8_arch2_kd_e6_s2": dict(G4, seed=2),
    "g9_arch2_kd_e6_s3": dict(G4, seed=3),
    "g10_arch2_kd_e6_s4": dict(G4, seed=4),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, cases, k, fp16=True):
    model.eval()
    device = next(model.parameters()).device
    out = np.zeros((len(cases), N_LABELS), np.float32)
    for i in range(len(cases)):
        x = torch.tensor(case_chunks(cases[i], model.clf.backbone, k), device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
            out[i] = model(x).float().cpu().numpy()[0]
    return out


def train_cases(model, train, dev, device, epochs=2, accum=16, lr_body=3e-4, lr_head=1e-3, warmup=0.06,
                weight_decay=0.01, clip=1.0, fp16=True, k=4, loss="asl", teacher=None, tau=2.0, kd_weight=0.5,
                eval_every=500, eval_n=2000, max_minutes=600, log=print, seed=0):
    # one case per forward pass, accum cases per update
    set_seed(seed)
    head = []
    body = []
    for n, p in model.named_parameters():
        if n.startswith("clf.backbone."):
            body.append(p)
        else:
            head.append(p)
    opt = torch.optim.AdamW([{"params": head, "lr": lr_head}, {"params": body, "lr": lr_body}],
                            weight_decay=weight_decay)
    total = math.ceil(len(train) / accum) * epochs
    n_warm = int(warmup * total)

    def lr_factor(u):
        # linear warmup, then linear decay to 0
        return min(1.0, (u + 1) / max(1, n_warm)) * max(0.0, (total - u) / max(1, total - n_warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    use_amp = fp16 and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    trainable = [p for p in head + body if p.requires_grad]
    Y = np.stack([c["y"] for c in train]).astype(np.float32)
    dev_sub = random.Random(1).sample(dev, min(eval_n, len(dev)))
    dev_y = np.stack([c["y"] for c in dev_sub])
    hist = []
    best = -1.0
    best_state = None
    t0 = time.time()
    u = 0
    for ep in range(epochs):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        tot = 0.0
        n = 0
        for i in range(len(order)):
            ci = order[i]
            x = torch.tensor(case_chunks(train[ci], model.clf.backbone, k), device=device)
            y = torch.tensor(Y[ci:ci + 1], device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                z = model(x).float()
            if teacher is not None:
                t = torch.tensor(teacher[ci:ci + 1], device=device)
                l, _ = kd_case_loss(z, t, y, tau, kd_weight)
            else:
                l = case_loss(z, y, loss)
            scaler.scale(l / accum).backward()
            tot += l.item()
            n += 1
            last = i == len(order) - 1
            if n % accum == 0 or last:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                u += 1
                if u % eval_every == 0 or last:
                    p = sigmoid(predict(model, dev_sub, k, fp16))
                    model.train()
                    f, thr = best_global(p, dev_y)
                    minutes = (time.time() - t0) / 60
                    avg_loss = tot / n
                    hist.append({"epoch": ep + 1, "update": u, "loss": avg_loss, "minutes": minutes,
                                 "macro_f1_global": f, "thr": thr})
                    if f > best:
                        best = f
                        best_state = {}
                        for name, v in model.named_parameters():
                            if v.requires_grad:
                                best_state[name] = v.detach().cpu().clone()
                    epoch = ep + 1
                    log(f"epoch {epoch} | update {u} | loss {avg_loss:.4f} | minutes {minutes:.1f} | "
                        f"dev macro_f1 (global thr) {f:.4f} | thr {thr:.2f}")
                if (time.time() - t0) / 60 > max_minutes:
                    log("time budget reached")
                    return hist, best, best_state
    return hist, best, best_state


def run(tag, cfg):
    out = f"{OUT_DIR}/{tag}"
    os.makedirs(out, exist_ok=True)
    logf = open(f"{out}/log.txt", "a")

    def log(*a):
        s = " ".join([str(x) for x in a])
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    check_lorma()
    check_nf4()
    check_distillation_loss()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    train = load_tokens(f"{TOKEN_DIR}/train.npz")
    dev = load_tokens(f"{TOKEN_DIR}/dev.npz")
    test = load_tokens(f"{TOKEN_DIR}/test.npz")
    log(tag, "| train", len(train), "dev", len(dev), "test", len(test), "| device", device, json.dumps(cfg))

    torch.manual_seed(cfg.get("seed", 0))
    clf = make_classifier(cfg["arch"], MODEL_DIRS[cfg["arch"]], cfg["bidirectional"], num_classes=N_LABELS)
    clf = attach_peft(clf, cfg["method"], cfg["method_cfg"])
    prior = np.stack([c["y"] for c in train]).mean(0).clip(1e-4, 1 - 1e-4)
    model = CaseModel(clf, cfg["head"], prior).to(device)
    n_train = sum([p.numel() for p in model.parameters() if p.requires_grad])
    n_all = sum([p.numel() for p in model.parameters()])
    train_m = n_train / 1e6
    all_m = n_all / 1e6
    log(cfg["arch"], cfg["method"], "head", cfg["head"], f"| trainable {train_m:.3f}M of {all_m:.1f}M")

    teacher = None
    if cfg.get("teachers"):
        teacher = load_teacher_cache(cfg["teachers"], [c["id"] for c in train])
        log("teacher cache from", cfg["teachers"])

    # arch1 did not learn in fp16 in Phase 3
    fp16 = cfg["arch"] != "arch1"
    k = cfg["chunks"]
    t0 = time.time()
    hist, best, state = train_cases(model, train, dev, device, epochs=cfg["epochs"], lr_body=cfg["lr_body"],
                                    fp16=fp16, k=k, loss=cfg["loss"], teacher=teacher,
                                    eval_every=cfg["eval_every"], eval_n=cfg["eval_n"],
                                    max_minutes=cfg["max_minutes"], log=log, seed=cfg.get("seed", 0))
    model.load_state_dict(state, strict=False)
    minutes = (time.time() - t0) / 60
    log(f"training done in {minutes:.1f} min, best dev macro_f1 (subset, global thr) {best:.4f}")
    if n_train < 30e6:
        torch.save(state, f"{out}/trainable_state.pt")

    dev_y = np.stack([c["y"] for c in dev]).astype(bool)
    logits = {}
    results = {}
    for kk in sorted(set(cfg["ks"])):
        logits[f"dev_k{kk}"] = predict(model, dev, kk, fp16)
        logits[f"test_k{kk}"] = predict(model, test, kk, fp16)
        results[f"k{kk}"] = report(sigmoid(logits[f"dev_k{kk}"]), dev_y)
        r = results[f"k{kk}"]
        f_half = r["f1_0.5"]
        f_global = r["f1_global"]
        f_label = r["crossfit_per_label"]
        f_micro = r["micro_f1"]
        log(f"dev k{kk} | 0.5 {f_half:.4f} | global {f_global:.4f} | crossfit per-label {f_label:.4f} | "
            f"micro {f_micro:.4f}")

    if cfg["teacher_cache"]:
        tl = predict(model, train, k, fp16)
        np.savez(f"{out}/teacher_train.npz", case_ids=np.array([c["id"] for c in train]), logits=tl)
        log("teacher cache for", len(train), "train cases")

    np.savez(f"{out}/scores.npz", dev_ids=np.array([c["id"] for c in dev]),
             test_ids=np.array([c["id"] for c in test]), dev_y=dev_y, **logits)
    pick = None
    for v in results:
        if pick is None or results[v]["crossfit_per_label"] > results[pick]["crossfit_per_label"]:
            pick = v
    thr = per_label(sigmoid(logits[f"dev_{pick}"]), dev_y)
    pred = apply(sigmoid(logits[f"test_{pick}"]), thr)
    write_submission(f"{out}/submission_{tag}_{pick}.zip", [c["id"] for c in test], pred)
    json.dump({"cfg": cfg, "trainable": n_train, "total": n_all, "hist": hist, "dev": results, "pick": pick},
              open(f"{out}/metrics.json", "w"), indent=1)
    best_f1 = results[pick]["crossfit_per_label"]
    log(f"done: pick {pick} crossfit per-label {best_f1:.4f}")


if __name__ == "__main__":
    run(sys.argv[1], RUNS[sys.argv[1]])
