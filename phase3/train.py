# training of the rhetorical role model, and the single-model runs (python train.py rr1)
# loss = 0.4 crf + 0.6 shift (malik et al) + 0.3 class-weighted ce, half kd when a teacher is given
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

import backbone as B
from data import DATA_DIR, LABELS, load_docs, tokenize
from decode import macro_f1, predict
from model import build_model
from peft_methods import LoRMALinear
from predict import save_probs, write_submission

OUT_DIR = "runs"
SEGMENT = 64  # max sentences per training piece
PIECE_TOKENS = 4096  # max tokens per training piece (2048 with overlapping windows)
LR_HEAD = 1e-3
KD_T = 2.0

# name: arch, method, bidirectional, backbone dir, epochs, lr_body, head_only_epochs (None means 2 for lorma, else 0)
RUNS = {
    "rr1": ("arch4", "lora", True, B.ARCH4_MNTP_DIR, 6, 5e-4, None),  # Arch 4 bidirectional + LoRA, Codabench 0.5139
    "x2": ("arch4", "lora", True, B.ARCH4_MNTP_P3_DIR, 15, 5e-4, 2),  # Arch 4 longer adaptation + LoRA, 0.5129
    "rr2b": ("arch4", "lorma", True, B.ARCH4_MNTP_DIR, 6, 5e-4, None),  # Arch 4 bidirectional + LoRMA+, 0.4739
    "rr3": ("arch3", "ia3", False, os.path.join(B.KIT_DIR, "arch3"), 6, 3e-3, None),  # Arch 3 + IA3, 0.4479
    "rr2": ("arch1", "lorma", False, os.path.join(B.KIT_DIR, "arch1"), 6, 5e-4, None),  # Arch 1 + LoRMA+, 0.1103
}


def class_weights(docs):
    # sqrt inverse frequency, no zero weights since a piece with only that label gives nan
    all_labels = []
    for d in docs:
        all_labels += d["labels"]
    counts = np.bincount(all_labels, minlength=len(LABELS))
    w = np.ones(len(LABELS))
    for l in range(len(LABELS)):
        if counts[l] > 0:
            w[l] = (counts.sum() / counts[l]) ** 0.5
    w = w / w.mean()
    return torch.tensor(np.minimum(w, 10.0)).float()


def check_teacher(soft, docs):
    for d in docs:
        if d["id"] not in soft:
            continue
        p = np.asarray(soft[d["id"]], dtype=np.float64)
        assert p.shape == (len(d["sents"]), len(LABELS)), d["id"]
        assert (p >= 0).all() and np.abs(p.sum(axis=1) - 1).max() < 1e-2, d["id"]


def kd_loss(E, teacher):
    # student softened at T = 2 against the teacher crf marginals, times T^2
    p = teacher.float().clamp_min(1e-8)
    p = p / p.sum(dim=-1, keepdim=True)
    return F.kl_div(F.log_softmax(E / KD_T, dim=-1), p, reduction="batchmean") * KD_T * KD_T


def make_pieces(docs, seg, piece_tokens):
    pieces = []
    for d in docs:
        start = 0
        used = 0
        for i in range(len(d["sents"])):
            if i > start and (i - start == seg or used + len(d["toks"][i]) > piece_tokens):
                pieces.append((d, start, i))
                start = i
                used = 0
            used += len(d["toks"][i])
        pieces.append((d, start, len(d["sents"])))
    return pieces


def train(model, train_docs, dev_docs, device, epochs=6, lr_body=1e-4, lr_head=LR_HEAD, seg=SEGMENT, soft=None, seed=0,
          head_only_epochs=None):
    train_ids = set()
    for d in train_docs:
        train_ids.add(d["id"])
    for d in dev_docs:
        assert d["id"] not in train_ids, "dev docs in training data"
    random.seed(seed)
    torch.manual_seed(seed)
    model.to(device)
    if head_only_epochs is None:
        # lorma did not learn when trained with the head from the start, so its backbone is frozen for 2 epochs
        head_only_epochs = 0
        for m in model.modules():
            if isinstance(m, LoRMALinear):
                head_only_epochs = 2
    body = []
    head = []
    for n, p in model.named_parameters():
        if n.startswith("backbone."):
            if p.requires_grad:
                body.append(p)
        else:
            head.append(p)
    opt = torch.optim.AdamW([{"params": body, "lr": lr_body}, {"params": head, "lr": lr_head}], weight_decay=0.01)
    weights = class_weights(train_docs).to(device)
    if soft is not None:
        check_teacher(soft, train_docs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    if model.context == "none":
        pieces = make_pieces(train_docs, seg, PIECE_TOKENS)
    else:
        pieces = make_pieces(train_docs, seg, PIECE_TOKENS // 2)
    total_steps = epochs * len(pieces)
    warmup = max(1, total_steps // 20)

    def lr_factor(step):
        return min(1.0, (step + 1) / warmup) * (1 - step / total_steps)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    history = []
    best_f1 = -1
    best_state = None
    for epoch in range(epochs):
        for p in body:
            p.requires_grad = epoch >= head_only_epochs
        model.train()
        random.shuffle(pieces)
        total = 0
        for d, a, b in pieces:
            y = torch.tensor(d["labels"][a:b]).to(device)
            E, S = model(d, a, b)
            crf = model.crf.loss(E, y) / len(y)
            ce = F.cross_entropy(E, y, weight=weights)
            loss = 0.4 * crf + 0.3 * ce
            if len(y) > 1:
                changed = (y[:-1] != y[1:]).float()
                loss = loss + 0.6 * F.binary_cross_entropy_with_logits(S[:-1], changed)
            if soft is not None and d["id"] in soft:
                teacher = torch.tensor(np.asarray(soft[d["id"]][a:b], dtype=np.float32)).to(device)
                loss = 0.5 * loss + 0.5 * kd_loss(E, teacher)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            params = []
            for p in body + head:
                if p.grad is not None:
                    params.append(p)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        gold = []
        preds = []
        for doc in dev_docs:
            gold.append(doc["labels"])
            preds.append(predict(model, doc)[0])
        dev_f1 = macro_f1(gold, preds)
        print(f"epoch {epoch + 1} | loss {total / len(pieces):.4f} | dev macro F1 {dev_f1:.4f}", flush=True)
        history.append({"epoch": epoch + 1, "loss": total / len(pieces), "dev_f1": dev_f1})
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_state = {}
            for k, v in model.state_dict().items():
                best_state[k] = v.detach().cpu().clone()
    model.load_state_dict(best_state)
    return history


def save_run(model, name, splits, out_dir=OUT_DIR):
    # probabilities with the training windows (context none) and with overlapping windows (centered, left)
    os.makedirs(out_dir, exist_ok=True)
    for split, docs in splits:
        probs = []
        for d in docs:
            probs.append(predict(model, d)[1])
        save_probs(out_dir + "/" + name + "_" + split + "_probs.npz", docs, probs)
    for ctx in ["centered", "left"]:
        model.set_context(ctx)
        for split, docs in splits:
            if split == "train":
                continue
            probs = []
            for d in docs:
                probs.append(predict(model, d)[1])
            save_probs(out_dir + "/" + name + "_" + split + "_probs_" + ctx + ".npz", docs, probs)
    model.set_context("none")


def trainable_state(model):
    out = {}
    for k, v in model.state_dict().items():
        if not k.startswith("backbone."):
            out[k] = v.cpu()
        elif "lora_" in k or "ia3_" in k or ".A" in k or ".B" in k:
            out[k] = v.cpu()
    return out


def load_splits(tokenizer):
    train_docs = tokenize(tokenizer, load_docs(DATA_DIR, "train"))
    dev_docs = tokenize(tokenizer, load_docs(DATA_DIR, "dev"))
    test_docs = tokenize(tokenizer, load_docs(DATA_DIR, "test"))
    return train_docs, dev_docs, test_docs


def run(name, device):
    arch, method, bidir, model_dir, epochs, lr_body, head_only = RUNS[name]
    tokenizer = B.load_tokenizer()
    train_docs, dev_docs, test_docs = load_splits(tokenizer)
    torch.manual_seed(0)
    model = build_model(arch, model_dir, method, bidir, context="none").to(device)
    train(model, train_docs, dev_docs, device, epochs=epochs, lr_body=lr_body, seed=0, head_only_epochs=head_only)
    splits = [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]
    save_run(model, name, splits)
    preds = []
    for d in test_docs:
        preds.append(predict(model, d)[0])
    write_submission(test_docs, preds, OUT_DIR + "/submission_" + name + ".zip")
    torch.save({"weights": trainable_state(model), "arch": arch, "method": method, "backbone": model_dir},
               OUT_DIR + "/" + name + "_model.pt")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(sys.argv[1], device)
