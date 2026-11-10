# training runs and predictions behind the other submissions
# python -m experiments.extra_runs rr1_frozen | kd_overlap | big | fixed_weights | refit | b3 | b4
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import backbone as B
import train as TR
from data import LABELS
from decode import macro_f1, predict
from distill import ARCH2_DIR, LR_BODY, STUDENT_EPOCHS, TEACHER_PATH
from model import HIDDEN, build_model
from predict import RUN_FILES, load_probs, save_probs, write_submission

OUT_DIR = TR.OUT_DIR
KD_MODEL_PATH = OUT_DIR + "/arch2_kd_model.pt"
ROUND2_TEACHER_PATH = OUT_DIR + "/teacher_round2_train_probs.npz"
NEVER = [LABELS.index("Dissent"), LABELS.index("PrecedentOverruled")]

# rr1 with the backbone frozen for the first 2 epochs, Codabench 0.4883
RR1_FROZEN = ("arch4", "lora", True, B.ARCH4_MNTP_DIR, 6, 5e-4, 2)

# name, arch, bidirectional, backbone dir, distilled from rr1 + x2, seed, epochs
BIG_MEMBERS = [
    ("arch2_kd_s1", "arch2", False, ARCH2_DIR, True, 1, 6),
    ("arch2_kd_s2", "arch2", False, ARCH2_DIR, True, 2, 6),
    ("arch2_kd_s3", "arch2", False, ARCH2_DIR, True, 3, 6),
    ("arch4_lora_s1", "arch4", True, B.ARCH4_MNTP_DIR, False, 1, 8),
    ("arch4_lora_s2", "arch4", True, B.ARCH4_MNTP_DIR, False, 2, 8),
]

REFIT_EPOCHS = 6
FIXED_WEIGHT_CAP = 10.0

# contrastive + prototype losses (santosh et al 2024): name -> train on dev too, epochs, knn at prediction
CONTRAST_RUNS = {"b3": (False, 6, True), "b4": (True, 5, False)}
CONTRAST_W = 0.1
PROTO_W = 0.1
CONTRAST_TAU = 0.1
CONTRAST_SCALE = 10.0
PROJ_DIM = 128
N_PROTOS = 3
KNN_K = 16
KNN_TEMP = 0.1
KNN_WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5]


class FeatureHook:
    # keeps the [role ; shift] features of the last forward pass
    def __init__(self, model):
        self.role = None
        self.shift = None
        model.role_lstm.register_forward_hook(self.save_role)
        model.shift_lstm.register_forward_hook(self.save_shift)

    def save_role(self, module, inputs, output):
        self.role = output[0][0]

    def save_shift(self, module, inputs, output):
        self.shift = output[0][0]

    def features(self):
        return torch.cat([self.role, self.shift], dim=1)


def contrastive_loss(z, y):
    # supervised contrastive, same label pairs weighted by exp(-distance / scale)
    z = F.normalize(z, dim=1)
    n = len(y)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = (z @ z.T / CONTRAST_TAU).masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    idx = torch.arange(n, device=z.device)
    dist = (idx[:, None] - idx[None, :]).abs().float()
    w = torch.exp(-dist / CONTRAST_SCALE) * ((y[:, None] == y[None, :]) & ~eye).float()
    has = w.sum(dim=1) > 0
    if not has.any():
        return z.sum() * 0
    return (-(w * log_prob).sum(dim=1)[has] / w.sum(dim=1)[has]).mean()


def prototype_loss(z, y, protos):
    # pull each sentence to the nearest prototype of its label
    z = F.normalize(z, dim=1)
    p = F.normalize(protos, dim=-1)
    sims = torch.einsum("nd,ckd->nck", z, p).max(dim=2).values / CONTRAST_TAU
    return F.cross_entropy(sims, y)


def fixed_class_weights(docs):
    # inverse sqrt frequency, average weight over the training sentences = 1
    all_labels = []
    for d in docs:
        all_labels += d["labels"]
    counts = np.bincount(all_labels, minlength=len(LABELS)).astype(float)
    prior = counts / counts.sum()
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w = w / (prior * w).sum()
    w = np.minimum(w, FIXED_WEIGHT_CAP)
    w = w / (prior * w).sum()
    return torch.tensor(w).float()


def train_variant(model, fit_docs, dev_docs, device, epochs, soft, seed=0, weighting="torch", hook=None, keep_last=False):
    # train.train with fixed class weights, the contrastive losses, or no dev selection (dev docs in the training data)
    random.seed(seed)
    torch.manual_seed(seed)
    model.to(device)
    body = []
    head = []
    for n, p in model.named_parameters():
        if n.startswith("backbone."):
            if p.requires_grad:
                body.append(p)
        else:
            head.append(p)
    opt = torch.optim.AdamW([{"params": body, "lr": LR_BODY}, {"params": head, "lr": TR.LR_HEAD}], weight_decay=0.01)
    if weighting == "fixed":
        weights = fixed_class_weights(fit_docs).to(device)
    else:
        weights = TR.class_weights(fit_docs).to(device)
    if soft is not None:
        TR.check_teacher(soft, fit_docs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    if model.context == "none":
        pieces = TR.make_pieces(fit_docs, TR.SEGMENT, TR.PIECE_TOKENS)
    else:
        pieces = TR.make_pieces(fit_docs, TR.SEGMENT, TR.PIECE_TOKENS // 2)
    total_steps = epochs * len(pieces)
    warmup = max(1, total_steps // 20)

    def lr_factor(step):
        return min(1.0, (step + 1) / warmup) * (1 - step / total_steps)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    best_f1 = -1
    best_state = None
    for epoch in range(epochs):
        model.train()
        random.shuffle(pieces)
        total = 0
        for d, a, b in pieces:
            y = torch.tensor(d["labels"][a:b]).to(device)
            E, S = model(d, a, b)
            crf = model.crf.loss(E, y) / len(y)
            if weighting == "fixed":
                ce = (F.cross_entropy(E, y, reduction="none") * weights[y]).mean()
            else:
                ce = F.cross_entropy(E, y, weight=weights)
            loss = 0.4 * crf + 0.3 * ce
            if len(y) > 1:
                changed = (y[:-1] != y[1:]).float()
                loss = loss + 0.6 * F.binary_cross_entropy_with_logits(S[:-1], changed)
            if hook is not None:
                z = model.proj(hook.features())
                loss = loss + CONTRAST_W * contrastive_loss(z, y) + PROTO_W * prototype_loss(z, y, model.protos)
            if soft is not None and d["id"] in soft:
                teacher = torch.tensor(np.asarray(soft[d["id"]][a:b], dtype=np.float32)).to(device)
                loss = 0.5 * loss + 0.5 * TR.kd_loss(E, teacher)
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
        if keep_last:
            print(f"epoch {epoch + 1} | loss {total / len(pieces):.4f}", flush=True)
            continue
        gold = []
        preds = []
        for doc in dev_docs:
            gold.append(doc["labels"])
            preds.append(predict(model, doc)[0])
        dev_f1 = macro_f1(gold, preds)
        print(f"epoch {epoch + 1} | loss {total / len(pieces):.4f} | dev macro F1 {dev_f1:.4f}", flush=True)
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_state = {}
            for k, v in model.state_dict().items():
                best_state[k] = v.detach().cpu().clone()
    if not keep_last:
        model.load_state_dict(best_state)


@torch.no_grad()
def masked_viterbi(model, doc):
    # crf viterbi with Dissent and PrecedentOverruled never predicted, as in the first submissions
    model.eval()
    E, _ = model(doc, 0, len(doc["sents"]))
    E = E.float()
    E[:, NEVER] = -10000
    return model.crf.viterbi(E)


def masked_argmax(p):
    s = np.log(np.clip(p, 1e-9, 1.0))
    s[:, NEVER] = -1e9
    return s.argmax(axis=1).tolist()


def run_rr1_frozen(device):
    TR.RUNS["rr1_frozen"] = RR1_FROZEN
    TR.run("rr1_frozen", device)


def run_kd_overlap(device):
    # distilled arch2 predicted with overlapping windows (no retraining), Codabench 0.5444
    # also saves the shift head probabilities used by the smoothing submissions
    train_docs, dev_docs, test_docs = TR.load_splits(B.load_tokenizer())
    model = build_model("arch2", ARCH2_DIR, "lora", False, context="centered")
    model.load_state_dict(torch.load(KD_MODEL_PATH, map_location="cpu"))
    model.to(device)
    model.eval()
    preds = []
    for d in test_docs:
        preds.append(masked_viterbi(model, d))
    write_submission(test_docs, preds, OUT_DIR + "/submission_arch2_kd_overlap.zip")
    for split, docs in [("dev", dev_docs), ("test", test_docs)]:
        shift = []
        with torch.no_grad():
            for d in docs:
                _, S = model(d, 0, len(d["sents"]))
                shift.append(torch.sigmoid(S.float()).cpu().numpy())
        save_probs(OUT_DIR + "/arch2_kd_" + split + "_shift.npz", docs, shift)


def fit_big_member(name, arch, bidir, model_dir, soft, seed, epochs, splits, device):
    torch.manual_seed(seed)
    model = build_model(arch, model_dir, "lora", bidir, context="none").to(device)
    TR.train(model, splits[0][1], splits[1][1], device, epochs=epochs, lr_body=LR_BODY, soft=soft, seed=seed)
    model.set_context("centered")
    out = {}
    for split, docs in splits:
        probs = []
        for d in docs:
            probs.append(predict(model, d)[1])
        save_probs(OUT_DIR + "/big_" + name + "_" + split + "_probs_centered.npz", docs, probs)
        out[split] = probs
    del model
    torch.cuda.empty_cache()
    return out


def run_big(device):
    # five more runs (three distilled arch2 seeds, two arch4 lora seeds) and an arch2 student distilled from their average
    train_docs, dev_docs, test_docs = TR.load_splits(B.load_tokenizer())
    splits = [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]
    teacher = load_probs(TEACHER_PATH)
    member_probs = []
    for name, arch, bidir, model_dir, use_teacher, seed, epochs in BIG_MEMBERS:
        soft = None
        if use_teacher:
            soft = teacher
        member_probs.append(fit_big_member(name, arch, bidir, model_dir, soft, seed, epochs, splits, device)["train"])
    student_soft = {}
    for i in range(len(train_docs)):
        p = 0
        for probs in member_probs:
            p = p + probs[i]
        p = p / len(member_probs)
        student_soft[train_docs[i]["id"]] = p / p.sum(axis=1, keepdims=True)
    fit_big_member("student", "arch2", False, ARCH2_DIR, student_soft, 0, STUDENT_EPOCHS, splits, device)


def run_fixed_weights(device):
    # distilled arch2 with per-sentence class weights and a fixed denominator
    train_docs, dev_docs, test_docs = TR.load_splits(B.load_tokenizer())
    soft = load_probs(TEACHER_PATH)
    torch.manual_seed(0)
    model = build_model("arch2", ARCH2_DIR, "lora", False, context="none").to(device)
    train_variant(model, train_docs, dev_docs, device, STUDENT_EPOCHS, soft, weighting="fixed")
    TR.save_run(model, "fixed_weights", [("train", train_docs), ("dev", dev_docs), ("test", test_docs)])


def run_refit(device):
    # distilled arch2 refit on train + dev, fixed number of epochs, last epoch kept
    train_docs, dev_docs, test_docs = TR.load_splits(B.load_tokenizer())
    soft = load_probs(TEACHER_PATH)
    torch.manual_seed(0)
    model = build_model("arch2", ARCH2_DIR, "lora", False, context="none").to(device)
    train_variant(model, train_docs + dev_docs, dev_docs, device, REFIT_EPOCHS, soft, keep_last=True)
    TR.save_run(model, "refit", [("train", train_docs), ("dev", dev_docs), ("test", test_docs)])


def make_round2_teacher(train_docs):
    # average of rr1, x2 and the distilled arch2 on the training docs
    files = [RUN_FILES["rr1"].format("train"), RUN_FILES["x2"].format("train"), OUT_DIR + "/arch2_kd_train_probs.npz"]
    members = []
    for f in files:
        members.append(load_probs(f))
    soft = {}
    probs = []
    for d in train_docs:
        p = 0
        for m in members:
            q = np.asarray(m[d["id"]], dtype=np.float64)
            p = p + q / q.sum(axis=1, keepdims=True)
        p = p / len(members)
        soft[d["id"]] = p
        probs.append(p)
    save_probs(ROUND2_TEACHER_PATH, train_docs, probs)
    return soft


@torch.no_grad()
def sentence_features(model, hook, docs):
    model.eval()
    out = []
    for d in docs:
        model(d, 0, len(d["sents"]))
        out.append(F.normalize(hook.features().float(), dim=1))
    return out


@torch.no_grad()
def knn_probs(bank, bank_labels, feats):
    # label distribution of the k most similar training sentences (cosine), weighted by similarity
    bank = torch.cat(bank)
    all_labels = []
    for ys in bank_labels:
        all_labels += ys
    labels = torch.tensor(all_labels, device=bank.device)
    out = []
    for f in feats:
        sim, idx = (f @ bank.T).topk(KNN_K, dim=1)
        w = torch.softmax(sim / KNN_TEMP, dim=1)
        p = torch.zeros(len(f), len(LABELS), device=f.device)
        p.scatter_add_(1, labels[idx], w)
        out.append(p.cpu().numpy())
    return out


def mix(probs, knn, lam):
    out = []
    for p, k in zip(probs, knn):
        out.append((1 - lam) * p + lam * k)
    return out


def knn_submission(model, hook, fit_docs, dev_docs, test_docs, name):
    # crf viterbi, or the knn mix if one of the knn weights is better on dev
    gold = []
    crf_dev = []
    probs_dev = []
    for d in dev_docs:
        gold.append(d["labels"])
        crf_dev.append(masked_viterbi(model, d))
        probs_dev.append(predict(model, d)[1])
    preds = []
    probs_test = []
    for d in test_docs:
        preds.append(masked_viterbi(model, d))
        probs_test.append(predict(model, d)[1])
    bank_labels = []
    for d in fit_docs:
        bank_labels.append(d["labels"])
    bank = sentence_features(model, hook, fit_docs)
    knn_dev = knn_probs(bank, bank_labels, sentence_features(model, hook, dev_docs))
    knn_test = knn_probs(bank, bank_labels, sentence_features(model, hook, test_docs))
    best = macro_f1(gold, crf_dev)
    for lam in KNN_WEIGHTS:
        dev_preds = []
        for p in mix(probs_dev, knn_dev, lam):
            dev_preds.append(masked_argmax(p))
        f1 = macro_f1(gold, dev_preds)
        print("knn weight", lam, "dev macro F1", round(f1, 4))
        if f1 > best:
            best = f1
            preds = []
            for p in mix(probs_test, knn_test, lam):
                preds.append(masked_argmax(p))
    write_submission(test_docs, preds, OUT_DIR + "/submission_" + name + ".zip")


def run_contrastive(name, device):
    # b3: round-2 teacher + contrastive and prototype losses + knn, b4: the same on train + dev without knn
    on_dev, epochs, knn = CONTRAST_RUNS[name]
    train_docs, dev_docs, test_docs = TR.load_splits(B.load_tokenizer())
    soft = make_round2_teacher(train_docs)
    torch.manual_seed(0)
    model = build_model("arch2", ARCH2_DIR, "lora", False, context="centered")
    model.proj = nn.Linear(4 * HIDDEN, PROJ_DIM)
    model.protos = nn.Parameter(torch.randn(len(LABELS), N_PROTOS, PROJ_DIM) * 0.1)
    hook = FeatureHook(model)
    fit_docs = train_docs
    if on_dev:
        fit_docs = train_docs + dev_docs
    train_variant(model, fit_docs, dev_docs, device, epochs, soft, hook=hook, keep_last=on_dev)
    TR.save_run(model, name, [("train", train_docs), ("dev", dev_docs), ("test", test_docs)])
    model.set_context("centered")
    if knn:
        knn_submission(model, hook, fit_docs, dev_docs, test_docs, name)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    step = sys.argv[1]
    if step == "rr1_frozen":
        run_rr1_frozen(device)
    elif step == "kd_overlap":
        run_kd_overlap(device)
    elif step == "big":
        run_big(device)
    elif step == "fixed_weights":
        run_fixed_weights(device)
    elif step == "refit":
        run_refit(device)
    elif step in CONTRAST_RUNS:
        run_contrastive(step, device)
    else:
        print("unknown step", step)
