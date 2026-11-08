# abstractive variant (g1): arch 2 with its lm head + lorma writes the headnote start from the selected sentences
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from bert_score import score as bert_score

import model as M
import peft_methods
from blend import select
from data import load_docs, TOKENIZER_PATH
from predict import write_prediction
from tokenizer import BPETokenizer, PieceTokenizer

ARCH2_DIR = "models/arch2"
SELECTOR = "work/runs/s1_arch2_lorma/scores.npz"
OUT_DIR = "work/runs/g1_arch2_lorma_abstractive"
DEVICE = "cuda"

SRC_TOKENS = 600
SUM_TOKENS = 400
EOS = 3
EPOCHS = 2
LR = 3e-4
BATCH = 4
MAX_MINUTES = 150
DEV_N = 100


def load_lm(model_dir):
    mod = M.import_arch(model_dir)
    model = mod.DecoderLM(mod.LMConfig(**M.load_config(model_dir)))
    sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    model.load_state_dict(sd, strict=False)
    return model


def source_ids(doc, chosen):
    ids = []
    for i in sorted(chosen):
        ids += doc["toks"][i]
    return ids[:SRC_TOKENS]


def make_examples(train_docs, pt):
    # source = oracle sentences, target = first 400 tokens of the headnote
    examples = []
    for d in train_docs:
        chosen = []
        for i in range(len(d["labels"])):
            if d["labels"][i]:
                chosen.append(i)
        if not chosen:
            continue
        summ = pt.encode(d["summary"])[:SUM_TOKENS].tolist()
        examples.append((source_ids(d, chosen), summ))
    return examples


def make_batch(rows_in):
    rows, masks = [], []
    for src, summ in rows_in:
        rows.append(src + [EOS] + summ + [EOS])
        # loss only on the summary tokens
        masks.append([0] * (len(src) + 1) + [1] * (len(summ) + 1))
    T = 0
    for r in rows:
        T = max(T, len(r))
    x = torch.zeros(len(rows), T, dtype=torch.long)
    m = torch.zeros(len(rows), T)
    for r in range(len(rows)):
        x[r, :len(rows[r])] = torch.tensor(rows[r])
        m[r, :len(masks[r])] = torch.tensor(masks[r], dtype=torch.float32)
    return x, m


def train_lm(model, examples, params):
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    random.seed(0)
    t0 = time.time()
    step = 0
    for ep in range(EPOCHS):
        random.shuffle(examples)
        for k in range(0, len(examples) - BATCH + 1, BATCH):
            x, m = make_batch(examples[k:k + BATCH])
            x, m = x.to(DEVICE), m.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x[:, :-1])
            loss_all = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none")
            target_mask = m[:, 1:].reshape(-1)
            loss = (loss_all * target_mask).sum() / target_mask.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 200 == 0:
                minutes = (time.time() - t0) / 60
                print(f"epoch {ep} | step {step} | loss {loss.item():.4f} | minutes {minutes:.1f}", flush=True)
            if (time.time() - t0) / 60 > MAX_MINUTES:
                break


@torch.no_grad()
def generate(model, tok, src):
    # greedy decoding, no 3-gram is repeated
    model.eval()
    ids = src + [EOS]
    out = []
    for _ in range(SUM_TOKENS):
        x = torch.tensor([ids[-1024:]]).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)[0, -1].float()
        if len(out) >= 2:
            seen = set()
            for j in range(len(out) - 2):
                if out[j] == out[-2] and out[j + 1] == out[-1]:
                    seen.add(out[j + 2])
            for t in seen:
                logits[t] = -1e9
        nxt = int(logits.argmax())
        if nxt == EOS:
            break
        out.append(nxt)
        ids.append(nxt)
    return tok.decode(out)


def top_sentences(doc, scores):
    # selector's best sentences up to SRC_TOKENS tokens
    order = np.argsort(-np.asarray(scores))
    chosen, used = [], 0
    for i in order:
        chosen.append(int(i))
        used += len(doc["toks"][i])
        if used >= SRC_TOKENS:
            break
    return chosen


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = BPETokenizer.load(TOKENIZER_PATH)
    pt = PieceTokenizer(tok)
    train_docs, dev_docs, test_docs = load_docs("train"), load_docs("dev"), load_docs("test")
    examples = make_examples(train_docs, pt)
    print("training examples", len(examples))

    model = load_lm(ARCH2_DIR)
    for p in model.parameters():
        p.requires_grad = False
    peft_methods.add_lorma(model.blocks)
    model = model.to(DEVICE)
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    train_lm(model, examples, params)

    sel = np.load(SELECTOR, allow_pickle=True)
    dev_docs = dev_docs[:DEV_N]
    dev_sel = list(sel["dev"])[:DEV_N]
    test_sel = list(sel["test"])

    dev_gen, refs, extract = [], [], []
    for k in range(len(dev_docs)):
        d = dev_docs[k]
        dev_gen.append(generate(model, tok, source_ids(d, top_sentences(d, dev_sel[k]))))
        refs.append(d["summary"])
        extract.append(select(d["sents"], dev_sel[k], 600))
    F_ext = bert_score(extract, refs, lang="en", batch_size=16, verbose=False)[2]
    F_gen = bert_score(dev_gen, refs, lang="en", batch_size=16, verbose=False)[2]
    print(f"dev bertscore extractive {float(F_ext.mean()):.4f} | generated {float(F_gen.mean()):.4f}")

    preds = []
    for k in range(len(test_docs)):
        d = test_docs[k]
        g = generate(model, tok, source_ids(d, top_sentences(d, test_sel[k])))
        if not g.strip():
            g = d["sents"][0]
        preds.append({"id": d["id"], "summary": g})
    write_prediction(preds, os.path.join(OUT_DIR, "submission_g1_arch2_abstractive_generated.zip"))


if __name__ == "__main__":
    main()
