# Retrieval fine-tune: teach Arch 2 to continue a prompt when a retrieved passage is placed in its context
import os
import json
import math
import time
import random
import hashlib
import ijson
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from generate_mbr import tokenizer, load_arch2, amp_ctx, use_amp, device, PROMPT_KEEP, MAX_NEW_TOKENS, TARGET_WORDS, PAD_ID

DATA_DIR = "data"
RAG_SOURCES = ["train_supreme_court.json", "train_high_courts.json"]
HELDOUT_PATH = os.path.join("outputs", "heldout_eval_set.json")
ARCH2_PATH = os.path.join("outputs", "arch2", "model.pt")
OUTPUT_DIR = os.path.join("outputs", "rag")

SEED = 42
ANCHOR = 12               # the prompt's last 12 words (the retrieval anchor)
RAG_DOC_PROB = 0.5        # sample half the eligible documents
RAG_EXAMPLES = 12000      # one 512 to 128 word window per sampled document
DISTRACT_FRAC = 0.25      # share of examples whose passage comes from another document
RAG_LR = 1e-4
FT_BATCH = 4
FT_ACCUM = 8              # 32 examples per update
IGNORE = -100


def rag_ids(prompt, passage):
    # [prompt without its last 12 words][last 12 words + passage][last 12 words], cut to PROMPT_KEEP tokens
    pw = prompt.split()
    anchor = " ".join(pw[-ANCHOR:])
    b = tokenizer.encode(anchor + " " + passage)
    a = tokenizer.encode(anchor)
    head = tokenizer.encode(" ".join(pw[:-ANCHOR]))[-max(0, PROMPT_KEEP - len(b) - len(a)):]
    return (head + b + a)[-PROMPT_KEEP:]


def build_examples():
    with open(HELDOUT_PATH) as f:
        held = json.load(f)
    held_keys = set()
    for e in held:
        held_keys.add(e["doc_key"])

    rng = random.Random(SEED + 31)
    windows = []
    for train_file in RAG_SOURCES:
        with open(os.path.join(DATA_DIR, train_file), "rb") as fh:
            for rec in ijson.items(fh, "item"):
                text = rec["text"]
                if hashlib.md5(text[:500].encode()).hexdigest() in held_keys:
                    continue
                w = text.split()
                if len(w) < 512 + TARGET_WORDS or rng.random() > RAG_DOC_PROB:
                    continue
                st = rng.randint(0, len(w) - 512 - TARGET_WORDS)
                windows.append((" ".join(w[st:st + 512]), " ".join(w[st + 512:st + 512 + TARGET_WORDS])))
                if len(windows) >= RAG_EXAMPLES:
                    break
        if len(windows) >= RAG_EXAMPLES:
            break

    pairs = []
    for p, c in windows:
        if rng.random() < DISTRACT_FRAC:
            passage = windows[rng.randrange(len(windows))][1]
        else:
            passage = c
        pairs.append((rag_ids(p, passage), tokenizer.encode(c)[:MAX_NEW_TOKENS]))
    tokenizer._encode_cache = {}
    print("RAG fine-tuning examples:", len(pairs))
    return pairs


def finetune(pairs):
    m = load_arch2(ARCH2_PATH).train()
    decay = []
    no_decay = []
    for p in m.parameters():
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=RAG_LR, betas=(0.9, 0.95), eps=1e-8)
    scaler = GradScaler(enabled=use_amp)

    order = np.random.default_rng(SEED + 7).permutation(len(pairs))
    per_upd = FT_BATCH * FT_ACCUM
    total = len(order) // per_upd
    t0 = time.time()
    hist = []
    for u in range(total):
        # 20-update warmup, cosine down to 0.1 x RAG_LR
        lr = RAG_LR * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * u / total)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        chunk = order[u * per_upd:(u + 1) * per_upd]

        micro = []
        for a in range(0, per_upd, FT_BATCH):
            idx = chunk[a:a + FT_BATCH]
            seqs = []
            for j in idx:
                seqs.append(pairs[j][0] + pairs[j][1])
            T = max([len(s) for s in seqs]) - 1
            x = np.full((len(idx), T), PAD_ID, dtype=np.int64)
            y = np.full((len(idx), T), IGNORE, dtype=np.int64)
            for r in range(len(idx)):
                s = seqs[r]
                pl = len(pairs[idx[r]][0])
                x[r, :len(s) - 1] = s[:-1]
                y[r, pl - 1:len(s) - 1] = s[pl:]  # loss on the continuation only
            micro.append((torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)))

        n_valid = 0
        for _, yb in micro:
            n_valid += int((yb != IGNORE).sum())
        ls = 0.0
        for xb, yb in micro:
            with amp_ctx():
                lg = m(xb)
            loss = F.cross_entropy(lg.float().view(-1, lg.size(-1)), yb.reshape(-1), ignore_index=IGNORE, reduction="sum")
            scaler.scale(loss / n_valid).backward()
            ls += loss.item()
            del lg, loss
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        hist.append(ls / n_valid)
        if (u + 1) % 25 == 0 or u == total - 1:
            print(f"update {u + 1}/{total} | continuation loss {np.mean(hist[-25:]):.4f} | lr {lr:.2e} | {(time.time() - t0) / 60:.1f} min")

    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.detach().cpu()
    torch.save(state, os.path.join(OUTPUT_DIR, "rag_ft.pt"))
    with open(os.path.join(OUTPUT_DIR, "rag_ft_log.json"), "w") as f:
        json.dump({"examples": len(pairs), "updates": total, "loss_history": hist}, f, indent=1)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pairs = build_examples()
    finetune(pairs)
