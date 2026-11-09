# continued pretraining and masked next-token training of arch 4
# python adapt.py tapt      continued causal LM pretraining (start: phase 1 arch 4)
# python adapt.py mntp 0    bidirectional adaptation with masked next-token prediction, LLM2Vec style (start: tapt model)
import json
import math
import os
import shutil
import sys
import time

import ijson
import numpy as np
import torch
import torch.nn.functional as F

import model as M
from preprocess import DATA_DIR, encode_docs, is_phase2, load_split, load_tokenizer, phase2_keys

# phase 1 training judgments and the share of the token budget taken from each
CORPUS_FILES = [("data/phase1/train_supreme_court.json", 0.6), ("data/phase1/train_high_courts.json", 0.4)]
ARCH = "arch4"
SEQ = 1024
BATCH, ACCUM = 2, 8
LR = 1e-4
MASK_ID = 1
MASK_FRAC = 0.2

SETTINGS = {
    "tapt": dict(tokens=25000000, updates=1500, start=M.MODEL_DIRS["arch4"], bidir=False),
    "mntp": dict(tokens=14000000, updates=800, start="runs/arch4_tapt", bidir=True),
}


def collect_corpus(pt, need, keys, snips):
    out = []
    for path, share in CORPUS_FILES:
        got = 0
        with open(path, "rb") as fh:
            for k, rec in enumerate(ijson.items(fh, "item")):
                if (k + 5) % 3:
                    continue
                t = " ".join(rec["text"].split())
                if len(t) < 5000 or is_phase2(t, keys, snips):
                    continue
                ids = pt.encode(t)[0]
                out.append(ids)
                got += len(ids)
                if got >= need * share:
                    break
    return out


def run(mode, seed):
    cfg = SETTINGS[mode]
    if mode == "tapt":
        save_dir = "runs/arch4_tapt"
    else:
        save_dir = "runs/arch4_mntp_s" + str(seed)
    os.makedirs(save_dir, exist_ok=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda"
    pt = load_tokenizer()
    texts, _ = load_split(DATA_DIR, ["fold1", "fold2"])
    test_texts, _ = load_split(DATA_DIR, ["fold3"])
    keys, snips = phase2_keys(list(texts.values()) + list(test_texts.values()))

    corpus_ids = collect_corpus(pt, cfg["tokens"], keys, snips)
    pieces = []
    for ids in corpus_ids:
        pieces.append(np.append(ids, 3))
    stream = np.concatenate(pieces).astype(np.int64)
    n_blocks = len(stream) // (SEQ + 1)
    blocks = stream[:n_blocks * (SEQ + 1)].reshape(n_blocks, SEQ + 1)
    print("corpus:", len(corpus_ids), "judgments,", len(stream), "tokens,", n_blocks, "blocks")

    # held-out check: loss on the phase 2 training texts before and after
    docs = {}
    for f in ("fold1", "fold2"):
        ft, _ = load_split(DATA_DIR, [f])
        docs.update(encode_docs(pt, ft))
    eval_stream = np.concatenate([doc["ids"] for doc in docs.values()])
    n_eval = (len(eval_stream) // (SEQ + 1)) * (SEQ + 1)
    eval_blocks = eval_stream[:n_eval].reshape(-1, SEQ + 1)[:120]

    bb = M.Backbone(ARCH, cfg["start"], bidirectional=cfg["bidir"]).to(device)
    W = bb.model.tok_emb.weight
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def causal_loss(x):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            h = bb(x[:, :-1])
        logits = h.float() @ W.float().T
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))

    def mntp_loss(x):
        # 20% of tokens become a mask token; each one is predicted from the output one position before it
        x = x[:, :-1]
        pick = torch.rand(x.shape, generator=gen) < MASK_FRAC
        pick[:, 0] = False
        pick = pick.to(x.device)
        inp = x.masked_fill(pick, MASK_ID)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            h = bb(inp)
        rows, cols = pick.nonzero(as_tuple=True)
        logits = h[rows, cols - 1].float() @ W.float().T
        return F.cross_entropy(logits, x[rows, cols])

    if mode == "tapt":
        lm_loss = causal_loss
    else:
        lm_loss = mntp_loss

    def eval_loss():
        bb.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, len(eval_blocks), BATCH):
                losses.append(lm_loss(torch.from_numpy(eval_blocks[i:i + BATCH]).to(device)).item())
        bb.train()
        return float(np.mean(losses))

    before = eval_loss()
    print(f"loss on phase 2 text before: {before:.4f}")
    opt = torch.optim.AdamW(bb.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    updates = min(cfg["updates"], n_blocks // (BATCH * ACCUM))
    bb.train()
    t0 = time.time()
    pos = 0
    for u in range(updates):
        # 100 warmup updates, then cosine decay to 10% of LR
        lr = LR * min(1.0, (u + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * u / updates)))
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(ACCUM):
            x = torch.from_numpy(blocks[order[pos:pos + BATCH]]).to(device)
            pos += BATCH
            scaler.scale(lm_loss(x) / ACCUM).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(bb.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if (u + 1) % 250 == 0 or u + 1 == updates:
            minutes = (time.time() - t0) / 60
            print(f"update {u + 1}/{updates} | eval loss {eval_loss():.4f} | minutes {minutes:.1f}", flush=True)
    after = eval_loss()
    print(f"loss on phase 2 text: {before:.4f} to {after:.4f}")

    torch.save(bb.model.state_dict(), os.path.join(save_dir, "model.pt"))
    for f in os.listdir(cfg["start"]):
        if f.endswith(".py") or f in ("metrics.json", "arch4_engram_compress.pt"):
            shutil.copy(os.path.join(cfg["start"], f), save_dir)
    with open(os.path.join(save_dir, "adapt_log.json"), "w") as f:
        json.dump({"tokens": int(len(stream)), "updates": updates, "loss_before": before, "loss_after": after}, f, indent=1)


if __name__ == "__main__":
    mode = sys.argv[1]
    seed = 0
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])
    run(mode, seed)
