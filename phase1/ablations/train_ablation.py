# trains and evaluates one ablation run: python train_ablation.py <tag>
import os
import sys
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA_DIR = os.path.join(HERE, "abl_data")
OUTPUT_DIR = os.path.join(HERE, "runs")
CONFIG_FILE = os.path.join(HERE, "configs.json")

VOCAB_SIZE = 16384
SEQ_LEN = 512
BATCH_SIZE = 16
GRAD_ACCUM = 2
TRAIN_TOKENS = 48_000_000
PEAK_LR = 1e-3
MIN_LR = 1e-4
WARMUP_STEPS = 150
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
DECAY_FRAC = 0.2
ENGRAM_LR_MULT = 5.0
N_EVALS = 10
DATA_SEED = 1234
MAX_MINUTES = 75
LOG_EVERY = 50

TAG = sys.argv[1]
cfg = None
for c in json.load(open(CONFIG_FILE)):
    if c["tag"] == TAG:
        cfg = c
assert cfg is not None, "unknown run tag"
GROUP = cfg["group"]
SEED = cfg.get("seed", 42)
OUT = os.path.join(OUTPUT_DIR, TAG)
os.makedirs(OUT, exist_ok=True)

logf = open(os.path.join(OUT, "log.txt"), "a")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


device = "cuda" if torch.cuda.is_available() else "cpu"

train = np.load(os.path.join(DATA_DIR, "train.npy"))
val = np.load(os.path.join(DATA_DIR, "val.npy"))
compress = np.load(os.path.join(DATA_DIR, "engram_compress.npy"))
log("data " + json.dumps(json.load(open(os.path.join(DATA_DIR, "meta.json")))))

BLOCK = SEQ_LEN + 1
n_train = len(train) // BLOCK
n_val = len(val) // BLOCK
train_blocks = torch.from_numpy(train[:n_train * BLOCK].astype(np.int64)).view(n_train, BLOCK)
val_blocks = torch.from_numpy(val[:n_val * BLOCK].astype(np.int64)).view(n_val, BLOCK)
# same data order for every run
perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(DATA_SEED))

tokens_per_update = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN
total_updates = min(TRAIN_TOKENS // tokens_per_update, n_train // (BATCH_SIZE * GRAD_ACCUM))
eval_every = max(1, total_updates // N_EVALS)

torch.manual_seed(SEED)
if GROUP == "A":
    from arch4_model import LMConfig, DecoderLM, RMSNorm
    mcfg = LMConfig(
        vocab_size=VOCAB_SIZE, d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        kv_latent_dim=cfg["kv_latent"], rope_dim=cfg["rope_dim"], d_ff=cfg["d_ff"], max_seq_len=SEQ_LEN, dropout=0.0,
        engram_layers=tuple(cfg["engram_layers"]), engram_orders=(2, 3), engram_heads=4, engram_head_dim=32,
        engram_rows=cfg["engram_rows"], qk_norm=cfg["qk_norm"], gated=cfg["gated"], latent=cfg["latent"],
    )
    model = DecoderLM(mcfg).to(device)
    model.set_engram_compression(torch.from_numpy(compress).long())
    model.set_attention_backend(True)
else:
    from arch2_model import LMConfig, DecoderLM, RMSNorm
    mcfg = LMConfig(
        vocab_size=VOCAB_SIZE, d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"], d_ff=cfg["d_ff"], max_seq_len=SEQ_LEN, dropout=0.0,
        pos_emb_type=cfg["pos_emb_type"], norm_type=cfg["norm_type"], ffn_type=cfg["ffn_type"],
        scale_emb=cfg.get("scale_emb", False),
    )
    model = DecoderLM(mcfg).to(device)


def init_weights(module):
    if module is model.head:
        return
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, RMSNorm):
        torch.nn.init.ones_(module.weight)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)


model.apply(init_weights)

# smaller init for the layers that write into the residual stream
residual_std = 0.02 / math.sqrt(2 * cfg["n_layers"])
engram_params = 0
for block in model.blocks:
    torch.nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
    if GROUP == "B" and mcfg.ffn_type == "gelu":
        torch.nn.init.normal_(block.ffn.net[2].weight, mean=0.0, std=residual_std)
    else:
        torch.nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=residual_std)
    if GROUP == "A" and block.engram is not None:
        # engram starts as an exact no-op
        torch.nn.init.zeros_(block.engram.w_v.weight)
        torch.nn.init.zeros_(block.engram.conv.weight)
        for p in block.engram.parameters():
            engram_params += p.numel()

n_params = 0
for p in model.parameters():
    n_params += p.numel()

decay_params = []
no_decay_params = []
table_params = []
for name, param in model.named_parameters():
    if GROUP == "A" and name.endswith(".engram.table.weight"):
        table_params.append(param)
    elif param.dim() >= 2:
        decay_params.append(param)
    else:
        no_decay_params.append(param)
groups = [{"params": decay_params, "weight_decay": WEIGHT_DECAY, "lr_mult": 1.0},
          {"params": no_decay_params, "weight_decay": 0.0, "lr_mult": 1.0}]
if table_params:
    groups.append({"params": table_params, "weight_decay": 0.0, "lr_mult": ENGRAM_LR_MULT})
optimizer = torch.optim.AdamW(groups, lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8)

decay_start = total_updates - int(DECAY_FRAC * total_updates)


def get_lr(step):
    if step < WARMUP_STEPS:
        return PEAK_LR * step / WARMUP_STEPS
    # group A: warmup, stable, then cosine decay over the last 20%
    if GROUP == "A":
        if step < decay_start:
            return PEAK_LR
        p = min(1.0, (step - decay_start) / max(1, total_updates - decay_start))
        return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * p)) * (PEAK_LR - MIN_LR)
    # group B: warmup, then cosine
    decay_ratio = (step - WARMUP_STEPS) / max(1, total_updates - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (PEAK_LR - MIN_LR)


def amp_ctx():
    return torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda")


def evaluate():
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for i in range(0, n_val, BATCH_SIZE):
            x = val_blocks[i:i + BATCH_SIZE].to(device)
            with amp_ctx():
                logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), x[:, 1:].reshape(-1), reduction="sum")
            loss_sum += loss.item()
            count += x[:, 1:].numel()
    model.train()
    return loss_sum / count


scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
log("run " + TAG + " " + json.dumps(cfg))
backbone_params = n_params - engram_params
val_tokens = n_val * SEQ_LEN
log(f"params {n_params} engram {engram_params} backbone {backbone_params} | updates {total_updates} x "
    f"{tokens_per_update} tokens | val tokens {val_tokens} | device {device}")

history = []
model.train()
t0 = time.time()
train_time = 0.0
skipped = 0
stopped_early = False
loss_window = []
for step in range(total_updates):
    ts = time.time()
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg["lr_mult"]
    accum_loss = 0.0
    for micro in range(GRAD_ACCUM):
        start = (step * GRAD_ACCUM + micro) * BATCH_SIZE
        x = train_blocks[perm[start:start + BATCH_SIZE]].to(device, non_blocking=True)
        with amp_ctx():
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), x[:, 1:].reshape(-1))
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        accum_loss += loss.item()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    # a drop in the loss scale means this fp16 step was skipped
    if scaler.get_scale() < old_scale:
        skipped += 1
    optimizer.zero_grad(set_to_none=True)
    train_time += time.time() - ts
    loss_window.append(accum_loss)

    done = step + 1
    tokens_done = done * tokens_per_update
    if done % LOG_EVERY == 0:
        avg = sum(loss_window) / len(loss_window)
        loss_window = []
        tok_s = tokens_done / train_time
        minutes = (time.time() - t0) / 60
        log(f"step {done} | train {avg:.4f} | lr {lr:.2e} | {minutes:.1f} min | {tok_s:.0f} tok/s")
    if done % eval_every == 0 or done == total_updates:
        vl = evaluate()
        minutes = (time.time() - t0) / 60
        history.append({"step": done, "tokens": tokens_done, "val_loss": vl, "minutes": minutes})
        ppl = math.exp(vl)
        log(f"eval step {done} | tokens {tokens_done} | val {vl:.4f} | ppl {ppl:.3f} | {minutes:.1f} min")
    if (time.time() - t0) / 60 > MAX_MINUTES and done < total_updates:
        stopped_early = True
        vl = evaluate()
        minutes = (time.time() - t0) / 60
        history.append({"step": done, "tokens": tokens_done, "val_loss": vl, "minutes": minutes})
        log(f"time cap reached at step {done}")
        break

final = history[-1]
if device == "cuda":
    gpu = torch.cuda.get_device_name(0)
else:
    gpu = "cpu"
result = {"tag": TAG, "group": GROUP, "config": cfg, "params": n_params, "engram_params": engram_params,
          "backbone_params": backbone_params, "updates_done": final["step"], "updates_planned": total_updates,
          "tokens_seen": final["tokens"], "final_val_loss": final["val_loss"], "final_val_ppl": math.exp(final["val_loss"]),
          "val_tokens": val_tokens, "tokens_per_sec": final["step"] * tokens_per_update / train_time,
          "train_minutes": train_time / 60, "wall_minutes": (time.time() - t0) / 60, "skipped_fp16_steps": skipped,
          "stopped_early": stopped_early, "history": history, "gpu": gpu}
json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)
final_loss = final["val_loss"]
final_ppl = math.exp(final_loss)
minutes = (time.time() - t0) / 60
log(f"done {TAG} | val {final_loss:.4f} | ppl {final_ppl:.3f} | {minutes:.1f} min")
