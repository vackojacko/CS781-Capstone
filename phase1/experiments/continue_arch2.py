# Continued pretraining of Arch 2 on text it never read, then test outputs with Arch 2's own sampling
import os
import json
import math
import random
import hashlib
import array
import gc
import ijson
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from arch2_model import LMConfig, DecoderLM
from generate_mbr import tokenizer, device
from experiments.resample import write_test_outputs

DATA_DIR = "data"
TRAIN_FILES = ["train_supreme_court.json", "train_high_courts.json", "train_acts.json", "train_tribunals.json"]
ARCH2_DIR = os.path.join("outputs", "arch2")
OUTPUT_DIR = os.path.join("outputs", "arch2_continued")
VOCAB_SIZE = 16384
SEED = 42

# arch 2's data quotas from pretrain.py, replayed to find where it stopped reading each source
A2_TRAIN_TOKENS = 300000000
A2_VAL_TOKENS = 5000000
NEW_TOKENS = 600000000
HELDOUT_PCT = 5            # new documents kept out of training for held out prompts
EVAL_N = 200
PROMPT_WORDS = 512
TARGET_WORDS = 128

MODEL_SEQ_LEN = 1024
BLOCK = MODEL_SEQ_LEN + 1
BATCH_SIZE = 4
GRAD_ACCUM = 8
START_LR = 6.04e-5         # where the arch 2 schedule ended
PEAK_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 500
TOTAL_UPDATES = 10052      # what fit in 9 hours on a T4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
CKPT_EVERY = 1000
LOG_EVERY = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)
use_amp = torch.cuda.is_available()


def build_data():
    a2_rng = random.Random(SEED)
    new_tok = array.array('H')
    val_tok = array.array('H')
    eval_set = []
    earlier = set()
    a2_train_ps = A2_TRAIN_TOKENS // len(TRAIN_FILES)
    a2_val_ps = A2_VAL_TOKENS // len(TRAIN_FILES)
    new_ps = NEW_TOKENS // len(TRAIN_FILES)
    eval_ps = EVAL_N // len(TRAIN_FILES)
    min_words = PROMPT_WORDS + TARGET_WORDS

    for si, train_file in enumerate(TRAIN_FILES):
        b_rng = random.Random(SEED * 1000 + si)
        seen = set()
        a2_train_len = 0
        val_limit = len(val_tok) + a2_val_ps
        new_start = len(new_tok)
        new_limit = new_start + new_ps
        eval_start = len(eval_set)
        replaying = True
        new_docs = 0

        with open(os.path.join(DATA_DIR, train_file), "rb") as f:
            for record in ijson.items(f, "item"):
                text = record["text"]
                h = hashlib.md5(text[:500].encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                if len(text.split()) < 50:
                    continue

                if replaying:
                    # same draws and quotas as the arch 2 run of pretrain.py
                    toks = tokenizer.encode(text)
                    earlier.add(h)
                    if a2_rng.random() < 0.05:
                        if len(val_tok) < val_limit:
                            val_tok.extend(toks[:val_limit - len(val_tok)])
                    elif a2_train_len < a2_train_ps:
                        a2_train_len += min(len(toks), a2_train_ps - a2_train_len)
                    if a2_train_len >= a2_train_ps and len(val_tok) >= val_limit:
                        replaying = False
                        tokenizer._encode_cache = {}
                        gc.collect()
                    continue

                if h in earlier:
                    continue
                new_docs += 1
                if b_rng.random() < HELDOUT_PCT / 100:
                    words = text.split()
                    if len(eval_set) - eval_start < eval_ps and len(words) >= min_words:
                        st = b_rng.randint(0, len(words) - min_words)
                        eval_set.append({"prompt": " ".join(words[st:st + PROMPT_WORDS]),
                                         "reference": " ".join(words[st + PROMPT_WORDS:st + min_words])})
                    continue
                if len(new_tok) < new_limit:
                    toks = tokenizer.encode(text)
                    new_tok.extend(toks[:new_limit - len(new_tok)])
                eval_full = len(eval_set) - eval_start >= eval_ps
                if len(new_tok) >= new_limit and (eval_full or new_docs > 200000):
                    break

        tokenizer._encode_cache = {}
        gc.collect()
        print(f"{train_file}: {len(new_tok) - new_start} new tokens, {len(eval_set) - eval_start} held out prompts")

    n_tr = len(new_tok) // BLOCK
    n_va = len(val_tok) // BLOCK
    train_blocks = np.frombuffer(new_tok, dtype=np.uint16)[:n_tr * BLOCK].reshape(n_tr, BLOCK).copy()
    val_blocks = np.frombuffer(val_tok, dtype=np.uint16)[:n_va * BLOCK].reshape(n_va, BLOCK).copy()
    return train_blocks, val_blocks, eval_set


class EpochSampler:
    def __init__(self, n, seed):
        self.n = n
        self.seed = seed
        self.epoch = 0
        self.pos = 0
        self.shuffle()

    def shuffle(self):
        self.perm = np.random.default_rng([self.seed, self.epoch]).permutation(self.n)

    def take(self, k):
        out = []
        while len(out) < k:
            if self.pos >= self.n:
                self.epoch += 1
                self.pos = 0
                self.shuffle()
            m = min(k - len(out), self.n - self.pos)
            out.extend(self.perm[self.pos:self.pos + m].tolist())
            self.pos += m
        return np.array(out, dtype=np.int64)


def to_xy(blocks):
    t = torch.from_numpy(np.ascontiguousarray(blocks).astype(np.int64))
    return t[:, :-1].to(device), t[:, 1:].to(device)


def lr_at(update):
    if update < WARMUP_STEPS:
        return START_LR + (PEAK_LR - START_LR) * update / WARMUP_STEPS
    progress = min(1.0, (update - WARMUP_STEPS) / (TOTAL_UPDATES - WARMUP_STEPS))
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * progress)) * (PEAK_LR - MIN_LR)


def val_loss(model, val_blocks):
    # mean of the per batch losses, like evaluate() in pretrain.py
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for a in range(0, len(val_blocks), BATCH_SIZE):
            x, y = to_xy(val_blocks[a:a + BATCH_SIZE])
            with autocast():
                logits = model(x)
            total += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1)).item()
            count += 1
    model.train()
    return total / count


data_path = os.path.join(OUTPUT_DIR, "train_blocks.npy")
if os.path.exists(data_path):
    train_blocks = np.load(data_path, mmap_mode="r")
    val_blocks = np.load(os.path.join(OUTPUT_DIR, "val_blocks.npy"))
else:
    train_blocks, val_blocks, eval_set = build_data()
    np.save(data_path, train_blocks)
    np.save(os.path.join(OUTPUT_DIR, "val_blocks.npy"), val_blocks)
    with open(os.path.join(OUTPUT_DIR, "eval_set.json"), "w") as f:
        json.dump(eval_set, f)
print("New training blocks:", len(train_blocks), "| validation blocks:", len(val_blocks))

cfg = LMConfig(vocab_size=VOCAB_SIZE, d_model=640, n_layers=12, n_heads=10, n_kv_heads=2, d_ff=1706,
               max_seq_len=MODEL_SEQ_LEN, dropout=0.0)
model = DecoderLM(cfg).to(device)

decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if param.dim() >= 2:
        decay_params.append(param)
    else:
        no_decay_params.append(param)
optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": WEIGHT_DECAY},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8)
scaler = GradScaler(enabled=use_amp)
sampler = EpochSampler(len(train_blocks), SEED + 1)

# start from the arch 2 weights, adam moments and grad scaler, or resume this run
ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint.pt")
update = 0
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    update = ckpt["update"]
    sampler.epoch = ckpt["epoch"]
    sampler.pos = ckpt["pos"]
    sampler.shuffle()
    print("Resuming at update", update)
else:
    model.load_state_dict(torch.load(os.path.join(ARCH2_DIR, "model.pt"), map_location=device, weights_only=False))
    ckpt = torch.load(os.path.join(ARCH2_DIR, "checkpoint.pt"), map_location=device, weights_only=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    if use_amp:
        scaler.load_state_dict(ckpt["scaler"])
del ckpt

model.train()
loss_window = []
while update < TOTAL_UPDATES:
    lr = lr_at(update)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for _ in range(GRAD_ACCUM):
        x, y = to_xy(train_blocks[sampler.take(BATCH_SIZE)])
        with autocast():
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1)) / GRAD_ACCUM
        scaler.scale(loss).backward()
        loss_sum += loss.item()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()

    # an fp16 overflow skips the step, so this update is done again with the next batches
    if scaler.get_scale() < scale_before or not math.isfinite(loss_sum):
        continue
    update += 1
    loss_window.append(loss_sum)

    if update % LOG_EVERY == 0:
        print(f"update {update}/{TOTAL_UPDATES} | loss {np.mean(loss_window):.4f} | lr {lr:.2e}")
        loss_window = []
    if update % CKPT_EVERY == 0:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "update": update, "epoch": sampler.epoch, "pos": sampler.pos}, ckpt_path)

torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
print(f"Val loss on the arch 2 validation set: {val_loss(model, val_blocks):.4f}")

model.eval()
write_test_outputs(model, OUTPUT_DIR)
