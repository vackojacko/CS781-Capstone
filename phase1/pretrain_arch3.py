# Arch 3: data pipeline, mixed-objective pretraining and test generation (Codabench 0.8275)
import os
import random
import json
import time
import math
import zipfile
import hashlib
import array
import gc
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ijson

from tokenizer import BPETokenizer
from arch3_model import LMConfig, EncoderDecoderLM, RMSNorm

DATA_DIR = "data"
TOKENIZER_PATH = "legal_bpe_16384.json"
OUTPUT_DIR = "outputs/arch3"
TRAIN_FILES = ["train_supreme_court.json", "train_high_courts.json", "train_acts.json", "train_tribunals.json"]
TEST_FILE = "test.json"

SEED = 42

# model: 10-layer encoder + 4-layer decoder, about 86M params
MODEL_DIM = 640
MODEL_ENC_LAYERS = 10
MODEL_DEC_LAYERS = 4
MODEL_HEADS = 10
MODEL_FF = 1712
ENC_SEQ_LEN = 704
DEC_SEQ_LEN = 320
MAX_ENC_LEN = ENC_SEQ_LEN
MAX_DEC_LEN = DEC_SEQ_LEN
MODEL_DROPOUT = 0.0

# special tokens are appended after the 16384 BPE ids
BASE_VOCAB = 16384
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
CONT_ID = BASE_VOCAB
DENOISE_ID = BASE_VOCAB + 1
N_SENTINELS = 62
SENTINEL_IDS = list(range(BASE_VOCAB + 2, BASE_VOCAB + 2 + N_SENTINELS))
VOCAB_SIZE = BASE_VOCAB + 2 + N_SENTINELS
IGNORE_INDEX = -100

# objectives
P_DENOISE = 0.20
FINAL_TASK_FRAC = 0.10
SPAN_NOISE_DENSITY = 0.15
SPAN_MEAN_LEN = 3.0
SPAN_SRC_LEN = 760

# data
TRAIN_TOKENS = 600_000_000
VAL_TOKENS = 5_000_000
BLOCK = ENC_SEQ_LEN + DEC_SEQ_LEN
VAL_DOC_PCT = 5
TASK_PROMPT_WORDS = 512
TASK_TARGET_WORDS = 128
TASK_POOL_SIZE = 120_000
EVAL_SET_SIZE = 250

# training
BATCH_SIZE = 4
GRAD_ACCUM = 8
PEAK_LR = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
TRAIN_BUDGET_HOURS = 18.0
MEASURE_FROM = 50
MEASURE_TO = 300
HORIZON_SAFETY = 0.95
CKPT_EVERY_UPDATES = 1000
CKPT_EVERY_MIN = 60
VAL_EVERY_UPDATES = 1000
MAX_VAL_BATCHES = 100
LOG_EVERY = 50

# generation
TARGET_WORDS = 128
MAX_GEN_TOKENS = MAX_DEC_LEN - 1
TEMPERATURE = 0.4
TOP_P = 0.92
TOP_K = 50
REPETITION_PENALTY = 1.15

LATEST_CKPT = "arch3_latest.pt"
FOUNDATION_CKPT = "arch3_foundation_prefinal.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


tokenizer = BPETokenizer.load(TOKENIZER_PATH)
assert tokenizer.vocab_size == BASE_VOCAB
print(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}")

NEVER_GENERATE = [PAD_ID, UNK_ID, BOS_ID, EOS_ID, CONT_ID, DENOISE_ID]
for s in SENTINEL_IDS:
    NEVER_GENERATE.append(s)
NEVER_GENERATE = sorted(set(NEVER_GENERATE))


# data: build once and cache, reload on later runs
CACHE = {
    "train": "arch3_train_blocks.npy",
    "val": "arch3_val_blocks.npy",
    "task": "arch3_task_pool.npz",
    "eval": "arch3_eval_set.json",
}
have_cache = True
for name in CACHE.values():
    if not os.path.exists(os.path.join(OUTPUT_DIR, name)):
        have_cache = False

if have_cache:
    print("Loading data cache")
    train_blocks = np.load(os.path.join(OUTPUT_DIR, CACHE["train"]), mmap_mode="r")
    val_blocks = np.load(os.path.join(OUTPUT_DIR, CACHE["val"]), mmap_mode="r")
    t = np.load(os.path.join(OUTPUT_DIR, CACHE["task"]))
    task_tokens = t["tokens"]
    task_offsets = t["offsets"]
    task_plen = t["plen"]
    with open(os.path.join(OUTPUT_DIR, CACHE["eval"])) as f:
        eval_set = json.load(f)
else:
    print("Building data from the corpus")
    rng = random.Random(SEED)
    seen = set()
    train_tok = array.array('H')
    val_tok = array.array('H')
    task_tok = array.array('H')
    task_off = [0]
    task_pl = []
    eval_set = []
    tokens_per_source = TRAIN_TOKENS // len(TRAIN_FILES)
    val_per_source = VAL_TOKENS // len(TRAIN_FILES)
    task_per_source = TASK_POOL_SIZE // len(TRAIN_FILES)
    eval_per_source = EVAL_SET_SIZE // len(TRAIN_FILES) + 1
    min_words = TASK_PROMPT_WORDS + TASK_TARGET_WORDS

    def doc_blocks(toks):
        n_blocks = len(toks) // BLOCK
        if n_blocks == 0:
            return None
        offset = rng.randint(0, len(toks) - n_blocks * BLOCK)  # random start, not always the heading
        return toks[offset:offset + n_blocks * BLOCK]

    def add_blocks(dst, toks, limit):
        room = (limit - len(dst)) // BLOCK * BLOCK
        if room > 0:
            dst.extend(toks[:room])
        return min(len(toks), max(room, 0)) // BLOCK

    def word_window(words):
        start = rng.randint(0, len(words) - min_words)
        prompt = " ".join(words[start:start + TASK_PROMPT_WORDS])
        target = " ".join(words[start + TASK_PROMPT_WORDS:start + min_words])
        return prompt, target

    for train_file in TRAIN_FILES:
        src = train_file.replace("train_", "").replace(".json", "")
        print("Source:", src)
        s = Counter()
        train_limit = len(train_tok) + tokens_per_source
        val_limit = len(val_tok) + val_per_source
        task_start = len(task_pl)
        eval_start = len(eval_set)

        with open(os.path.join(DATA_DIR, train_file), 'rb') as f:
            for record in ijson.items(f, 'item'):
                s["docs_scanned"] += 1
                text = record["text"]
                key = hashlib.md5(text[:500].encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                words = text.split()
                if len(words) < 50:
                    continue
                s["docs_kept"] += 1
                is_val = int(key[:8], 16) % 100 < VAL_DOC_PCT

                if is_val:
                    if len(eval_set) - eval_start < eval_per_source and len(words) >= min_words:
                        p, tg = word_window(words)
                        eval_set.append({"source": src, "doc_key": key, "prompt": p, "reference": tg})
                    if len(val_tok) + BLOCK <= val_limit:
                        blocks = doc_blocks(tokenizer.encode(text))
                        if blocks is not None:
                            add_blocks(val_tok, blocks, val_limit)
                else:
                    need_blocks = len(train_tok) + BLOCK <= train_limit
                    need_task = len(task_pl) - task_start < task_per_source and len(words) >= min_words
                    if need_blocks:
                        toks = tokenizer.encode(text)
                        blocks = doc_blocks(toks)
                        if blocks is not None:
                            add_blocks(train_tok, blocks, train_limit)
                    if need_task:
                        p, tg = word_window(words)
                        p_ids = tokenizer.encode(p)
                        t_ids = tokenizer.encode(tg)
                        if len(p_ids) > ENC_SEQ_LEN - 1:
                            p_ids = p_ids[-(ENC_SEQ_LEN - 1):]  # same left truncation as inference
                        if len(t_ids) > DEC_SEQ_LEN:
                            t_ids = t_ids[:DEC_SEQ_LEN]
                        task_tok.extend(p_ids)
                        task_tok.extend(t_ids)
                        task_off.append(len(task_tok))
                        task_pl.append(len(p_ids))

                done_blocks = len(train_tok) + BLOCK > train_limit and len(val_tok) + BLOCK > val_limit
                done_task = len(task_pl) - task_start >= task_per_source
                done_eval = len(eval_set) - eval_start >= eval_per_source
                if done_blocks and done_task and done_eval:
                    break
                if s["docs_scanned"] % 20000 == 0:
                    print(f"  {s['docs_scanned']} docs | train={len(train_tok)} val={len(val_tok)} task={len(task_pl) - task_start}")
        gc.collect()
        print(f"  {dict(s)}")

    del seen
    train_blocks = np.frombuffer(train_tok, dtype=np.uint16).reshape(-1, BLOCK).copy()
    val_blocks = np.frombuffer(val_tok, dtype=np.uint16).reshape(-1, BLOCK).copy()
    task_tokens = np.frombuffer(task_tok, dtype=np.uint16).copy()
    task_offsets = np.array(task_off, dtype=np.int64)
    task_plen = np.array(task_pl, dtype=np.int32)
    del train_tok, val_tok, task_tok
    gc.collect()
    eval_set = eval_set[:EVAL_SET_SIZE]

    np.save(os.path.join(OUTPUT_DIR, CACHE["train"]), train_blocks)
    np.save(os.path.join(OUTPUT_DIR, CACHE["val"]), val_blocks)
    np.savez(os.path.join(OUTPUT_DIR, CACHE["task"]), tokens=task_tokens, offsets=task_offsets, plen=task_plen)
    with open(os.path.join(OUTPUT_DIR, CACHE["eval"]), "w") as f:
        json.dump(eval_set, f)

print(f"Train blocks: {len(train_blocks)} | val blocks: {len(val_blocks)} | task pool: {len(task_plen)} | eval set: {len(eval_set)}")
tokenizer._encode_cache = {}
gc.collect()


class EpochSampler:
    # seeded permutation, reshuffled every epoch; state (epoch, pos) resumes exactly
    def __init__(self, n, seed):
        self.n = n
        self.seed = seed
        self.epoch = 0
        self.pos = 0
        self.make_perm()

    def make_perm(self):
        self.perm = np.random.default_rng([self.seed, self.epoch]).permutation(self.n)

    def take(self, k):
        out = []
        while len(out) < k:
            if self.pos >= self.n:
                self.epoch += 1
                self.pos = 0
                self.make_perm()
            m = min(k - len(out), self.n - self.pos)
            out.extend(self.perm[self.pos:self.pos + m].tolist())
            self.pos += m
        return np.array(out, dtype=np.int64)

    def state(self):
        return {"epoch": self.epoch, "pos": self.pos}

    def load(self, st):
        self.epoch = st["epoch"]
        self.pos = st["pos"]
        self.make_perm()


block_sampler = EpochSampler(len(train_blocks), SEED + 1)
task_sampler = EpochSampler(len(task_plen), SEED + 2)


def objective_for(update_idx, final_start):
    # every microbatch of one update uses the same objective
    if final_start is not None and update_idx >= final_start:
        return "task"
    r = np.random.default_rng([SEED, 7, update_idx]).random()
    if r < P_DENOISE:
        return "denoise"
    return "cont"


def partition(total, n, rng):
    # random split of total into n positive parts
    if n == 1:
        return np.array([total])
    cuts = np.sort(rng.choice(np.arange(1, total), n - 1, replace=False))
    return np.diff(np.concatenate([[0], cuts, [total]]))


def span_corrupt(tokens, rng):
    L = len(tokens)
    n_noise = min(max(int(round(L * SPAN_NOISE_DENSITY)), 1), L - 1)
    n_spans = min(max(int(round(n_noise / SPAN_MEAN_LEN)), 1), N_SENTINELS, L - n_noise)

    noise_lens = partition(n_noise, n_spans, rng)
    keep_lens = partition(L - n_noise, n_spans, rng)
    inp = [DENOISE_ID]
    tgt = []
    p = 0
    for i in range(n_spans):
        inp.extend(tokens[p:p + keep_lens[i]])
        p += keep_lens[i]
        inp.append(SENTINEL_IDS[i])
        tgt.append(SENTINEL_IDS[i])
        tgt.extend(tokens[p:p + noise_lens[i]])
        p += noise_lens[i]
    tgt.append(EOS_ID)
    return inp, tgt


def collate(encs, tgts):
    # right pad; decoder input = BOS + target shifted right
    B = len(encs)
    Te = max([len(e) for e in encs])
    Td = max([len(t) for t in tgts])
    assert Te <= ENC_SEQ_LEN and Td <= DEC_SEQ_LEN
    enc = np.full((B, Te), PAD_ID, dtype=np.int64)
    enc_mask = np.zeros((B, Te), dtype=bool)
    dec_in = np.full((B, Td), PAD_ID, dtype=np.int64)
    tgt = np.full((B, Td), IGNORE_INDEX, dtype=np.int64)
    for i in range(B):
        e = encs[i]
        t = tgts[i]
        enc[i, :len(e)] = e
        enc_mask[i, :len(e)] = True
        dec_in[i, 0] = BOS_ID
        dec_in[i, 1:len(t)] = t[:-1]
        tgt[i, :len(t)] = t
    return enc, enc_mask, dec_in, tgt


def make_microbatch(obj, update_idx, micro_idx, blocks, bs, ts):
    encs = []
    tgts = []
    if obj == "cont":
        # encoder = <cont> + 703 tokens, decoder target = the next 320 tokens
        blk = blocks[bs.take(BATCH_SIZE)].astype(np.int64)
        for b in blk:
            encs.append(np.concatenate([[CONT_ID], b[:ENC_SEQ_LEN - 1]]))
            tgts.append(b[ENC_SEQ_LEN - 1:ENC_SEQ_LEN - 1 + DEC_SEQ_LEN])
    elif obj == "denoise":
        blk = blocks[bs.take(BATCH_SIZE)].astype(np.int64)
        for b_i in range(len(blk)):
            rng = np.random.default_rng([SEED, 11, update_idx, micro_idx, b_i])
            e, t = span_corrupt(blk[b_i][:SPAN_SRC_LEN].tolist(), rng)
            encs.append(e)
            tgts.append(t)
    else:
        idx = ts.take(BATCH_SIZE)
        for j in idx:
            a = task_offsets[j]
            z = task_offsets[j + 1]
            pl = task_plen[j]
            seq = task_tokens[a:z].astype(np.int64)
            encs.append(np.concatenate([[CONT_ID], seq[:pl]]))
            tgts.append(seq[pl:])
    return collate(encs, tgts)


def to_device(mb):
    out = []
    for a in mb:
        out.append(torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True))
    return tuple(out)


cfg = LMConfig(
    vocab_size=VOCAB_SIZE,
    d_model=MODEL_DIM,
    n_enc_layers=MODEL_ENC_LAYERS,
    n_dec_layers=MODEL_DEC_LAYERS,
    n_heads=MODEL_HEADS,
    d_ff=MODEL_FF,
    max_enc_len=MAX_ENC_LEN,
    max_dec_len=MAX_DEC_LEN,
    dropout=MODEL_DROPOUT,
)

torch.manual_seed(SEED)
model = EncoderDecoderLM(cfg).to(device)


def init_weights(module):
    if module is model.head:
        return  # tied with tok_emb, initialised once there
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, RMSNorm):
        torch.nn.init.ones_(module.weight)


model.apply(init_weights)

# residual projections get std 0.02 / sqrt(2 * (10 + 4))
total_layers = MODEL_ENC_LAYERS + MODEL_DEC_LAYERS
residual_std = 0.02 / math.sqrt(2 * total_layers)
residual_proj = []
for block in model.enc_blocks:
    residual_proj.append(block.attn.out_proj)
    residual_proj.append(block.ffn.down_proj)
for block in model.dec_blocks:
    residual_proj.append(block.self_attn.out_proj)
    residual_proj.append(block.cross_attn.out_proj)
    residual_proj.append(block.ffn.down_proj)
for lin in residual_proj:
    torch.nn.init.normal_(lin.weight, mean=0.0, std=residual_std)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params} ({n_params / 1e6:.1f}M)")

use_amp = torch.cuda.is_available()


def amp_ctx():
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return torch.autocast(device_type="cpu", dtype=torch.float16, enabled=False)


# weight decay only on 2D params
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


def lr_at(update, planned_total):
    # linear warmup, then cosine over the planned horizon (fixed during warmup)
    if update < WARMUP_STEPS:
        return PEAK_LR * update / WARMUP_STEPS
    assert planned_total is not None
    progress = min(1.0, (update - WARMUP_STEPS) / max(1, planned_total - WARMUP_STEPS))
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * progress)) * (PEAK_LR - MIN_LR)


def plan_horizon(tp):
    t_cont = float(np.mean(tp["cont"]))
    if tp["denoise"]:
        t_den = float(np.mean(tp["denoise"]))
    else:
        t_den = t_cont
    t_task = t_cont
    per_update = (1 - FINAL_TASK_FRAC) * ((1 - P_DENOISE) * t_cont + P_DENOISE * t_den) + FINAL_TASK_FRAC * t_task
    total = int(TRAIN_BUDGET_HOURS * 3600 * HORIZON_SAFETY / per_update)
    final_start = int(total * (1 - FINAL_TASK_FRAC))
    return total, final_start


NEVER_GEN_T = torch.tensor(NEVER_GENERATE, device=device)


def generate(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS, target_words=TARGET_WORDS,
             temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, repetition_penalty=REPETITION_PENALTY,
             seed=SEED):
    model.eval()
    p_ids = tokenizer.encode(prompt)
    p_ids = p_ids[-(ENC_SEQ_LEN - 1):]
    enc_input = torch.tensor([[CONT_ID] + p_ids], dtype=torch.long, device=device)
    max_tokens = min(max_tokens, MAX_DEC_LEN - 1)
    gen = None
    if temperature > 0:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    with torch.no_grad():
        with amp_ctx():
            enc_out = model.encode(enc_input)
        dec = [BOS_ID]
        generated = []
        for _ in range(max_tokens):
            dec_input = torch.tensor([dec], dtype=torch.long, device=device)
            with amp_ctx():
                logits = model.decode(dec_input, enc_out)
            logits = logits[0, -1].float()
            logits[NEVER_GEN_T] = float('-inf')

            if repetition_penalty != 1.0 and generated:
                ids = torch.tensor(sorted(set(generated[-100:])), device=device)
                vals = logits[ids]
                # divide positive logits, multiply negative ones
                logits[ids] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

            if temperature == 0.0:
                tok = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k > 0:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                    logits[logits < kth] = float('-inf')
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                logits[sorted_idx[remove]] = float('-inf')
                probs = torch.softmax(logits, dim=-1)
                tok = int(torch.multinomial(probs, 1, generator=gen))

            dec.append(tok)
            generated.append(tok)
            n_words = len(tokenizer.decode(generated).split())
            if n_words > target_words or (n_words == target_words and tokenizer._ends_word.get(tok, False)):
                break

    words = tokenizer.decode(generated).split()[:target_words]
    return " ".join(words)


def val_loss_cont(max_batches=None):
    model.eval()
    val_sampler = EpochSampler(len(val_blocks), SEED + 3)
    unused_sampler = EpochSampler(1, 0)
    n_batches = len(val_blocks) // BATCH_SIZE
    if max_batches is not None:
        n_batches = min(max_batches, n_batches)
    tot = 0.0
    ntok = 0
    with torch.no_grad():
        for b in range(n_batches):
            enc, enc_mask, dec_in, tgt = to_device(make_microbatch("cont", 0, b, val_blocks, val_sampler, unused_sampler))
            with amp_ctx():
                logits = model(enc, dec_in, enc_mask)
            tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), tgt.view(-1),
                                   ignore_index=IGNORE_INDEX, reduction="sum").item()
            ntok += int((tgt != IGNORE_INDEX).sum())
    model.train()
    return tot / max(1, ntok)


def new_run_state():
    run = {"update": 0, "skipped_updates": 0, "planned_total": None, "final_start": None,
           "tp": {"cont": [], "denoise": []}, "history": [], "val_history": [],
           "pre_final_saved": False, "done": False}
    return run


def save_checkpoint(name):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "model_config": vars(cfg),
        "run": run,
        "samplers": {"block": block_sampler.state(), "task": task_sampler.state()},
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
    }
    torch.save(state, os.path.join(OUTPUT_DIR, name))
    print(f"  saved {name} at update {run['update']}")


scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
run = new_run_state()

# resume from a local checkpoint if one exists
resume_path = os.path.join(OUTPUT_DIR, LATEST_CKPT)
if os.path.exists(resume_path):
    print(f"Resuming from {resume_path}")
    ck = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scaler.load_state_dict(ck["scaler"])
    run = ck["run"]
    block_sampler.load(ck["samplers"]["block"])
    task_sampler.load(ck["samplers"]["task"])
    random.setstate(ck["rng"]["python"])
    np.random.set_state(ck["rng"]["numpy"])
    torch.set_rng_state(ck["rng"]["torch"].cpu())
    del ck

OBJS = ["cont", "denoise", "task"]
win = {}
for o in OBJS:
    win[o] = [0.0, 0]
consecutive_skips = 0
last_ckpt = time.time()
model.train()

while True:
    u = run["update"]
    if run["planned_total"] is not None and u >= run["planned_total"]:
        run["done"] = True
        break

    # save the mixed-objective foundation before the task-matched segment starts
    if run["final_start"] is not None and u >= run["final_start"] and not run["pre_final_saved"]:
        save_checkpoint(FOUNDATION_CKPT)
        run["pre_final_saved"] = True

    t0 = time.time()
    obj = objective_for(u, run["final_start"])
    micro = []
    for mi in range(GRAD_ACCUM):
        micro.append(make_microbatch(obj, u, mi, train_blocks, block_sampler, task_sampler))
    n_valid = 0
    for mb in micro:
        n_valid += int((mb[3] != IGNORE_INDEX).sum())
    lr = lr_at(u, run["planned_total"])
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for mb in micro:
        enc, enc_mask, dec_in, tgt = to_device(mb)
        with amp_ctx():
            logits = model(enc, dec_in, enc_mask)
        loss_tok = F.cross_entropy(logits.float().view(-1, logits.size(-1)), tgt.view(-1),
                                   ignore_index=IGNORE_INDEX, reduction="sum")
        # mean over all valid target tokens of the whole update
        scaler.scale(loss_tok / n_valid).backward()
        loss_sum += loss_tok.item()
        del logits, loss_tok
    scaler.unscale_(optimizer)
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if use_amp:
        torch.cuda.synchronize()
    dt = time.time() - t0

    # fp16 overflow: no parameter change and the LR schedule does not advance
    if (use_amp and scaler.get_scale() < scale_before) or not math.isfinite(loss_sum):
        run["skipped_updates"] += 1
        consecutive_skips += 1
        if consecutive_skips >= 50:
            raise RuntimeError(f"{consecutive_skips} consecutive skipped updates at update {u}")
        continue
    consecutive_skips = 0
    run["update"] = u + 1
    win[obj][0] += loss_sum
    win[obj][1] += n_valid

    # measure throughput during warmup and fix the update horizon
    if MEASURE_FROM <= u < MEASURE_TO and obj in run["tp"]:
        run["tp"][obj].append(dt)
    if run["planned_total"] is None and run["update"] >= MEASURE_TO:
        total, fs = plan_horizon(run["tp"])
        assert total > 2 * WARMUP_STEPS
        run["planned_total"] = total
        run["final_start"] = fs
        print(f"Horizon fixed: {total} updates, task-matched segment from update {fs}")

    if run["update"] % LOG_EVERY == 0:
        entry = {"update": run["update"], "lr": lr, "gnorm": gnorm}
        msg = ""
        for o in OBJS:
            if win[o][1] > 0:
                entry["loss_" + o] = win[o][0] / win[o][1]
                msg += f"{o}={win[o][0] / win[o][1]:.3f} "
        print(f"upd {run['update']}/{run['planned_total']} | {msg}| lr {lr:.2e} | gnorm {gnorm:.2f} | {dt:.2f}s/upd")
        run["history"].append(entry)
        for o in OBJS:
            win[o] = [0.0, 0]

    if run["update"] % VAL_EVERY_UPDATES == 0:
        vl = val_loss_cont(MAX_VAL_BATCHES)
        run["val_history"].append({"update": run["update"], "val_cont_loss": vl})
        print(f"  val continuation loss {vl:.4f} (ppl {math.exp(min(vl, 20)):.1f})")

    if run["update"] % CKPT_EVERY_UPDATES == 0 or time.time() - last_ckpt >= CKPT_EVERY_MIN * 60:
        save_checkpoint(LATEST_CKPT)
        last_ckpt = time.time()

save_checkpoint(LATEST_CKPT)
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
final_val_loss = val_loss_cont()
print(f"Final val continuation loss {final_val_loss:.4f} (ppl {math.exp(min(final_val_loss, 20)):.2f})")


# test predictions
with open(os.path.join(DATA_DIR, TEST_FILE), "r", encoding="utf-8") as f:
    test_data = json.load(f)
print(f"Number of test examples: {len(test_data)}")

predictions = {}
items = list(test_data.items())
t0 = time.time()
for i, (example_id, record) in enumerate(items):
    text = generate(model, tokenizer, record["text"], seed=SEED + 100000 + i)
    predictions[str(example_id)] = {"text": text}
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(items)} | {time.time() - t0:.0f}s")

PREDICTION_PATH = os.path.join(OUTPUT_DIR, "prediction.json")
SUBMISSION_ZIP = os.path.join(OUTPUT_DIR, "submission.zip")
with open(PREDICTION_PATH, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(PREDICTION_PATH, arcname="prediction.json")
print(f"submission.zip created: {SUBMISSION_ZIP} | predictions: {len(predictions)}")
