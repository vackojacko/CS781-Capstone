# Arch 4 pretraining (WSD schedule, task-matched final segment) and test generation
import os
import random
import json
import time
import math
import zipfile
import hashlib
import copy
import array
import gc
import unicodedata
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ijson
import matplotlib.pyplot as plt
from bert_score import BERTScorer

from tokenizer import BPETokenizer
from arch4_model import LMConfig, DecoderLM, RMSNorm

DATA_DIR = "data"
TOKENIZER_PATH = "legal_bpe_16384.json"
OUTPUT_DIR = "outputs/arch4"
TRAIN_FILES = ["train_supreme_court.json", "train_high_courts.json", "train_acts.json", "train_tribunals.json"]
TEST_FILE = "test.json"

SEED = 42

# model
MODEL_DIM = 640
MODEL_LAYERS = 16
MODEL_HEADS = 10
ROPE_DIM = 32
KV_LATENT = 320
MODEL_FF = 1712
MODEL_SEQ_LEN = 1024
MODEL_DROPOUT = 0.0
VOCAB_SIZE = 16384
PAD_ID = 0
BOS_ID = 2
EOS_ID = 3
IGNORE_INDEX = -100

# Engram memory
ENGRAM_LAYERS = (2, 8)
ENGRAM_ORDERS = (2, 3)
ENGRAM_HEADS = 4
ENGRAM_HEAD_DIM = 32
ENGRAM_ROWS = 81920
ENGRAM_LR_MULT = 5.0

# data
TRAIN_TOKENS = 900_000_000
VAL_TOKENS = 5_000_000
BLOCK = MODEL_SEQ_LEN + 1
VAL_DOC_PCT = 5
TASK_PROMPT_WORDS = 512
TASK_TARGET_WORDS = 128
PROMPT_MAX_TOKENS = 704
TARGET_MAX_TOKENS = 320
TASK_POOL_SIZE = 150_000
EVAL_SET_SIZE = 400
FINAL_TASK_FRAC = 0.10

# training
BATCH_SIZE = 4
GRAD_ACCUM = 8
PEAK_LR = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
MAX_TRAIN_HOURS = 10.0
MIN_STABLE_UPDATES = 4000
PLATEAU_WINDOW = 3
PLATEAU_REL_TOL = 0.002
DECAY_FRAC = 0.20
MEASURE_FROM = 50
MEASURE_TO = 300
CKPT_EVERY_UPDATES = 1000
CKPT_EVERY_MIN = 30
SNAPSHOT_EVERY = 3000
SNAPSHOT_START = 3000
VAL_EVERY_UPDATES = 1000
MAX_VAL_BATCHES = 100
EVAL_EVERY_UPDATES = 8000
VAL_SAMPLE_BLOCKS = 400
LOG_EVERY = 50

# generation
TARGET_WORDS = 128
MAX_GEN_TOKENS = MODEL_SEQ_LEN - PROMPT_MAX_TOKENS
TEMPERATURE = 0.4
TOP_P = 0.92
TOP_K = 50
REPETITION_PENALTY = 1.15

LATEST_CKPT = "arch4_latest.pt"
FOUNDATION_CKPT = "arch4_before_task_segment.pt"
STABLE_END_CKPT = "arch4_stable_end.pt"
BEST_PHASE1 = "arch4_best_phase1_model.pt"
CACHE = {
    "train": "arch4v2_train_blocks.npy",
    "val": "arch4v2_val_blocks.npy",
    "task": "arch4v2_task_pool.npz",
    "eval": "arch4v2_eval_set.json",
    "meta": "arch4v2_data_meta.json",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


tokenizer = BPETokenizer.load(TOKENIZER_PATH)
assert tokenizer.vocab_size == VOCAB_SIZE
print(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}")


# Engram compression: tokens with the same normalised lowercase text share one n-gram id
def compress_key(b, ends_word):
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return ("bytes", b, ends_word)
    s = unicodedata.normalize("NFKC", s)
    chars = []
    for ch in unicodedata.normalize("NFKD", s):
        if not unicodedata.combining(ch):
            chars.append(ch)
    s = "".join(chars)
    return (" ".join(s.lower().split()), ends_word)


keys = {}
compress = []
for t in range(VOCAB_SIZE):
    b = tokenizer._token_bytes.get(t)
    if b is None:
        key = ("special", t)
    else:
        key = compress_key(b, bool(tokenizer._ends_word.get(t, False)))
    if key not in keys:
        keys[key] = len(keys)
    compress.append(keys[key])
ENGRAM_COMPRESS = torch.tensor(compress, dtype=torch.long)
torch.save(ENGRAM_COMPRESS, os.path.join(OUTPUT_DIR, "arch4_engram_compress.pt"))
print(f"Engram compressed vocab: {len(keys)} n-gram ids from {VOCAB_SIZE} tokens")


TOKENIZER_MD5 = hashlib.md5(open(TOKENIZER_PATH, "rb").read()).hexdigest()
cache_ready = True
for name in CACHE.values():
    if not os.path.exists(os.path.join(OUTPUT_DIR, name)):
        cache_ready = False

if cache_ready:
    print("Loading the data cache from an earlier run.")
    train_blocks = np.load(os.path.join(OUTPUT_DIR, CACHE["train"]), mmap_mode="r")
    val_blocks = np.load(os.path.join(OUTPUT_DIR, CACHE["val"]), mmap_mode="r")
    task_file = np.load(os.path.join(OUTPUT_DIR, CACHE["task"]))
    task_tokens = task_file["tokens"]
    task_offsets = task_file["offsets"]
    task_plen = task_file["plen"]
    with open(os.path.join(OUTPUT_DIR, CACHE["eval"])) as f:
        eval_set = json.load(f)
    with open(os.path.join(OUTPUT_DIR, CACHE["meta"])) as f:
        data_meta = json.load(f)
    assert data_meta["tokenizer_md5"] == TOKENIZER_MD5 and data_meta["block"] == BLOCK
else:
    print("Building data from the official corpus.")
    rng = random.Random(SEED)
    seen = set()
    train_tok = array.array('H')
    val_tok = array.array('H')
    task_tok = array.array('H')
    task_off = [0]
    task_pl = []
    eval_set = []
    val_doc_keys = []
    stats = {}
    tokens_per_source = TRAIN_TOKENS // len(TRAIN_FILES)
    val_per_source = VAL_TOKENS // len(TRAIN_FILES)
    task_per_source = TASK_POOL_SIZE // len(TRAIN_FILES)
    eval_per_source = EVAL_SET_SIZE // len(TRAIN_FILES) + 1
    min_words = TASK_PROMPT_WORDS + TASK_TARGET_WORDS

    def doc_blocks(toks):
        n_blocks = len(toks) // BLOCK
        if n_blocks == 0:
            return None
        # random offset so blocks do not always start at the heading
        offset = rng.randint(0, len(toks) - n_blocks * BLOCK)
        return toks[offset:offset + n_blocks * BLOCK]

    def add_blocks(dst, toks, limit):
        room = (limit - len(dst)) // BLOCK * BLOCK
        if room > 0:
            dst.extend(toks[:room])

    def word_window(words):
        start = rng.randint(0, len(words) - min_words)
        prompt = " ".join(words[start:start + TASK_PROMPT_WORDS])
        target = " ".join(words[start + TASK_PROMPT_WORDS:start + min_words])
        return prompt, target

    for train_file in TRAIN_FILES:
        src = train_file.replace("train_", "").replace(".json", "")
        print("Source:", src)
        s = Counter()
        stats[src] = s
        train_limit = len(train_tok) + tokens_per_source
        val_limit = len(val_tok) + val_per_source
        task_start = len(task_pl)
        eval_start = len(eval_set)

        with open(os.path.join(DATA_DIR, train_file), 'rb') as f:
            for record in ijson.items(f, 'item'):
                s["docs_scanned"] += 1
                text = record["text"]
                # split key: documents with the same opening go to the same partition
                key = hashlib.md5(text[:500].encode()).hexdigest()
                full_key = hashlib.md5(" ".join(text.split()).encode()).hexdigest()
                if full_key in seen:
                    s["dup_dropped"] += 1
                    continue
                seen.add(full_key)
                words = text.split()
                if len(words) < 50:
                    s["under_50_words_dropped"] += 1
                    continue
                s["docs_kept"] += 1
                is_val = int(key[:8], 16) % 100 < VAL_DOC_PCT

                if is_val:
                    s["val_docs"] += 1
                    val_doc_keys.append(key)
                    if len(eval_set) - eval_start < eval_per_source and len(words) >= min_words:
                        p, t = word_window(words)
                        eval_set.append({"source": src, "doc_key": key, "prompt": p, "reference": t})
                    if len(val_tok) + BLOCK <= val_limit:
                        blocks = doc_blocks(tokenizer.encode(text))
                        if blocks is not None:
                            add_blocks(val_tok, blocks, val_limit)
                else:
                    s["train_docs"] += 1
                    need_blocks = len(train_tok) + BLOCK <= train_limit
                    need_task = len(task_pl) - task_start < task_per_source and len(words) >= min_words
                    if need_blocks:
                        toks = tokenizer.encode(text)
                        blocks = doc_blocks(toks)
                        if blocks is None:
                            s["train_docs_under_1_block"] += 1
                        else:
                            add_blocks(train_tok, blocks, train_limit)
                    if need_task:
                        p, t = word_window(words)
                        p_ids = tokenizer.encode(p)
                        t_ids = tokenizer.encode(t)
                        if len(p_ids) > PROMPT_MAX_TOKENS:
                            s["task_prompt_truncated"] += 1
                            p_ids = p_ids[-PROMPT_MAX_TOKENS:]
                        if len(t_ids) > TARGET_MAX_TOKENS:
                            s["task_target_overlength"] += 1
                            t_ids = t_ids[:TARGET_MAX_TOKENS]
                        task_tok.extend(p_ids)
                        task_tok.extend(t_ids)
                        task_off.append(len(task_tok))
                        task_pl.append(len(p_ids))

                done_blocks = len(train_tok) + BLOCK > train_limit and len(val_tok) + BLOCK > val_limit
                done_pools = (len(task_pl) - task_start >= task_per_source
                              and len(eval_set) - eval_start >= eval_per_source)
                if done_blocks and done_pools:
                    break
                if s["docs_scanned"] % 20000 == 0:
                    print(f"  {s['docs_scanned']} docs | train={len(train_tok)} val={len(val_tok)} "
                          f"task={len(task_pl) - task_start} eval={len(eval_set) - eval_start}")

        s["task_examples"] = len(task_pl) - task_start
        s["eval_examples"] = len(eval_set) - eval_start
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

    stats_out = {}
    for k in stats:
        stats_out[k] = dict(stats[k])
    eval_doc_keys = []
    for e in eval_set:
        eval_doc_keys.append(e["doc_key"])
    data_meta = {
        "seed": SEED, "block": BLOCK, "prompt_max_tokens": PROMPT_MAX_TOKENS,
        "target_max_tokens": TARGET_MAX_TOKENS, "val_doc_pct": VAL_DOC_PCT,
        "train_tokens_requested": TRAIN_TOKENS, "eval_set_size": EVAL_SET_SIZE, "tokenizer_md5": TOKENIZER_MD5,
        "train_tokens_actual": int(len(train_blocks) * BLOCK), "val_tokens_actual": int(len(val_blocks) * BLOCK),
        "split_rule": "doc is validation iff int(md5(text[:500])[:8], 16) % 100 < VAL_DOC_PCT",
        "stats": stats_out, "val_doc_keys": val_doc_keys, "eval_doc_keys": eval_doc_keys,
    }
    np.save(os.path.join(OUTPUT_DIR, CACHE["train"]), train_blocks)
    np.save(os.path.join(OUTPUT_DIR, CACHE["val"]), val_blocks)
    np.savez(os.path.join(OUTPUT_DIR, CACHE["task"]), tokens=task_tokens, offsets=task_offsets, plen=task_plen)
    with open(os.path.join(OUTPUT_DIR, CACHE["eval"]), "w") as f:
        json.dump(eval_set, f)
    with open(os.path.join(OUTPUT_DIR, CACHE["meta"]), "w") as f:
        json.dump(data_meta, f)

print(f"Train blocks: {len(train_blocks)} ({len(train_blocks) * BLOCK / 1e6:.0f}M tokens) | val blocks: {len(val_blocks)}")
print(f"Task pool: {len(task_plen)} examples | eval set: {len(eval_set)}")

# tuning set and audit set: one example per document group, 50 per source each
by_src = {}
groups_seen = set()
for e in eval_set:
    if e["doc_key"] in groups_seen:
        continue
    groups_seen.add(e["doc_key"])
    if e["source"] not in by_src:
        by_src[e["source"]] = []
    by_src[e["source"]].append(e)
half = max(1, EVAL_SET_SIZE // 8)
TUNE_SET = []
AUDIT_SET = []
for src in by_src:
    TUNE_SET.extend(by_src[src][:half])
    AUDIT_SET.extend(by_src[src][half:2 * half])
print(f"Held-out prompts: tuning {len(TUNE_SET)} | audit {len(AUDIT_SET)}")

# fixed seeded sample of validation blocks across all sources
VAL_IDX = np.sort(np.random.default_rng(SEED + 5).choice(len(val_blocks), min(VAL_SAMPLE_BLOCKS, len(val_blocks)), replace=False))

tokenizer._encode_cache = {}
gc.collect()


class EpochSampler:
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
    if final_start is not None and update_idx >= final_start:
        return "task"
    return "lm"


def make_microbatch(obj):
    # lm: one 1025-token block per row; task: prompt + continuation, loss on the continuation only
    if obj == "lm":
        blk = train_blocks[block_sampler.take(BATCH_SIZE)].astype(np.int64)
        return blk[:, :-1], blk[:, 1:]
    seqs = []
    plens = []
    for j in task_sampler.take(BATCH_SIZE):
        a = task_offsets[j]
        z = task_offsets[j + 1]
        seqs.append(task_tokens[a:z].astype(np.int64))
        plens.append(int(task_plen[j]))
    T = max([len(s) for s in seqs]) - 1
    assert T <= MODEL_SEQ_LEN
    x = np.full((BATCH_SIZE, T), PAD_ID, dtype=np.int64)
    y = np.full((BATCH_SIZE, T), IGNORE_INDEX, dtype=np.int64)
    for i in range(BATCH_SIZE):
        s = seqs[i]
        pl = plens[i]
        n = len(s) - 1
        x[i, :n] = s[:-1]
        y[i, pl - 1:n] = s[pl:]
    return x, y


def to_device(x, y):
    x = torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True)
    y = torch.from_numpy(np.ascontiguousarray(y)).to(device, non_blocking=True)
    return x, y


cfg = LMConfig(
    vocab_size=VOCAB_SIZE, d_model=MODEL_DIM, n_layers=MODEL_LAYERS, n_heads=MODEL_HEADS,
    kv_latent_dim=KV_LATENT, rope_dim=ROPE_DIM, d_ff=MODEL_FF, max_seq_len=MODEL_SEQ_LEN, dropout=MODEL_DROPOUT,
    engram_layers=tuple(ENGRAM_LAYERS), engram_orders=tuple(ENGRAM_ORDERS),
    engram_heads=ENGRAM_HEADS, engram_head_dim=ENGRAM_HEAD_DIM, engram_rows=ENGRAM_ROWS,
)

torch.manual_seed(SEED)
model = DecoderLM(cfg).to(device)
model.set_engram_compression(ENGRAM_COMPRESS)


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


model.apply(init_weights)

residual_std = 0.02 / math.sqrt(2 * MODEL_LAYERS)
for block in model.blocks:
    torch.nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
    torch.nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=residual_std)
    if block.engram is not None:
        # zero value path and convolution: Engram starts as an exact no-op
        torch.nn.init.zeros_(block.engram.w_v.weight)
        torch.nn.init.zeros_(block.engram.conv.weight)

n_params = sum(p.numel() for p in model.parameters())
engram_params = 0
for b in model.blocks:
    if b.engram is not None:
        engram_params += sum(p.numel() for p in b.engram.parameters())
print(f"Model parameters: {n_params} ({n_params / 1e6:.1f}M) | Engram {engram_params / 1e6:.1f}M")
assert 50e6 <= n_params <= 300e6

use_amp = torch.cuda.is_available()


def amp_ctx():
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return torch.autocast(device_type="cpu", dtype=torch.float16, enabled=False)


def choose_attention_backend():
    # use sdpa only if it matches the manual attention and is faster on this gpu
    if not torch.cuda.is_available():
        return "manual"
    mb = copy.deepcopy(model).train()
    xb = torch.from_numpy(train_blocks[:BATCH_SIZE, :-1].astype(np.int64)).to(device)
    timings = {}
    outs = {}
    for name, flag in [("manual", False), ("sdpa", True)]:
        mb.set_attention_backend(flag)
        with torch.no_grad(), amp_ctx():
            outs[name] = mb(xb).float()
        for i in range(4):
            if i == 1:
                torch.cuda.synchronize()
                t0 = time.time()
            mb.zero_grad(set_to_none=True)
            with amp_ctx():
                lo = F.cross_entropy(mb(xb).float().view(-1, VOCAB_SIZE), xb.reshape(-1))
            lo.backward()
        torch.cuda.synchronize()
        timings[name] = (time.time() - t0) / 3
    rel = float((outs["manual"] - outs["sdpa"]).abs().max() / outs["manual"].abs().max())
    faster = timings["sdpa"] < 0.95 * timings["manual"]
    del mb, outs, xb
    torch.cuda.empty_cache()
    print(f"attention benchmark: manual {timings['manual']:.3f}s, sdpa {timings['sdpa']:.3f}s, rel diff {rel:.6f}")
    if rel < 2e-2 and faster:
        return "sdpa"
    return "manual"


ATTENTION_BACKEND = choose_attention_backend()
model.set_attention_backend(ATTENTION_BACKEND == "sdpa")
print(f"Attention backend: {ATTENTION_BACKEND}")

# weight decay on 2d weights only, engram tables get 5x lr and no weight decay
decay_params = []
no_decay_params = []
table_params = []
for name, param in model.named_parameters():
    if name.endswith(".engram.table.weight"):
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


def lr_at(update, decay_start, planned_total):
    # warmup, constant peak, then cosine decay once decay_start is known
    if update < WARMUP_STEPS:
        return PEAK_LR * update / WARMUP_STEPS
    if decay_start is None or update < decay_start:
        return PEAK_LR
    p = min(1.0, (update - decay_start) / max(1, planned_total - decay_start))
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * p)) * (PEAK_LR - MIN_LR)


def set_lr(lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg["lr_mult"]


def decay_plan(decay_start):
    total = decay_start + max(1, int(DECAY_FRAC * decay_start))
    final_start = max(decay_start, total - int(FINAL_TASK_FRAC * total))
    return total, final_start


NEVER_GEN_T = torch.tensor([PAD_ID, 1, BOS_ID, EOS_ID], device=device)


def generate(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS, target_words=TARGET_WORDS,
             temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, repetition_penalty=REPETITION_PENALTY,
             seed=SEED):
    model.eval()
    p_ids = tokenizer.encode(prompt)
    prompt_truncated = len(p_ids) > PROMPT_MAX_TOKENS
    tokens = p_ids[-PROMPT_MAX_TOKENS:]
    budget = min(max_tokens, MODEL_SEQ_LEN - len(tokens))
    gen = None
    if temperature > 0:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    generated = []
    complete = False
    cache = None
    with torch.inference_mode():
        nxt = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(budget):
            with amp_ctx():
                logits, cache = model.step(nxt, cache)
            logits = logits[0, -1].float()
            logits[NEVER_GEN_T] = float('-inf')
            if repetition_penalty != 1.0 and generated:
                ids = torch.tensor(sorted(set(generated[-100:])), device=device)
                vals = logits[ids]
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
                tok = int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen))
            tokens.append(tok)
            generated.append(tok)
            nxt = torch.tensor([[tok]], dtype=torch.long, device=device)
            n_words = len(tokenizer.decode(generated).split())
            if n_words > target_words or (n_words == target_words and tokenizer._ends_word.get(tok, False)):
                complete = True
                break
    words = tokenizer.decode(generated).split()[:target_words]
    text = " ".join(words)
    info = {"n_words": len(words), "n_tokens": len(generated), "hit_token_cap": not complete,
            "prompt_truncated": prompt_truncated}
    return text, info


def val_loss(max_batches=None):
    model.eval()
    if max_batches is None:
        idx = np.arange(len(val_blocks))
    else:
        idx = VAL_IDX[:max_batches * BATCH_SIZE]
    tot = 0.0
    ntok = 0
    with torch.no_grad():
        for a in range(0, len(idx), BATCH_SIZE):
            blk = val_blocks[idx[a:a + BATCH_SIZE]].astype(np.int64)
            x, y = to_device(blk[:, :-1], blk[:, 1:])
            with amp_ctx():
                lg = model(x)
            tot += F.cross_entropy(lg.float().view(-1, lg.size(-1)), y.reshape(-1), reduction="sum").item()
            ntok += y.numel()
    model.train()
    return tot / max(1, ntok)


def repetition_rate(text, n=4):
    w = text.split()
    grams = []
    for i in range(len(w) - n + 1):
        grams.append(tuple(w[i:i + n]))
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def evaluate_heldout(tag):
    # bertscore with roberta-large on the tuning prompts
    t0 = time.time()
    preds = []
    infos = []
    for i in range(len(TUNE_SET)):
        text, info = generate(model, tokenizer, TUNE_SET[i]["prompt"], seed=SEED + i)
        preds.append(text)
        infos.append(info)
    refs = [ex["reference"] for ex in TUNE_SET]
    scorer = BERTScorer(model_type="roberta-large", lang="en", device=str(device))
    P, R, F1 = scorer.score(preds, refs)
    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vl = val_loss(MAX_VAL_BATCHES)
    short = 0
    cap_hits = 0
    for x in infos:
        short += x["n_words"] < TARGET_WORDS
        cap_hits += x["hit_token_cap"]
    reps = [repetition_rate(p) for p in preds]
    res = {"tag": tag, "update": run["update"], "bertscore_P": P.mean().item(), "bertscore_R": R.mean().item(),
           "bertscore_F1": F1.mean().item(), "val_loss": vl, "val_ppl": math.exp(min(vl, 20)),
           "short_outputs": short, "token_cap_hits": cap_hits, "repetition_4gram": float(np.mean(reps)),
           "n": len(TUNE_SET), "seconds": time.time() - t0}
    with open(os.path.join(OUTPUT_DIR, f"eval_{tag}.json"), "w") as f:
        json.dump({"result": res, "predictions": preds}, f)
    print(f"  eval {tag} update {res['update']}: BERTScore F1 {res['bertscore_F1']:.4f} | val loss {vl:.4f} | "
          f"short {short} | cap hits {cap_hits} | rep4 {res['repetition_4gram']:.3f}")
    model.train()
    return res


OBJS = ["lm", "task"]


def new_run_state():
    return {
        "update": 0, "attempts": 0, "skipped_updates": 0,
        "planned_total": None, "final_start": None, "horizon_info": None,
        "tp": {"lm": []}, "active_train_sec": 0.0, "eval_sec": 0.0,
        "updates": {"lm": 0, "task": 0}, "examples": {"lm": 0, "task": 0}, "loss_tokens": {"lm": 0, "task": 0},
        "history": [], "val_history": [], "evals": [], "checkpoints": [],
        "pre_final_saved": False, "best_phase1": None, "done": False,
        "decay_start": None, "decay_reason": None,
    }


def save_checkpoint(name, kind):
    run["checkpoints"].append({"name": name, "kind": kind, "update": run["update"], "unix": time.time()})
    if torch.cuda.is_available():
        cuda_rng = torch.cuda.get_rng_state_all()
    else:
        cuda_rng = None
    state = {
        "kind": kind, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "model_config": cfg.to_dict(), "run": run,
        "samplers": {"block": block_sampler.state(), "task": task_sampler.state()},
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": cuda_rng},
        "engram_compress": ENGRAM_COMPRESS,
    }
    torch.save(state, os.path.join(OUTPUT_DIR, name))
    log = {}
    for k in run:
        if k != "tp":
            log[k] = run[k]
    with open(os.path.join(OUTPUT_DIR, "run_log.json"), "w") as f:
        json.dump(log, f, default=float)
    print(f"  saved {name} at update {run['update']} ({kind})")


def do_eval(tag):
    t0 = time.time()
    res = evaluate_heldout(tag)
    run["evals"].append(res)
    if run["best_phase1"] is None or res["bertscore_F1"] > run["best_phase1"]["bertscore_F1"]:
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, BEST_PHASE1))
        run["best_phase1"] = {"tag": tag, "update": run["update"], "bertscore_F1": res["bertscore_F1"]}
        print(f"  new best Phase 1 checkpoint ({tag})")
    run["eval_sec"] += time.time() - t0


def start_decay(reason):
    total, fs = decay_plan(run["update"])
    run["decay_start"] = run["update"]
    run["planned_total"] = total
    run["final_start"] = fs
    run["decay_reason"] = reason
    save_checkpoint(STABLE_END_CKPT, "end of stable phase")
    print(f"Decay starts at update {run['update']} ({reason}). Run ends at update {total}, "
          f"task-matched segment from {fs}")


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
    ENGRAM_COMPRESS = ck["engram_compress"].cpu()
    model.set_engram_compression(ENGRAM_COMPRESS)
    ATTENTION_BACKEND = run["attention_backend"]
    model.set_attention_backend(ATTENTION_BACKEND == "sdpa")
    block_sampler.load(ck["samplers"]["block"])
    task_sampler.load(ck["samplers"]["task"])
    random.setstate(ck["rng"]["python"])
    np.random.set_state(ck["rng"]["numpy"])
    torch.set_rng_state(ck["rng"]["torch"].cpu())
    if torch.cuda.is_available() and ck["rng"]["cuda"] is not None:
        cuda_states = []
        for st in ck["rng"]["cuda"]:
            cuda_states.append(st.cpu())
        torch.cuda.set_rng_state_all(cuda_states)
    del ck
    print(f"  update {run['update']} / planned {run['planned_total']}")
else:
    print("Starting a new run.")

run["attention_backend"] = ATTENTION_BACKEND
last_ckpt = time.time()
win = {"lm": [0.0, 0], "task": [0.0, 0]}
consecutive_skips = 0
model.train()

while True:
    u = run["update"]
    if run["planned_total"] is not None and u >= run["planned_total"]:
        run["done"] = True
        break

    if run["final_start"] is not None and u >= run["final_start"] and not run["pre_final_saved"]:
        save_checkpoint(FOUNDATION_CKPT, "general LM, before task-matched segment")
        run["pre_final_saved"] = True
        do_eval("prefinal")
        save_checkpoint(LATEST_CKPT, "latest (before task-matched segment)")
        last_ckpt = time.time()

    t0 = time.time()
    obj = objective_for(u, run["final_start"])
    micro = []
    for _ in range(GRAD_ACCUM):
        micro.append(make_microbatch(obj))
    n_valid = 0
    for mb in micro:
        n_valid += int((mb[1] != IGNORE_INDEX).sum())
    lr = lr_at(u, run["decay_start"], run["planned_total"])
    set_lr(lr)

    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for mb in micro:
        bx, by = to_device(mb[0], mb[1])
        with amp_ctx():
            logits = model(bx)
        loss_tok = F.cross_entropy(logits.float().view(-1, logits.size(-1)), by.reshape(-1),
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
    run["active_train_sec"] += dt
    run["attempts"] += 1

    if (use_amp and scaler.get_scale() < scale_before) or not math.isfinite(loss_sum):
        # fp16 overflow: no parameter change and the schedule does not advance
        run["skipped_updates"] += 1
        consecutive_skips += 1
        if consecutive_skips >= 50:
            raise RuntimeError(f"{consecutive_skips} consecutive skipped updates at update {u}")
        continue
    consecutive_skips = 0
    run["update"] = u + 1
    run["updates"][obj] += 1
    run["examples"][obj] += BATCH_SIZE * GRAD_ACCUM
    run["loss_tokens"][obj] += n_valid
    win[obj][0] += loss_sum
    win[obj][1] += n_valid

    if MEASURE_FROM <= u < MEASURE_TO and obj in run["tp"]:
        run["tp"][obj].append(dt)
    if run["horizon_info"] is None and run["update"] >= MEASURE_TO:
        per = float(np.mean(run["tp"]["lm"]))
        run["horizon_info"] = {"s_per_update": per}
        print(f"Throughput: {per:.2f}s per update")

    # end the stable phase if the time cap would be passed once the decay is added
    if run["decay_start"] is None and run["update"] >= WARMUP_STEPS:
        if run["active_train_sec"] * (1 + DECAY_FRAC) >= MAX_TRAIN_HOURS * 3600 * 0.97:
            start_decay(f"time cap: decay must start now to finish within {MAX_TRAIN_HOURS}h")

    if run["update"] % LOG_EVERY == 0:
        parts = ""
        entry = {"update": run["update"], "lr": lr, "gnorm": gnorm, "scale": scaler.get_scale(),
                 "active_train_sec": run["active_train_sec"]}
        for o in OBJS:
            if win[o][1] > 0:
                parts += f"{o}={win[o][0] / win[o][1]:.3f} "
                entry[f"loss_{o}"] = win[o][0] / win[o][1]
        print(f"upd {run['update']}/{run['planned_total']} | {parts}| lr {lr:.2e} | gnorm {gnorm:.2f} | "
              f"skipped {run['skipped_updates']} | {dt:.2f}s/upd | train {run['active_train_sec'] / 3600:.2f}h")
        run["history"].append(entry)
        win = {"lm": [0.0, 0], "task": [0.0, 0]}

    if run["update"] % VAL_EVERY_UPDATES == 0:
        tv = time.time()
        vl = val_loss(MAX_VAL_BATCHES)
        run["eval_sec"] += time.time() - tv
        run["val_history"].append({"update": run["update"], "val_loss": vl})
        print(f"  val loss {vl:.4f} (ppl {math.exp(min(vl, 20)):.2f})")
        hist = [v["val_loss"] for v in run["val_history"]]
        if run["decay_start"] is None and run["update"] >= MIN_STABLE_UPDATES and len(hist) > PLATEAU_WINDOW:
            rel = (hist[-PLATEAU_WINDOW - 1] - hist[-1]) / hist[-PLATEAU_WINDOW - 1]
            if rel < PLATEAU_REL_TOL:
                start_decay(f"validation plateau: improved {rel * 100:.2f}% over the last {PLATEAU_WINDOW} checks")

    if run["update"] % EVAL_EVERY_UPDATES == 0:
        if run["final_start"] is None or run["update"] < run["final_start"]:
            do_eval(f"u{run['update']}")

    if run["update"] % CKPT_EVERY_UPDATES == 0 or time.time() - last_ckpt >= CKPT_EVERY_MIN * 60:
        save_checkpoint(LATEST_CKPT, "latest")
        last_ckpt = time.time()

    if run["update"] % SNAPSHOT_EVERY == 0 and run["update"] >= SNAPSHOT_START:
        half_state = {}
        for k, v in model.state_dict().items():
            half_state[k] = v.half()
        torch.save(half_state, os.path.join(OUTPUT_DIR, "arch4_snap_" + str(run["update"]).zfill(6) + ".pt"))

save_checkpoint(LATEST_CKPT, "latest (training complete)")
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
do_eval("final")
final_val_loss = val_loss()
final_ppl = math.exp(min(final_val_loss, 20))
run["final_val_loss"] = final_val_loss
save_checkpoint(LATEST_CKPT, "latest (training complete)")
print(f"Final val loss {final_val_loss:.4f} (ppl {final_ppl:.2f}) | best Phase 1 checkpoint: {run['best_phase1']}")

h = run["history"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for o in OBJS:
    xs = []
    ys = []
    for x in h:
        if f"loss_{o}" in x:
            xs.append(x["update"])
            ys.append(x[f"loss_{o}"])
    if xs:
        axes[0].plot(xs, ys, label=o)
if run["final_start"]:
    axes[0].axvline(run["final_start"], color="gray", ls="--", label="task-matched segment")
axes[0].set_title("Training loss per objective")
axes[0].legend()
axes[1].plot([v["update"] for v in run["val_history"]], [v["val_loss"] for v in run["val_history"]], marker="o")
axes[1].set_title("Validation loss")
axes[2].plot([x["update"] for x in h], [x["lr"] for x in h])
axes[2].set_title("Learning rate")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"), dpi=150)

with open(os.path.join(DATA_DIR, TEST_FILE), "r", encoding="utf-8") as f:
    test_data = json.load(f)
print(f"Number of test examples: {len(test_data)}")

# test predictions use the best checkpoint by held-out tuning F1
best = run["best_phase1"]
if best is not None and best["tag"] != "final":
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, BEST_PHASE1), map_location=device))
    print(f"Using the best checkpoint by held-out tuning F1: {best}")
else:
    print(f"Using the final model (best by held-out tuning F1: {best})")

predictions = {}
gen_report = Counter()
items = list(test_data.items())
t0 = time.time()
for i in range(len(items)):
    example_id, record = items[i]
    text, info = generate(model, tokenizer, record["text"], seed=SEED + 100000 + i)
    predictions[str(example_id)] = {"text": text}
    gen_report["short_outputs"] += info["n_words"] < TARGET_WORDS
    gen_report["token_cap_hits"] += info["hit_token_cap"]
    gen_report["prompt_truncated"] += info["prompt_truncated"]
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(items)} | {time.time() - t0:.0f}s")
print(f"Generation complete in {(time.time() - t0) / 60:.1f} min | {dict(gen_report)}")

assert len(predictions) == len(test_data)
PREDICTION_PATH = os.path.join(OUTPUT_DIR, "prediction.json")
SUBMISSION_ZIP = os.path.join(OUTPUT_DIR, "submission.zip")
with open(PREDICTION_PATH, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(PREDICTION_PATH, arcname="prediction.json")
print(f"submission.zip created: {SUBMISSION_ZIP} | predictions: {len(predictions)}")

metrics = {
    "architecture": "deep_decoder_mla_qknorm_gated_attention_engram",
    "params": n_params, "engram_params": engram_params, "model_config": cfg.to_dict(),
    "updates": run["update"], "planned_total": run["planned_total"], "final_start": run["final_start"],
    "decay_start": run["decay_start"], "decay_reason": run["decay_reason"],
    "skipped_updates": run["skipped_updates"], "updates_per_objective": run["updates"],
    "active_train_hours": run["active_train_sec"] / 3600, "evals": run["evals"], "best_phase1": run["best_phase1"],
    "val_history": run["val_history"], "final_val_loss": run["final_val_loss"],
    "generation_config": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
                          "repetition_penalty": REPETITION_PENALTY},
    "test_generation_report": dict(gen_report),
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2, default=float)
print("metrics.json saved")
