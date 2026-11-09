# Batched KV-cached sampling for Arch 2 and MBR (ROUGE-1/2 agreement) over 16 samples per prompt
import os
import json
import time
import zipfile
from collections import Counter
import numpy as np
import torch
from tokenizer import BPETokenizer
from arch2_model import LMConfig, DecoderLM

DATA_DIR = "data"
TEST_FILE = "test.json"
TOKENIZER_PATH = "legal_bpe_16384.json"
ARCH2_PATH = os.path.join("outputs", "arch2", "model.pt")
OUTPUT_DIR = os.path.join("outputs", "mbr16")

VOCAB_SIZE = 16384
PAD_ID = 0
NEVER_GEN = [0, 1, 2, 3]       # pad, unk, bos, eos are never generated
MODEL_SEQ_LEN = 1024
TARGET_WORDS = 128
PROMPT_KEEP = 704              # most recent prompt tokens kept
MAX_NEW_TOKENS = MODEL_SEQ_LEN - PROMPT_KEEP
MAX_ROWS = 64                  # rows generated together in one batch

T35 = {"temperature": 0.5, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.20}
N_SAMPLES = 16
TEST_SEED = 9000000

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = torch.cuda.is_available()
tokenizer = BPETokenizer.load(TOKENIZER_PATH)


def amp_ctx():
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return torch.autocast(device_type="cpu", dtype=torch.float16, enabled=False)


def load_arch2(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    cfg = LMConfig(vocab_size=VOCAB_SIZE, d_model=640, n_layers=12, n_heads=10, n_kv_heads=2, d_ff=1706,
                   max_seq_len=MODEL_SEQ_LEN, dropout=0.0)
    model = DecoderLM(cfg)
    weights = {}
    for k in state:
        weights[k] = state[k].float()
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


def kv_step(model, idx, cache=None, pad=None):
    # same math as DecoderLM.forward but reuses the keys and values of earlier tokens
    B, T = idx.shape
    if cache is None:
        pos = 0
        if pad is None:
            pad = torch.zeros(B, dtype=torch.long, device=idx.device)
    else:
        pos = cache["pos"]
        pad = cache["pad"]
    S = pos + T
    q_abs = torch.arange(pos, S, device=idx.device)
    k_abs = torch.arange(S, device=idx.device)
    positions = (q_abs[None, :] - pad[:, None]).clamp(min=0)
    causal = k_abs[None, None, :] <= q_abs[None, :, None]
    not_pad = (k_abs[None, None, :] >= pad[:, None, None]) | (k_abs[None, None, :] == q_abs[None, :, None])
    allowed = causal & not_pad

    x = model.tok_emb(idx)
    new_kv = []
    for li, blk in enumerate(model.blocks):
        a = blk.attn
        h = blk.ln1(x)
        q = a.q_proj(h).view(B, T, a.n_heads, a.d_head).transpose(1, 2)
        k = a.k_proj(h).view(B, T, a.n_kv_heads, a.d_head).transpose(1, 2)
        v = a.v_proj(h).view(B, T, a.n_kv_heads, a.d_head).transpose(1, 2)
        cos = a.rope_cos[positions].unsqueeze(1)
        sin = a.rope_sin[positions].unsqueeze(1)
        d = a.d_head // 2
        q = torch.cat([q[..., :d] * cos - q[..., d:] * sin, q[..., d:] * cos + q[..., :d] * sin], dim=-1)
        k = torch.cat([k[..., :d] * cos - k[..., d:] * sin, k[..., d:] * cos + k[..., :d] * sin], dim=-1)
        if cache is not None:
            k = torch.cat([cache["kv"][li][0], k], dim=2)
            v = torch.cat([cache["kv"][li][1], v], dim=2)
        new_kv.append((k, v))
        if a.n_rep > 1:
            kk = k.repeat_interleave(a.n_rep, dim=1)
            vv = v.repeat_interleave(a.n_rep, dim=1)
        else:
            kk = k
            vv = v
        scores = (q @ kk.transpose(-2, -1)) * a.scale
        scores = scores.masked_fill(~allowed[:, None], float('-inf'))
        out = (torch.softmax(scores, dim=-1) @ vv).transpose(1, 2).contiguous().view(B, T, -1)
        x = x + a.out_proj(out)
        x = x + blk.ffn(blk.ln2(x))
    logits = model.head(model.ln_f(x))
    return logits, {"pos": S, "kv": new_kv, "pad": pad}


def expand_cache(cache, idx):
    kv = []
    for k, v in cache["kv"]:
        kv.append((k.index_select(0, idx), v.index_select(0, idx)))
    return {"pos": cache["pos"], "kv": kv, "pad": cache["pad"].index_select(0, idx)}


def filter_logits(logits, cfg, gens, active):
    B, V = logits.shape
    dev = logits.device
    lg = logits.clone()
    lg[:, NEVER_GEN] = float('-inf')

    # repetition penalty on the last 100 generated tokens
    rows_idx = []
    tok_idx = []
    for r in range(B):
        if active[r] and cfg["repetition_penalty"] != 1.0 and gens[r]:
            ids = sorted(set(gens[r][-100:]))
            rows_idx += [r] * len(ids)
            tok_idx += ids
    if rows_idx:
        rp = torch.full((B, 1), cfg["repetition_penalty"], device=dev, dtype=lg.dtype)
        pen = torch.zeros(B, V, dtype=torch.bool, device=dev)
        pen[torch.tensor(rows_idx, device=dev), torch.tensor(tok_idx, device=dev)] = True
        lg = torch.where(pen, torch.where(lg > 0, lg / rp, lg * rp), lg)

    if cfg["temperature"] <= 0:
        return lg
    temp = torch.full((B, 1), cfg["temperature"], device=dev, dtype=lg.dtype)
    lg = lg / temp

    top_p = torch.full((B, 1), cfg["top_p"], device=dev, dtype=lg.dtype)
    if cfg["top_k"] > 0:
        k = min(cfg["top_k"], V)
        tv, ti = torch.topk(lg, k, dim=-1)
        kth = tv[:, k - 1:k]
        lg = lg.masked_fill(lg < kth, float('-inf'))
        # top-p over the top-k survivors
        tv = tv.masked_fill(tv < kth, float('-inf'))
        cum = torch.cumsum(torch.softmax(tv, dim=-1), dim=-1)
        rm = cum > top_p
        rm[:, 1:] = rm[:, :-1].clone()
        rm[:, 0] = False
        rm &= top_p < 1.0
        lg.scatter_(-1, ti, lg.gather(-1, ti).masked_fill(rm, float('-inf')))
    elif cfg["top_p"] < 1.0:
        sl, si = torch.sort(lg, dim=-1, descending=True)
        cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
        rm = cum > top_p
        rm[:, 1:] = rm[:, :-1].clone()
        rm[:, 0] = False
        lg = lg.masked_fill(torch.zeros_like(rm).scatter(-1, si, rm), float('-inf'))
    return lg


def generate_batch(model, bases, cfg, seeds, owner):
    # rows are left padded and each row samples with its own seeded generator
    B = len(bases)
    L = 0
    for b in bases:
        L = max(L, len(b))
    pads = []
    rows = []
    for b in bases:
        pads.append(L - len(b))
        rows.append([PAD_ID] * (L - len(b)) + list(b))
    padded = max(pads) > 0
    pad_t = torch.tensor(pads, dtype=torch.long, device=device)
    gens = []
    for r in range(B):
        gens.append([])
    done = [False] * B
    rngs = []
    for s in seeds:
        rngs.append(torch.Generator(device=device).manual_seed(int(s)))

    # rows with the same prompt index share one prompt pass
    uniq = {}
    row_to_u = []
    for w in owner:
        if w not in uniq:
            uniq[w] = len(uniq)
        row_to_u.append(uniq[w])
    first = []
    for u in range(len(uniq)):
        first.append(row_to_u.index(u))

    cache = None
    nxt = torch.tensor(rows, dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(min(MAX_NEW_TOKENS, MODEL_SEQ_LEN - L)):
            if all(done):
                break
            with amp_ctx():
                if cache is None:
                    fi = torch.tensor(first, device=device)
                    if padded:
                        out, cache = kv_step(model, nxt.index_select(0, fi), None, pad=pad_t.index_select(0, fi))
                    else:
                        out, cache = kv_step(model, nxt.index_select(0, fi), None)
                    ri = torch.tensor(row_to_u, device=device)
                    cache = expand_cache(cache, ri)
                    logits = out[:, -1].float().index_select(0, ri)
                else:
                    out, cache = kv_step(model, nxt, cache)
                    logits = out[:, -1].float()
            active = []
            for d in done:
                active.append(not d)
            lg = filter_logits(logits, cfg, gens, active)
            noise = torch.zeros_like(lg)
            if cfg["temperature"] > 0:
                for r in range(B):
                    if active[r]:
                        u = torch.rand(lg.shape[1:], generator=rngs[r], device=lg.device).clamp_(1e-10, 1 - 1e-10)
                        noise[r] = -torch.log(-torch.log(u))
            toks = (lg + noise).argmax(dim=-1).tolist()
            for r in range(B):
                if done[r]:
                    rows[r].append(PAD_ID)
                    continue
                tok = toks[r]
                rows[r].append(tok)
                gens[r].append(tok)
                n_words = len(tokenizer.decode(gens[r]).split())
                if n_words > TARGET_WORDS or (n_words == TARGET_WORDS and tokenizer._ends_word.get(tok, False)):
                    done[r] = True
            last = []
            for row in rows:
                last.append([row[-1]])
            nxt = torch.tensor(last, dtype=torch.long, device=device)
    out = []
    for r in range(B):
        words = tokenizer.decode(gens[r]).split()[:TARGET_WORDS]
        out.append((" ".join(words), len(words)))
    return out


def generate_many(model, jobs, cfg):
    # jobs is a list of (prompt, seeds)
    enc = []
    for prompt, seeds in jobs:
        assert 0 < len(seeds) <= MAX_ROWS
        enc.append(tokenizer.encode(prompt)[-PROMPT_KEEP:])
    results = [None] * len(jobs)
    i = 0
    t0 = time.time()
    while i < len(jobs):
        j = i
        n_rows = 0
        while j < len(jobs) and n_rows + len(jobs[j][1]) <= MAX_ROWS:
            n_rows += len(jobs[j][1])
            j += 1
        bases = []
        seeds = []
        owner = []
        for k in range(i, j):
            for s in jobs[k][1]:
                bases.append(enc[k])
                seeds.append(s)
                owner.append(k)
        outs = generate_batch(model, bases, cfg, seeds, owner)
        for k in range(i, j):
            results[k] = []
        for o, w in zip(outs, owner):
            results[w].append(o)
        if j // 50 > i // 50 or j == len(jobs):
            print(f"  {j}/{len(jobs)} prompts | {(time.time() - t0) / 60:.1f} min")
        i = j
    return results


def rouge_f(a, b, n):
    ga = Counter()
    for i in range(len(a) - n + 1):
        ga[tuple(a[i:i + n])] += 1
    gb = Counter()
    for i in range(len(b) - n + 1):
        gb[tuple(b[i:i + n])] += 1
    ov = sum((ga & gb).values())
    if not ga or not gb or ov == 0:
        return 0.0
    p = ov / sum(ga.values())
    r = ov / sum(gb.values())
    return 2 * p * r / (p + r)


def mbr_select(cands):
    # index of the candidate with the highest mean ROUGE-1/2 F1 against the other candidates
    toks = [c.split() for c in cands]
    scores = []
    for i in range(len(toks)):
        u = []
        for j in range(len(toks)):
            if j != i:
                u.append(0.5 * (rouge_f(toks[i], toks[j], 1) + rouge_f(toks[i], toks[j], 2)))
        if u:
            scores.append(float(np.mean(u)))
        else:
            scores.append(0.0)
    return int(np.argmax(scores))


def pick_mbr(outs):
    # MBR among the samples that reach exactly 128 words (all samples if none do)
    ok = [o for o in outs if o[1] == TARGET_WORDS]
    if not ok:
        ok = outs
    return ok[mbr_select([o[0] for o in ok])][0]


def write_submission(preds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pj = os.path.join(out_dir, "prediction.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")


if __name__ == "__main__":
    model = load_arch2(ARCH2_PATH)
    with open(os.path.join(DATA_DIR, TEST_FILE), encoding="utf-8") as f:
        test_data = json.load(f)
    test_ids = [str(k) for k in test_data.keys()]

    jobs = []
    for i, tid in enumerate(test_ids):
        seeds = [TEST_SEED + i * 100 + k for k in range(N_SAMPLES)]
        jobs.append((test_data[tid]["text"], seeds))
    outs = generate_many(model, jobs, T35)

    preds = {}
    for tid, o in zip(test_ids, outs):
        preds[tid] = {"text": pick_mbr(o)}
    write_submission(preds, OUTPUT_DIR)
    print("Wrote", len(preds), "MBR x16 predictions to", OUTPUT_DIR)
