# A pool of 28 Arch 2 samples per test prompt (16 T35, 4 T36, 4 plain, 4 T34) and MBR over parts of it
import os
import json
import torch
import generate_mbr
from generate_mbr import load_arch2, generate_many, pick_mbr, write_submission, ARCH2_PATH, DATA_DIR, TEST_FILE, TEST_SEED

OUTPUT_DIR = os.path.join("outputs", "pool")

CONFIGS = {
    "t35": {"temperature": 0.5, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.20},
    "t36": {"temperature": 0.3, "top_p": 0.85, "top_k": 50, "repetition_penalty": 1.20},
    "plain": {"temperature": 0.4, "top_p": 0.92, "top_k": 50, "repetition_penalty": 1.15, "old_penalty": True},
    "t34": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.10},
}
POOL = [("t35", 16), ("t36", 4), ("plain", 4), ("t34", 4)]

# positions in the pool of 28
T35 = list(range(0, 16))
T36 = list(range(16, 20))
PLAIN = list(range(20, 24))
T34 = list(range(24, 28))
SELECTORS = {
    "mbr4b_t35": T35[4:8],
    "mbr8b_t35": T35[8:],
    "mbr12b_t35": T35[4:],
    "single_t35_b": [T35[1]],
    "mix4": T35[:1] + T36[:1] + PLAIN[:1] + T34[:1],
    "mix8": T35[:2] + T36[:2] + PLAIN[:2] + T34[:2],
    "mix12": T35[:3] + T36[:3] + PLAIN[:3] + T34[:3],
    "mix16": T35[:4] + T36 + PLAIN + T34,
    "mix20": T35[:8] + T36 + PLAIN + T34,
    "all28": T35 + T36 + PLAIN + T34,
}

base_filter = generate_mbr.filter_logits


def filter_with_old_penalty(logits, cfg, gens, active):
    # arch 2's own sampling divides every repeated logit by the penalty, also the negative ones
    if not cfg.get("old_penalty"):
        return base_filter(logits, cfg, gens, active)
    lg = logits.clone()
    for r in range(len(gens)):
        if active[r] and gens[r]:
            ids = torch.tensor(sorted(set(gens[r][-100:])), device=lg.device)
            lg[r, ids] = lg[r, ids] / cfg["repetition_penalty"]
    cfg = dict(cfg)
    cfg["repetition_penalty"] = 1.0
    return base_filter(lg, cfg, gens, active)


generate_mbr.filter_logits = filter_with_old_penalty

if __name__ == "__main__":
    model = load_arch2(ARCH2_PATH)
    with open(os.path.join(DATA_DIR, TEST_FILE), encoding="utf-8") as f:
        test_data = json.load(f)
    test_ids = [str(k) for k in test_data.keys()]

    pool = [[] for _ in test_ids]
    start = 0
    for name, n in POOL:
        jobs = []
        for i, tid in enumerate(test_ids):
            seeds = [TEST_SEED + i * 100 + k for k in range(start, start + n)]
            jobs.append((test_data[tid]["text"], seeds))
        outs = generate_many(model, jobs, CONFIGS[name])
        for i in range(len(test_ids)):
            pool[i] += outs[i]
        start += n

    for name, idx in SELECTORS.items():
        preds = {}
        for i, tid in enumerate(test_ids):
            preds[tid] = {"text": pick_mbr([pool[i][k] for k in idx])}
        write_submission(preds, os.path.join(OUTPUT_DIR, name))
        print("Wrote", name)
