# Final Phase 1 submission: retrieval fine-tuned Arch 2 + MBR x16 on prompts with a retrieved passage,
# the MBR x16 predictions of the plain Arch 2 on all other prompts
import os
import json
from generate_mbr import (tokenizer, load_arch2, generate_many, pick_mbr, write_submission, device,
                          T35, N_SAMPLES)
from retrieval_finetune import rag_ids

DATA_DIR = "data"
TEST_FILE = "test.json"
ARCH2_PATH = os.path.join("outputs", "arch2", "model.pt")
FT_PATH = os.path.join("outputs", "rag", "rag_ft.pt")
OVERLAP_PATH = os.path.join("outputs", "retrieval", "overlap_matches.json")
HELDOUT_PATH = os.path.join("outputs", "heldout_eval_set.json")
BASE_PRED_PATH = os.path.join("outputs", "mbr16", "prediction.json")
OUTPUT_DIR = os.path.join("outputs", "final")

TEST_SEED = 52000000
RUN_HELDOUT = False       # True: held-out BERTScore check (plain MBR x16 vs retrieval + MBR x16)


def rag_select(model, prompts_cands, cfg, rows_per_prompt, seed_base):
    # sampling: rows_per_prompt rows spread over the retrieved passages; greedy: one row per passage
    jobs = []
    owner = []
    for i, (prompt, cands) in enumerate(prompts_cands):
        if cfg["temperature"] > 0:
            n_rows = rows_per_prompt
        else:
            n_rows = len(cands)
        for c in range(len(cands)):
            rows = [r for r in range(n_rows) if r % len(cands) == c]
            if rows:
                ctx = tokenizer.decode(rag_ids(prompt, cands[c]["text"]))
                jobs.append((ctx, [seed_base + i * 100 + r for r in rows]))
                owner.append(i)
    outs = generate_many(model, jobs, cfg)
    per = []
    for _ in prompts_cands:
        per.append([])
    for i, o in zip(owner, outs):
        per[i].extend(o)
    return [pick_mbr(rows) for rows in per]


def plain_mbr(model, prompts, n, seed_base):
    jobs = []
    for i, p in enumerate(prompts):
        jobs.append((p, [seed_base + i * 100 + r for r in range(n)]))
    return [pick_mbr(o) for o in generate_many(model, jobs, T35)]


def bertscore_f1(preds, refs):
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type="roberta-large", lang="en", device=device)
    P, R, F1 = scorer.score(preds, refs)
    return float(F1.mean())


with open(OVERLAP_PATH) as f:
    overlap = json.load(f)
P = overlap["prompts"]
FOUND = overlap["found"]
ft_model = load_arch2(FT_PATH)

if RUN_HELDOUT:
    with open(HELDOUT_PATH) as f:
        held = json.load(f)
    base_model = load_arch2(ARCH2_PATH)
    h_pc = []
    h_refs = []
    for k, p in enumerate(P):
        if p["kind"] == "heldout" and FOUND[k]["cands"]:
            e = held[int(p["id"][1:])]
            h_pc.append((e["prompt"], FOUND[k]["cands"]))
            h_refs.append(e["reference"])
    plain = plain_mbr(base_model, [p for p, _ in h_pc], N_SAMPLES, 41000000)
    rag = rag_select(ft_model, h_pc, T35, N_SAMPLES, 42000000)
    print(f"held-out prompts with a passage: {len(h_pc)}")
    print(f"plain MBR x16 F1 {bertscore_f1(plain, h_refs):.4f} | retrieval + MBR x16 F1 {bertscore_f1(rag, h_refs):.4f}")

with open(os.path.join(DATA_DIR, TEST_FILE), encoding="utf-8") as f:
    test_data = json.load(f)
test_ids = [str(k) for k in test_data.keys()]
with open(BASE_PRED_PATH) as f:
    base_pred = json.load(f)
assert set(base_pred) == set(test_ids)

test_index = {}
for k, p in enumerate(P):
    if p["kind"] == "test":
        test_index[p["id"]] = k
t_ids = []
t_pc = []
for t in test_ids:
    if t in test_index and FOUND[test_index[t]]["cands"]:
        t_ids.append(t)
        t_pc.append((test_data[t]["text"], FOUND[test_index[t]]["cands"]))
print(f"{len(t_ids)}/{len(test_ids)} test prompts get retrieval-augmented outputs")

out = rag_select(ft_model, t_pc, T35, N_SAMPLES, TEST_SEED)
preds = {}
for t in test_ids:
    preds[t] = dict(base_pred[t])
for t, text in zip(t_ids, out):
    preds[t] = {"text": text}
write_submission(preds, OUTPUT_DIR)

n_exact = 0
for p in preds.values():
    if len(p["text"].split()) == 128:
        n_exact += 1
print(f"Wrote {len(preds)} predictions to {OUTPUT_DIR} | exactly 128 words: {n_exact}")
