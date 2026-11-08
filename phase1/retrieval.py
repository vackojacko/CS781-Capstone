# Retrieval: search 12-word anchors of every prompt verbatim in the training files (Aho-Corasick)
# and keep up to 5 of the 128-word passages that follow a match in another document
import os
import json
import time
import hashlib
from collections import defaultdict
import ahocorasick
import ijson

DATA_DIR = "data"
TRAIN_FILES = ["train_supreme_court.json", "train_high_courts.json", "train_acts.json", "train_tribunals.json"]
TEST_FILE = "test.json"
HELDOUT_PATH = os.path.join("outputs", "heldout_eval_set.json")
OUTPUT_DIR = os.path.join("outputs", "retrieval")

ANCHOR_WORDS = 12
CONT_WORDS = 128
MAX_CANDS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
T0 = time.time()

with open(os.path.join(DATA_DIR, TEST_FILE), encoding="utf-8") as f:
    test_data = json.load(f)
with open(HELDOUT_PATH) as f:
    heldout = json.load(f)

P = []
for k, v in test_data.items():
    P.append({"kind": "test", "id": str(k), "prompt": v["text"], "doc_key": None})
for i, e in enumerate(heldout):
    P.append({"kind": "heldout", "id": "h" + str(i), "prompt": e["prompt"], "doc_key": e["doc_key"]})

# two anchors per prompt: its last 12 words ("end") and the 12 words ending 50 words earlier ("mid")
A = ahocorasick.Automaton()
anchors = defaultdict(list)
for pi, p in enumerate(P):
    w = p["prompt"].split()
    if len(w) >= ANCHOR_WORDS:
        anchors[" " + " ".join(w[-ANCHOR_WORDS:]) + " "].append((pi, "end", 0))
    if len(w) >= ANCHOR_WORDS + 50:
        anchors[" " + " ".join(w[-ANCHOR_WORDS - 50:-50]) + " "].append((pi, "mid", 50))
for key, owners in anchors.items():
    A.add_word(key, owners)
A.make_automaton()
print("anchors:", len(anchors))

found = []
for p in P:
    found.append({"end": 0, "mid": 0, "self_found": False, "cands": []})

for train_file in TRAIN_FILES:
    src = train_file.replace("train_", "").replace(".json", "")
    t0 = time.time()
    with open(os.path.join(DATA_DIR, train_file), "rb") as fh:
        for di, rec in enumerate(ijson.items(fh, "item")):
            text = rec["text"]
            norm = " " + " ".join(text.split()) + " "
            hits = list(A.iter(norm))
            if not hits:
                continue
            key = hashlib.md5(text[:500].encode()).hexdigest()  # same document key as the Arch 2 pipeline
            for end_idx, owners in hits:
                for pi, kind, skip in owners:
                    # a held-out prompt's own document does not count
                    if P[pi]["doc_key"] == key:
                        found[pi]["self_found"] = True
                        continue
                    found[pi][kind] += 1
                    if len(found[pi]["cands"]) < MAX_CANDS:
                        after = norm[end_idx + 1:end_idx + 1 + 12 * (skip + CONT_WORDS)].split()
                        cont = after[skip:skip + CONT_WORDS]
                        if len(cont) == CONT_WORDS:
                            found[pi]["cands"].append({"source": src, "doc": di, "anchor": kind, "text": " ".join(cont)})
    print(f"done {src} in {(time.time() - t0) / 60:.1f} min")

for kind in ["test", "heldout"]:
    n = 0
    n_found = 0
    for p, f_ in zip(P, found):
        if p["kind"] == kind:
            n += 1
            if f_["cands"]:
                n_found += 1
    print(f"{kind}: {n_found}/{n} prompts with a retrieved {CONT_WORDS}-word passage")

prompts_out = []
for p in P:
    prompts_out.append({"kind": p["kind"], "id": p["id"], "doc_key": p["doc_key"]})
with open(os.path.join(OUTPUT_DIR, "overlap_matches.json"), "w") as f:
    json.dump({"prompts": prompts_out, "found": found}, f)
print(f"Saved overlap_matches.json | total {(time.time() - T0) / 60:.1f} min")
