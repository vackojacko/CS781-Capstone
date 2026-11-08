# Held-out prompts from documents Arch 2 never read: replays Arch 2's data pipeline to find where it stopped
# in each source, then samples 512 to 128 word windows from 5% of the later documents
import os
import json
import random
import hashlib
import ijson
from tokenizer import BPETokenizer

DATA_DIR = "data"
TRAIN_FILES = ["train_supreme_court.json", "train_high_courts.json", "train_acts.json", "train_tribunals.json"]
TOKENIZER_PATH = "legal_bpe_16384.json"
OUTPUT_PATH = os.path.join("outputs", "heldout_eval_set.json")

SEED = 42
EVAL_N = 400              # 50 per source for tuning + the next 50 per source for an audit
A2_TRAIN_TOKENS = 300000000
A2_VAL_TOKENS = 5000000
HELDOUT_PCT = 5
TARGET_WORDS = 128

tokenizer = BPETokenizer.load(TOKENIZER_PATH)
a2_rng = random.Random(SEED)
val_len = 0
eval_set = []
earlier_keys = set()
a2_train_ps = A2_TRAIN_TOKENS // len(TRAIN_FILES)
a2_val_ps = A2_VAL_TOKENS // len(TRAIN_FILES)
eval_ps = EVAL_N // len(TRAIN_FILES)
min_words = TARGET_WORDS + 512

for si, train_file in enumerate(TRAIN_FILES):
    src = train_file.replace("train_", "").replace(".json", "")
    b_rng = random.Random(SEED * 1000 + si)
    seen = set()
    a2_train_len = 0
    phase = "A"   # A: documents Arch 2 read, B: documents after its stop point
    val_limit = val_len + a2_val_ps
    eval_start = len(eval_set)
    with open(os.path.join(DATA_DIR, train_file), 'rb') as f:
        for record in ijson.items(f, 'item'):
            text = record["text"]
            h = hashlib.md5(text[:500].encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            if len(text.split()) < 50:
                continue
            if phase == "A":
                toks = tokenizer.encode(text)
                earlier_keys.add(h)
                if a2_rng.random() < 0.05:
                    if val_len < val_limit:
                        val_len += min(len(toks), val_limit - val_len)
                else:
                    if a2_train_len < a2_train_ps:
                        a2_train_len += min(len(toks), a2_train_ps - a2_train_len)
                if a2_train_len >= a2_train_ps and val_len >= val_limit:
                    phase = "B"
                    tokenizer._encode_cache = {}
                continue
            if h in earlier_keys:
                continue
            if b_rng.random() < HELDOUT_PCT / 100:
                words = text.split()
                if len(eval_set) - eval_start < eval_ps and len(words) >= min_words:
                    st = b_rng.randint(0, len(words) - min_words)
                    eval_set.append({"source": src, "doc_key": h, "prompt": " ".join(words[st:st + 512]),
                                     "reference": " ".join(words[st + 512:st + min_words])})
            if len(eval_set) - eval_start >= eval_ps:
                break
    print(f"{src}: held-out prompts {len(eval_set) - eval_start}")
    tokenizer._encode_cache = {}

with open(OUTPUT_PATH, "w") as f:
    json.dump(eval_set, f, indent=1)
print("Saved", len(eval_set), "held-out prompts to", OUTPUT_PATH)
