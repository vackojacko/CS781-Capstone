# builds the fixed token subset shared by every ablation run (Arch 2 data pipeline)
import os
import sys
import json
import time
import random
import hashlib
import array
import unicodedata

import numpy as np
import ijson

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from tokenizer import BPETokenizer

DATA_DIR = "data"
TOKENIZER_PATH = "legal_bpe_16384.json"
OUT = os.path.join(HERE, "abl_data")
TRAIN_FILES = ["train_supreme_court.json", "train_high_courts.json", "train_acts.json", "train_tribunals.json"]

SEED = 42
TRAIN_TOKENS = 64_000_000
VAL_TOKENS = 1_000_000
VOCAB_SIZE = 16384

os.makedirs(OUT, exist_ok=True)
tokenizer = BPETokenizer.load(TOKENIZER_PATH)
assert tokenizer.vocab_size == VOCAB_SIZE


# engram compression key: same text after unicode folding, lowercasing and whitespace cleanup
def make_key(b, ends_word):
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


# decode once so the tokenizer builds its token byte tables
tokenizer.decode([5])
keys = {}
compress = []
for t in range(VOCAB_SIZE):
    b = tokenizer._token_bytes.get(t)
    if b is None:
        key = ("special", t)
    else:
        key = make_key(b, bool(tokenizer._ends_word.get(t, False)))
    compress.append(keys.setdefault(key, len(keys)))
np.save(os.path.join(OUT, "engram_compress.npy"), np.array(compress, dtype=np.int64))
print("engram compressed vocab", len(keys), flush=True)

train_tokens = array.array('H')
val_tokens = array.array('H')

random.seed(SEED)
tokens_per_source = TRAIN_TOKENS // len(TRAIN_FILES)
val_per_source = VAL_TOKENS // len(TRAIN_FILES)
stats = {}
t0 = time.time()

for train_file in TRAIN_FILES:
    source_name = train_file.replace("train_", "").replace(".json", "")
    print("source", source_name, flush=True)
    file_path = os.path.join(DATA_DIR, train_file)

    source_train_limit = len(train_tokens) + tokens_per_source
    source_val_limit = len(val_tokens) + val_per_source
    seen_hashes = set()
    doc_count = 0
    kept_count = 0
    val_docs = 0
    train_docs = 0

    with open(file_path, 'rb') as f:
        for record in ijson.items(f, 'item'):
            doc_count += 1
            text = record["text"]

            h = hashlib.md5(text[:500].encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            if len(text.split()) < 50:
                continue

            kept_count += 1
            toks = tokenizer.encode(text)

            # 5% of documents go to validation
            if random.random() < 0.05:
                if len(val_tokens) < source_val_limit:
                    val_tokens.extend(toks[:source_val_limit - len(val_tokens)])
                    val_docs += 1
            else:
                if len(train_tokens) < source_train_limit:
                    train_tokens.extend(toks[:source_train_limit - len(train_tokens)])
                    train_docs += 1

            if len(train_tokens) >= source_train_limit and len(val_tokens) >= source_val_limit:
                break

            if doc_count % 5000 == 0:
                print("  ", doc_count, "docs", kept_count, "kept", len(train_tokens), len(val_tokens),
                      round(time.time() - t0), "s", flush=True)

    tokenizer._encode_cache = {}
    stats[source_name] = {"scanned": doc_count, "kept": kept_count, "train_docs": train_docs, "val_docs": val_docs,
                          "train_tokens_total": len(train_tokens), "val_tokens_total": len(val_tokens)}
    print(stats[source_name], flush=True)

train_np = np.frombuffer(train_tokens, dtype=np.uint16).copy()
val_np = np.frombuffer(val_tokens, dtype=np.uint16).copy()
np.save(os.path.join(OUT, "train.npy"), train_np)
np.save(os.path.join(OUT, "val.npy"), val_np)
meta = {"train_tokens": int(len(train_np)), "val_tokens": int(len(val_np)), "seed": SEED,
        "split": "per document, random.random() < 0.05 goes to validation (Arch 2 pipeline)",
        "dedup": "md5 of first 500 chars, docs < 50 words dropped", "sources": stats,
        "train_md5": hashlib.md5(train_np.tobytes()).hexdigest(), "val_md5": hashlib.md5(val_np.tobytes()).hexdigest(),
        "minutes": (time.time() - t0) / 60}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print(meta, flush=True)
