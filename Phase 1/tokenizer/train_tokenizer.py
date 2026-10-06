"""
Kaggle script: Train BPE tokenizer on the legal corpus.

Upload this file + bpe_tokenizer.py to Kaggle.
Run as a notebook cell or as: !python train_tokenizer.py

Trains on ALL 4 legal sources (Acts, High Courts, Supreme Court, Tribunals)
to ensure vocabulary covers the full domain. Samples proportionally.

Estimated time: ~15-25 min for 16K vocab on Kaggle CPU.
GPU not needed - this is pure Python string processing.

Checkpointing: saves every 1000 merges. If Kaggle times out, re-run to resume.
"""

import json
import os
import time
import random
from huggingface_hub import hf_hub_download

# ============================================================
# CONFIG
# ============================================================
VOCAB_SIZE = 16384
CHECKPOINT_EVERY = 1000
SEED = 42
random.seed(SEED)

# Sample size per source (words, not records).
# Total ~12-15M words across all 4 sources.
# More data = better vocabulary coverage of rare legal terms.
# Diminishing returns beyond ~150GB (Reddy et al. 2025),
# but 15M words is nowhere near that - we want as much as is practical.
TARGET_WORDS_PER_SOURCE = 4_000_000

# All 4 training sources
HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
TRAIN_FILES = [
    "Phase_1_Pretraining/Train/train_supreme_court.json",
    "Phase_1_Pretraining/Train/train_high_courts.json",
    "Phase_1_Pretraining/Train/train_acts.json",
    "Phase_1_Pretraining/Train/train_tribunals.json",
]

# Paths
OUTPUT_DIR = "/kaggle/working"
CKPT_DIR = os.path.join(OUTPUT_DIR, "bpe_checkpoints")
TOKENIZER_PATH = os.path.join(OUTPUT_DIR, f"legal_bpe_{VOCAB_SIZE}.json")

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ============================================================
# 1. Download and sample from ALL 4 sources
# ============================================================
print("=" * 60)
print("STEP 1: Downloading and sampling from all 4 legal sources")
print("=" * 60)

all_texts = []

for train_file in TRAIN_FILES:
    source_name = train_file.split("/")[-1].replace("train_", "").replace(".json", "")
    print(f"\n  [{source_name}] Downloading...")

    file_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=train_file,
        repo_type="dataset",
        token=HF_TOKEN,
    )

    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"  [{source_name}] {len(records):,} records")

    # shuffle and collect text up to target word count
    random.shuffle(records)
    source_texts = []
    word_count = 0
    for r in records:
        text = r["text"]
        words = len(text.split())
        source_texts.append(text)
        word_count += words
        if word_count >= TARGET_WORDS_PER_SOURCE:
            break

    all_texts.extend(source_texts)
    print(f"  [{source_name}] Sampled {len(source_texts):,} records ({word_count:,} words)")

# Shuffle all sources together so merges aren't biased by order
random.shuffle(all_texts)
corpus = " ".join(all_texts)
total_words = len(corpus.split())
print(f"\n  Combined corpus: {total_words:,} words, {len(corpus) / 1e6:.1f} MB")

# ============================================================
# 2. Train tokenizer (with checkpointing)
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 2: Training BPE tokenizer (vocab={VOCAB_SIZE})")
print("=" * 60)

from bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
t0 = time.time()
tokenizer.fit(corpus, checkpoint_dir=CKPT_DIR, checkpoint_every=CHECKPOINT_EVERY)
total_time = time.time() - t0

# ============================================================
# 3. Save final tokenizer
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Saving tokenizer")
print("=" * 60)

tokenizer.save(TOKENIZER_PATH)

# ============================================================
# 4. Sanity checks
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Sanity checks")
print("=" * 60)

tok = BPETokenizer.load(TOKENIZER_PATH)
print(f"  Loaded vocab size: {len(tok)}")

# Test roundtrip on diverse legal text
test_texts = [
    "The Supreme Court held that the petition filed by the appellant was dismissed.",
    "Section 302 of the Indian Penal Code prescribes punishment for murder.",
    "The High Court of Judicature at Allahabad ruled in favor of the respondent.",
    "The tribunal directed the employer to reinstate the workman with full back wages.",
    "Under Article 226 of the Constitution of India, the writ petition is maintainable.",
]

all_pass = True
for text in test_texts:
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    fertility = len(encoded) / len(text.split())
    match = decoded.strip() == text
    if not match:
        all_pass = False
    print(f"\n  Original:  {text}")
    print(f"  Tokens:    {len(encoded)} ({fertility:.2f} tok/word)")
    print(f"  Roundtrip: {'PASS' if match else 'FAIL'}")

# Corpus-level fertility (on a larger sample)
sample_text = " ".join(all_texts[:50])
sample_words = sample_text.split()
bpe_tokens = tok.encode(sample_text)
byte_tokens = len(sample_text.encode("utf-8"))
print(f"\n  Fertility on 50-record sample:")
print(f"    Words:      {len(sample_words):,}")
print(f"    BPE tokens: {len(bpe_tokens):,} ({len(bpe_tokens)/len(sample_words):.2f} tok/word)")
print(f"    Raw bytes:  {byte_tokens:,}")
print(f"    Compression vs bytes: {byte_tokens/len(bpe_tokens):.2f}x")

print(f"\n  Total training time: {total_time / 60:.1f} min")
print(f"  All roundtrip tests: {'PASSED' if all_pass else 'SOME FAILED'}")
print(f"\n  Tokenizer: {TOKENIZER_PATH}")
print("  Save this file as a Kaggle dataset to reuse across all experiments.")
print("\nDone!")
