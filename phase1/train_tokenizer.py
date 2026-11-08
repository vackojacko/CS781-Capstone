# Train the 16,384-token BPE tokenizer on the first 15M words of each legal source
import os
import time
import random
import ijson
from tokenizer import BPETokenizer

VOCAB_SIZE = 16384
CHECKPOINT_EVERY = 1000
SEED = 42
random.seed(SEED)

TARGET_WORDS_PER_SOURCE = 15000000

DATA_DIR = "data"
TRAIN_FILES = ["train_supreme_court.json", "train_high_courts.json", "train_acts.json", "train_tribunals.json"]

OUTPUT_DIR = "outputs"
CKPT_DIR = os.path.join(OUTPUT_DIR, "bpe_checkpoints")
TOKENIZER_PATH = "legal_bpe_" + str(VOCAB_SIZE) + ".json"

all_texts = []
for train_file in TRAIN_FILES:
    source_name = train_file.replace("train_", "").replace(".json", "")
    source_texts = []
    word_count = 0
    with open(os.path.join(DATA_DIR, train_file), "rb") as f:
        for record in ijson.items(f, "item"):
            text = record["text"]
            source_texts.append(text)
            word_count += len(text.split())
            if word_count >= TARGET_WORDS_PER_SOURCE:
                break
    all_texts.extend(source_texts)
    print(f"[{source_name}] sampled {len(source_texts)} records ({word_count} words)")

random.shuffle(all_texts)
corpus = " ".join(all_texts)
print(f"Combined corpus: {len(corpus.split())} words")

tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
t0 = time.time()
tokenizer.fit(corpus, checkpoint_dir=CKPT_DIR, checkpoint_every=CHECKPOINT_EVERY)
print(f"Total training time: {(time.time() - t0) / 60:.1f} min")
tokenizer.save(TOKENIZER_PATH)

tok = BPETokenizer.load(TOKENIZER_PATH)
print("Loaded vocab size:", tok.vocab_size)

test_texts = [
    "The Supreme Court held that the petition filed by the appellant was dismissed.",
    "Section 302 of the Indian Penal Code prescribes punishment for murder.",
    "The High Court of Judicature at Allahabad ruled in favor of the respondent.",
    "The tribunal directed the employer to reinstate the workman with full back wages.",
    "Under Article 226 of the Constitution of India the writ petition is maintainable.",
]
for text in test_texts:
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    if decoded.strip() == text:
        status = "PASS"
    else:
        status = "FAIL"
    print(f"[{status}] {len(encoded)} tokens, {len(encoded) / len(text.split()):.2f} tokens/word")

sample_text = " ".join(all_texts[:50])
sample_words = sample_text.split()
bpe_tokens = tok.encode(sample_text)
byte_count = len(sample_text.encode("utf-8"))
print(f"Tokens/word: {len(bpe_tokens) / len(sample_words):.2f}")
print(f"Compression: {byte_count / len(bpe_tokens):.2f}x vs raw bytes")
