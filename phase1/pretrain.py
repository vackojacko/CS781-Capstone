# Pretraining + test generation for Arch 1 (GPT-2 style) and Arch 2 (LLaMA style)
import os
import random
import json
import time
import math
import zipfile
import hashlib
import array
import gc
import ijson
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tokenizer import BPETokenizer

ARCH = "arch1"  # "arch1" or "arch2"

DATA_DIR = "data"
TRAIN_FILES = ["train_supreme_court.json", "train_high_courts.json", "train_acts.json", "train_tribunals.json"]
TEST_FILE = "test.json"
TOKENIZER_PATH = "legal_bpe_16384.json"
OUTPUT_DIR = os.path.join("outputs", ARCH)
VOCAB_SIZE = 16384

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

if ARCH == "arch1":
    MODEL_DIM = 768
    MODEL_LAYERS = 12
    MODEL_HEADS = 12
    MODEL_KV_HEADS = 12
    MODEL_FF = 3072
    TRAIN_TOKENS = 800000000
    CKPT_EVERY = 500
else:
    MODEL_DIM = 640
    MODEL_LAYERS = 12
    MODEL_HEADS = 10
    MODEL_KV_HEADS = 2       # GQA: 5 query heads per KV head
    MODEL_FF = 1706          # SwiGLU: floor(2/3 * 4 * 640)
    TRAIN_TOKENS = 300000000
    CKPT_EVERY = 1000
MODEL_SEQ_LEN = 1024
MODEL_DROPOUT = 0.0

VAL_TOKENS = 5000000
BATCH_SIZE = 4
GRAD_ACCUM = 8           # effective batch = 32 sequences = 32,768 tokens
PEAK_LR = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

TARGET_WORDS = 128
MAX_GEN_TOKENS = 384
TEMPERATURE = 0.4
TOP_P = 0.92
TOP_K = 50
REPETITION_PENALTY = 1.15

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = BPETokenizer.load(TOKENIZER_PATH)
print("Loaded tokenizer: vocab_size =", tokenizer.vocab_size)

# stream each source, dedup by MD5 of the first 500 chars, drop docs under 50 words,
# equal token budget per source, 5% of documents go to validation
train_tokens = array.array('H')
val_tokens = array.array('H')

random.seed(SEED)
tokens_per_source = TRAIN_TOKENS // len(TRAIN_FILES)
val_per_source = VAL_TOKENS // len(TRAIN_FILES)

for train_file in TRAIN_FILES:
    source_name = train_file.replace("train_", "").replace(".json", "")
    print("Source:", source_name)

    source_train_limit = len(train_tokens) + tokens_per_source
    source_val_limit = len(val_tokens) + val_per_source
    seen_hashes = set()
    doc_count = 0
    kept_count = 0

    with open(os.path.join(DATA_DIR, train_file), 'rb') as f:
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

            if random.random() < 0.05:
                if len(val_tokens) < source_val_limit:
                    val_tokens.extend(toks[:source_val_limit - len(val_tokens)])
            else:
                if len(train_tokens) < source_train_limit:
                    train_tokens.extend(toks[:source_train_limit - len(train_tokens)])

            if len(train_tokens) >= source_train_limit and len(val_tokens) >= source_val_limit:
                break

            if doc_count % 10000 == 0:
                print(f"  {doc_count} docs | {kept_count} kept | train={len(train_tokens)} val={len(val_tokens)}")

    del seen_hashes
    gc.collect()
    print(f"  Done: {doc_count} scanned, {kept_count} kept | train {len(train_tokens)} | val {len(val_tokens)}")

print("Final train tokens:", len(train_tokens))
print("Final val tokens:", len(val_tokens))

# free the tokenizer encode cache after tokenizing the corpus
tokenizer._encode_cache = {}
gc.collect()


class TokenBlockDataset(Dataset):
    def __init__(self, tokens, block_size):
        total = (len(tokens) // block_size) * block_size
        arr_np = np.frombuffer(tokens, dtype=np.uint16)[:total]
        self.data = torch.from_numpy(arr_np.astype(np.int16)).view(-1, block_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].long()
        return x[:-1], x[1:]


train_dataset = TokenBlockDataset(train_tokens, MODEL_SEQ_LEN + 1)
val_dataset = TokenBlockDataset(val_tokens, MODEL_SEQ_LEN + 1)
del train_tokens, val_tokens
gc.collect()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print("Training sequences:", len(train_dataset))
print("Validation sequences:", len(val_dataset))

if ARCH == "arch1":
    from arch1_model import LMConfig, DecoderLM
    cfg = LMConfig(vocab_size=VOCAB_SIZE, d_model=MODEL_DIM, n_layers=MODEL_LAYERS, n_heads=MODEL_HEADS,
                   n_kv_heads=MODEL_KV_HEADS, d_ff=MODEL_FF, max_seq_len=MODEL_SEQ_LEN, dropout=MODEL_DROPOUT,
                   pos_emb_type="sinusoidal", norm_first=True)
    norm_class = nn.LayerNorm
else:
    from arch2_model import LMConfig, DecoderLM, RMSNorm
    cfg = LMConfig(vocab_size=VOCAB_SIZE, d_model=MODEL_DIM, n_layers=MODEL_LAYERS, n_heads=MODEL_HEADS,
                   n_kv_heads=MODEL_KV_HEADS, d_ff=MODEL_FF, max_seq_len=MODEL_SEQ_LEN, dropout=MODEL_DROPOUT)
    norm_class = RMSNorm

model = DecoderLM(cfg).to(device)


# GPT-2 style init
def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, norm_class):
        torch.nn.init.ones_(module.weight)
        if ARCH == "arch1":
            torch.nn.init.zeros_(module.bias)


model.apply(init_weights)

# scale residual projections by 1/sqrt(2*n_layers)
residual_std = 0.02 / math.sqrt(2 * MODEL_LAYERS)
for block in model.blocks:
    torch.nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
    if ARCH == "arch1":
        torch.nn.init.normal_(block.ffn.net[2].weight, mean=0.0, std=residual_std)
    else:
        torch.nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=residual_std)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params} ({n_params / 1e6:.1f}M)")

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

total_steps = len(train_loader) // GRAD_ACCUM
print("Total training steps:", total_steps)


def get_lr(step):
    if step < WARMUP_STEPS:
        return PEAK_LR * step / WARMUP_STEPS
    decay_ratio = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (PEAK_LR - MIN_LR)


def evaluate():
    model.eval()
    val_loss_sum = 0.0
    val_count = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            with autocast():
                vlogits = model(vx)
                vB, vL, vV = vlogits.shape
                vloss = F.cross_entropy(vlogits.view(vB * vL, vV), vy.view(vB * vL))
            val_loss_sum += vloss.item()
            val_count += 1
    model.train()
    return val_loss_sum / val_count


CKPT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")
scaler = GradScaler()
loss_history = []
val_losses = []

start_step = 0
if os.path.exists(CKPT_PATH):
    print("Resuming from checkpoint:", CKPT_PATH)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    start_step = ckpt["step"]
    loss_history = ckpt["loss_history"]
    val_losses = ckpt["val_losses"]

model.train()
step = 0
accum_loss = 0.0
t0 = time.time()

for context, target in train_loader:
    if step < start_step * GRAD_ACCUM:
        step += 1
        continue

    context, target = context.to(device), target.to(device)

    lr = get_lr(step // GRAD_ACCUM)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    with autocast():
        logits = model(context)
        B, L, V = logits.shape
        loss = F.cross_entropy(logits.view(B * L, V), target.view(B * L))
        loss = loss / GRAD_ACCUM

    scaler.scale(loss).backward()
    accum_loss += loss.item()

    if (step + 1) % GRAD_ACCUM == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        actual_step = step // GRAD_ACCUM

        if actual_step % 100 == 0:
            elapsed = time.time() - t0
            tokens_seen = (actual_step + 1) * BATCH_SIZE * GRAD_ACCUM * MODEL_SEQ_LEN
            print(f"Step {actual_step} | loss={accum_loss:.4f} | lr={lr:.2e} | {elapsed:.0f}s | {tokens_seen / 1e6:.1f}M tokens")
            loss_history.append({"step": actual_step, "loss": accum_loss, "tokens": tokens_seen})

        accum_loss = 0.0

        if actual_step % CKPT_EVERY == 0 and actual_step > 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": actual_step,
                "loss_history": loss_history,
                "val_losses": val_losses,
            }, CKPT_PATH)

            avg_val = evaluate()
            val_ppl = math.exp(min(avg_val, 20))
            val_losses.append({"step": actual_step, "val_loss": avg_val, "val_ppl": val_ppl})
            print(f"  Val loss={avg_val:.4f} | Val PPL={val_ppl:.1f}")

    step += 1

final_val_loss = evaluate()
model.eval()
final_ppl = math.exp(min(final_val_loss, 20))
print(f"Final val loss: {final_val_loss:.4f} | Final PPL: {final_ppl:.1f}")
print(f"Total training time: {(time.time() - t0) / 60:.1f} min")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot([h["step"] for h in loss_history], [h["loss"] for h in loss_history])
ax1.set_xlabel("Step")
ax1.set_ylabel("Training Loss")
ax1.grid(True, alpha=0.3)
if val_losses:
    ax2.plot([v["step"] for v in val_losses], [v["val_ppl"] for v in val_losses], marker='o', color='orange')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Validation Perplexity")
    ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"), dpi=150)


def generate(model, tokenizer, prompt, max_tokens=384, target_words=128,
             temperature=0.4, top_p=0.92, top_k=50, repetition_penalty=1.15):
    model.eval()
    tokens = tokenizer.encode(prompt)
    generated = []

    with torch.no_grad():
        for _ in range(max_tokens):
            context = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            context = context[:, -cfg.max_seq_len:]

            with autocast():
                logits = model(context)
            logits = logits[0, -1].float()

            # repetition penalty over the last 100 generated tokens
            for token_id in set(generated[-100:]):
                logits[token_id] /= repetition_penalty

            logits = logits / temperature

            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[-1]] = float('-inf')

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()

            tokens.append(token_id)
            generated.append(token_id)

            if len(tokenizer.decode(generated).split()) >= target_words:
                break

    continuation = tokenizer.decode(generated)
    words = continuation.split()[:target_words]
    return " ".join(words)


with open(os.path.join(DATA_DIR, TEST_FILE), "r", encoding="utf-8") as f:
    test_data = json.load(f)
print("Number of test examples:", len(test_data))

predictions = {}
items = list(test_data.items())
t0 = time.time()
for i, (example_id, record) in enumerate(items):
    prediction = generate(model, tokenizer, record["text"], max_tokens=MAX_GEN_TOKENS, target_words=TARGET_WORDS,
                          temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, repetition_penalty=REPETITION_PENALTY)
    predictions[str(example_id)] = {"text": prediction}
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(items)} | {time.time() - t0:.0f}s")

prediction_path = os.path.join(OUTPUT_DIR, "prediction.json")
with open(prediction_path, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
with zipfile.ZipFile(os.path.join(OUTPUT_DIR, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(prediction_path, arcname="prediction.json")

torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
metrics = {
    "arch": ARCH,
    "params": n_params,
    "train_tokens": len(train_dataset) * (MODEL_SEQ_LEN + 1),
    "final_val_loss": final_val_loss,
    "final_ppl": final_ppl,
    "loss_history": loss_history,
    "val_losses": val_losses,
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Done: {n_params / 1e6:.1f}M params | val PPL {final_ppl:.1f}")
