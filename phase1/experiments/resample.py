# Second sampling run of the trained Arch 1 with the same settings as pretrain.py
import os
import json
import torch
from torch.cuda.amp import autocast
from arch1_model import LMConfig, DecoderLM
from generate_mbr import tokenizer, device, write_submission

DATA_DIR = "data"
TEST_FILE = "test.json"
MODEL_PATH = os.path.join("outputs", "arch1", "model.pt")
OUTPUT_DIR = os.path.join("outputs", "arch1_resample")
VOCAB_SIZE = 16384
MODEL_SEQ_LEN = 1024
SEED = 42

TARGET_WORDS = 128
MAX_GEN_TOKENS = 384
TEMPERATURE = 0.4
TOP_P = 0.92
TOP_K = 50
REPETITION_PENALTY = 1.15


def generate(model, prompt):
    # same sampling as pretrain.py
    model.eval()
    tokens = tokenizer.encode(prompt)
    generated = []

    with torch.no_grad():
        for _ in range(MAX_GEN_TOKENS):
            context = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            context = context[:, -MODEL_SEQ_LEN:]

            with autocast():
                logits = model(context)
            logits = logits[0, -1].float()

            for token_id in set(generated[-100:]):
                logits[token_id] /= REPETITION_PENALTY

            logits = logits / TEMPERATURE

            top_k_vals, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
            logits[logits < top_k_vals[-1]] = float('-inf')

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > TOP_P
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()

            tokens.append(token_id)
            generated.append(token_id)

            if len(tokenizer.decode(generated).split()) >= TARGET_WORDS:
                break

    words = tokenizer.decode(generated).split()[:TARGET_WORDS]
    return " ".join(words)


def write_test_outputs(model, out_dir):
    torch.manual_seed(SEED)
    with open(os.path.join(DATA_DIR, TEST_FILE), encoding="utf-8") as f:
        test_data = json.load(f)
    preds = {}
    for i, (example_id, record) in enumerate(test_data.items()):
        preds[str(example_id)] = {"text": generate(model, record["text"])}
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(test_data)}")
    write_submission(preds, out_dir)
    print("Wrote", len(preds), "predictions to", out_dir)


if __name__ == "__main__":
    cfg = LMConfig(vocab_size=VOCAB_SIZE, d_model=768, n_layers=12, n_heads=12, n_kv_heads=12, d_ff=3072,
                   max_seq_len=MODEL_SEQ_LEN, dropout=0.0, pos_emb_type="sinusoidal", norm_first=True)
    model = DecoderLM(cfg)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
    model = model.to(device)
    write_test_outputs(model, OUTPUT_DIR)
