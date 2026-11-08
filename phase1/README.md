# Phase 1: Legal Language Model Pretraining

## Task

The model reads a 512-word prompt from an Indian legal document and writes the next 128 words.
Codabench scores it with BERTScore F1 (roberta-large).
I trained four different models from scratch (50M to 300M parameters each) with my own Assignment 1 BPE. They are the starting point of Phases 2 to 7.

## Data and preprocessing

- Four training files (`data/train_*.json`: Supreme Court, High Courts, Acts, Tribunals) and the test prompts (`data/test.json`), read as a stream with `ijson`.
- Exact duplicates are removed by MD5 hash: of the first 500 characters (Arch 1 to 3) or of the whole normalised text (Arch 4). Documents under 50 words are dropped.
- 5% of documents go to validation: at random with seed 42 (Arch 1, 2) or by lowest hash (Arch 3, 4).
- Each source gets the same token budget. The text is not lowercased or cleaned.
- Arch 1 and 2 cut one long token stream into 1025-token blocks. Arch 3 and 4 cut blocks inside one document, from a random offset.
- Tokenizer: byte-level BPE with an end-of-word marker, 16,384 tokens (4 special, 256 bytes, 1 marker, 16,123 merges), trained on the first 15M words of each source.
- `heldout_set.py` makes 400 held-out prompts (512 words plus the next 128) from documents Arch 2 never read. 200 are for tuning and 200 are an audit set.

## Models

All modules are written by hand in PyTorch from my Assignment 3 code.

| | Arch 1 | Arch 2 | Arch 3 | Arch 4 |
|---|---|---|---|---|
| Type | GPT-2 style decoder | LLaMA style decoder | Encoder-decoder | Latent-attention decoder with n-gram memory |
| Layers x width | 12 x 768 | 12 x 640 | (10 enc + 4 dec) x 640 | 16 x 640 |
| Attention | MHA, 12 heads | GQA, 10 query / 2 KV heads | MHA, cross-attention in the decoder | MLA, latent 320 |
| Positions | sinusoidal | RoPE | RoPE (self-attention only) | decoupled RoPE (32 dims) |
| Norm | LayerNorm, pre-norm | RMSNorm | RMSNorm | RMSNorm, QK-norm |
| FFN | GELU 3072 | SwiGLU 1706 | SwiGLU 1712 | SwiGLU 1712 |
| Context | 1024 | 1024 | 704 encoder / 320 decoder | 1024 |
| Vocabulary | 16,384 | 16,384 | 16,384 + 64 special tokens | 16,384 |
| Parameters | 97.6M (97,602,048) | 61.6M (61,604,480) | 86.1M (86,058,240) | 139.1M (139,089,664) |
| Tokens seen | 735M (planned 800M) | 300M (+329M continued) | 695M encoder + 275M decoder | 361M |
| Codabench (plain sampling) | 0.8299 | 0.8334 | 0.8275 | 0.8365 |

- Arch 1 is my A3 decoder at GPT-2 small size.
- Arch 2 has 2 KV heads for 10 query heads, so key and value projections are 80% smaller.
- Arch 3 has a bidirectional encoder and a causal decoder. 80% of updates train continuation and 20% train T5 span corruption.
- Arch 4 adds QK-norm for fp16 stability, a sigmoid output gate, and an Engram memory of hashed bigram and trigram tables at layers 2 and 8 (42.9M parameters).

## Pretraining

| Setting | Value |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.95), eps 1e-8, weight decay 0.1 on 2-D weights |
| Gradient clipping | 1.0 |
| Batch | 4 sequences x 8 accumulation steps = 32,768 tokens per update |
| Learning rate | peak 6e-4, minimum 6e-5, 500 warmup updates |
| Precision | fp16 with a gradient scaler, Kaggle T4, no dropout |

- Arch 1 and 2 (`pretrain.py`): cosine decay over the whole run.
- Arch 3 (`pretrain_arch3.py`): cosine schedule fixed to fit 18 hours (31,455 updates).
- Arch 4 (`pretrain_arch4.py`): warmup-stable-decay. The learning rate stays at its peak until validation loss stops improving, then decays. Engram tables get 5x the learning rate and no weight decay.

I also continued Arch 2 for 329M tokens (Codabench 0.8351). That run is not in this folder. Its outputs are only used in the consensus step below.

## Decoding

- Plain sampling for all four models: temperature 0.4, top-p 0.92, top-k 50, repetition penalty 1.15.
- `T35` in the code: temperature 0.5, top-p 0.95, top-k 20, repetition penalty 1.2. It was the best of 40 settings I tried for Arch 2.
- MBR decoding (`generate_mbr.py`): Arch 2 samples 16 continuations with `T35`. I keep the 128-word sample that agrees most with the others (mean ROUGE-1/2 F1).

## Retrieval fine-tuning

- `retrieval.py` searches the last 12 words of each prompt (and the 12 words ending 50 words earlier) word for word in the training files. For each match it keeps the next 128 words, up to 5 passages per prompt. 40.0% of test prompts get a passage.
- Without fine-tuning, adding the passage dropped held-out BERTScore from 0.8817 to 0.8471 (182 prompts).
- `retrieval_finetune.py` teaches Arch 2 to use the passage. It trains on 12,000 Supreme Court and High Court windows. In 25% of them the passage is wrong, so the model also learns to ignore it. Settings: AdamW, LR 1e-4 with cosine decay to 1e-5, 32 examples per update, 375 updates (about 16 minutes on a T4).
- `predict.py`: for prompts with a passage, the fine-tuned model samples 16 continuations and MBR picks one. Other prompts keep the plain MBR x16 output.

## Files

| File | What it does |
|---|---|
| `tokenizer.py` | my Assignment 1 BPE tokenizer: training, encoding, save and load |
| `train_tokenizer.py` | trains the tokenizer and writes `legal_bpe_16384.json` |
| `arch1_model.py` to `arch4_model.py` | the four models |
| `pretrain.py` | pretraining and test generation for Arch 1 or Arch 2, chosen by `ARCH` |
| `pretrain_arch3.py` | pretraining and test generation for Arch 3 |
| `pretrain_arch4.py` | pretraining, held-out BERTScore checks and test generation for Arch 4 |
| `heldout_set.py` | picks the 400 held-out prompts |
| `generate_mbr.py` | batched sampling for Arch 2, MBR selection and the MBR x16 submission |
| `retrieval.py` | anchor search, saves up to 5 passages per prompt |
| `retrieval_finetune.py` | builds the retrieval examples and fine-tunes Arch 2 |
| `predict.py` | retrieval fine-tune + MBR x16 submission. `RUN_HELDOUT = True` adds a held-out check |
| `consensus.py` | the best submission, a vote over finished runs |

## How to run

Needs Python 3, `torch`, `numpy`, `ijson`, `matplotlib`, `pyahocorasick`, and `bert_score` for the held-out checks. Training needs a CUDA GPU (I used a Kaggle T4).

Path constants are at the top of each script:

- `DATA_DIR = "data"`: `train_supreme_court.json`, `train_high_courts.json`, `train_acts.json`, `train_tribunals.json`, `test.json`.
- `TOKENIZER_PATH = "legal_bpe_16384.json"`.
- Outputs go to `outputs/<name>/`.

```
python train_tokenizer.py          # writes legal_bpe_16384.json
python pretrain.py                 # ARCH = "arch1", writes outputs/arch1/
python pretrain.py                 # ARCH = "arch2", writes outputs/arch2/
python pretrain_arch3.py           # writes outputs/arch3/
python pretrain_arch4.py           # writes outputs/arch4/
python heldout_set.py              # writes outputs/heldout_eval_set.json
python generate_mbr.py             # needs outputs/arch2/model.pt, writes outputs/mbr16/
python retrieval.py                # writes outputs/retrieval/overlap_matches.json
python retrieval_finetune.py       # needs outputs/arch2/model.pt, writes outputs/rag/rag_ft.pt
python predict.py                  # writes outputs/final/
```

- The pretraining scripts resume from their latest checkpoint when run again in a new Kaggle session.
- Run `heldout_set.py` before the retrieval scripts, and `generate_mbr.py` before `predict.py`.
- Every `submission.zip` holds one `prediction.json` that maps each test id to `{"text": <128 words>}`.

## Codabench submissions

| Submission | BERTScore F1 | How to make it |
|---|:--:|---|
| Arch 1, plain sampling | 0.8299 | `pretrain.py` with `ARCH = "arch1"` writes `outputs/arch1/submission.zip` |
| Arch 2, plain sampling | 0.8334 | `pretrain.py` with `ARCH = "arch2"` writes `outputs/arch2/submission.zip` |
| Arch 3, plain sampling | 0.8275 | `pretrain_arch3.py` writes `outputs/arch3/submission.zip` |
| Arch 4, plain sampling | 0.8365 | `pretrain_arch4.py` writes `outputs/arch4/submission.zip` |
| Arch 2 + MBR x16 | 0.8402 | `generate_mbr.py` (`T35`, `N_SAMPLES = 16`, `TEST_SEED = 9000000`) writes `outputs/mbr16/submission.zip` |
| Arch 2 + retrieval fine-tune + MBR x16 | 0.8681 | `retrieval.py`, `retrieval_finetune.py`, then `predict.py` (`T35`, `N_SAMPLES = 16`, `TEST_SEED = 52000000`) writes `outputs/final/submission.zip` |
| Consensus over finished runs (best) | 0.8688 | `consensus.py` on the run outputs listed at its top writes `submission_consensus.zip` |

### Consensus step (`consensus.py`)

For each prompt, it keeps the run output that agrees most with the other outputs (mean ROUGE-1/2 F1).

- 400 prompts with a passage: candidates are the retrieval fine-tunes of the original and continued Arch 2 and a second-seed run. Their greedy outputs also vote.
- 600 other prompts: candidates are 18 Arch 2 MBR runs and the MBR x16 output. The Arch 4 and continued Arch 2 outputs also vote.

Rebuilt from the saved runs, `consensus.py` matches the submitted file on 956 of 1,000 prompts. The other 44 differ because one voter file (the original model's greedy outputs) was not kept.

Not made by this folder: Arch 2 after continued pretraining (0.8351) and Arch 2 + MBR over 8 samples (0.8388).
