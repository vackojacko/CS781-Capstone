# Phase 1: Legal Language Model Pretraining

## Task and metric

Given a 512-word prompt taken from an Indian legal document, the model writes the next 128 words.
Predictions are scored on Codabench with BERTScore F1 (roberta-large) against the true continuation.

Rules that shape the code:

- Every model is trained from scratch on the provided corpus only.
- Every model has 50M to 300M parameters.
- At least four substantially different architectures.
- The tokenizer is my own Assignment 1 BPE, retrained on the legal corpus.

The four models trained here are the starting points of Phases 2 to 7.
The later phases keep their own copies of the model code; this folder holds the original Phase 1 model code and training scripts.

## Data and preprocessing

- Four official training files: Supreme Court, High Courts, Acts and Tribunals (`data/train_*.json`), plus the test prompts (`data/test.json`).
- The files are streamed with `ijson`, so they never have to fit in memory.
- Exact duplicates are removed. Arch 1, Arch 2 and Arch 3 use the MD5 hash of the first 500 characters. Arch 4 uses the MD5 hash of the whole whitespace-normalised text.
- Documents under 50 words are dropped.
- 5% of documents are held out for validation, split by document. Arch 1 and Arch 2 draw this 5% at random (seed 42). Arch 3 and Arch 4 put a document in validation when its hash falls in the lowest 5%.
- Each source gets an equal token budget, so the large Supreme Court file does not dominate.
- Text is not lowercased or cleaned. Citations, numbers and headings are kept.
- Tokenizer: byte-level BPE with an end-of-word marker, 16,384 tokens (4 special tokens, 256 bytes, the marker and 16,123 merges). It is trained on the first 15M words of each source. On legal text it needs a median 1.09 tokens per word (95th percentile 1.34), so a 512-word prompt is about 556 tokens and a 128-word continuation about 139 tokens.
- Arch 1 and Arch 2 cut one long token stream into 1025-token blocks. Arch 3 and Arch 4 cut blocks inside single documents, from a random offset, so a block never spans two documents.
- Held-out prompts for decoding studies (`heldout_set.py`): 400 prompts, 512 words of prompt and the next 128 words as reference, from documents after the point where Arch 2 stopped reading each source. Neither Arch 2 nor its retrieval fine-tune trains on them. The first 50 per source (200) are for tuning and the next 50 per source (200) are an audit set.

## Models

All modules are written by hand in PyTorch from my Assignment 3 attention code. No model classes are imported. Parameter counts sum all tensors, with the tied embedding counted once.

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
| Tokens consumed | 735M (planned 800M; the run made one full pass over the data, which came to 735M because a source with less text than its 200M share is not topped up from the others) | 300M (+329M continued) | 695M encoder + 275M decoder | 361M |
| Codabench (plain sampling) | 0.8299 | 0.8334 | 0.8275 | 0.8365 |

What is specific about each one:

- **Arch 1** (`arch1_model.py`) is my Assignment 3 decoder scaled to GPT-2 small size. It keeps fixed sinusoidal positions instead of a learned table, since the context never exceeds 1024. Pre-norm LayerNorm, GELU MLP, tied embeddings.
- **Arch 2** (`arch2_model.py`) is a compact LLaMA style decoder: RoPE, RMSNorm, SwiGLU and grouped-query attention. With 2 KV heads for 10 query heads, each key or value projection has 80% fewer parameters. Width and depth were chosen so 300M tokens fit one 12-hour Kaggle session.
- **Arch 3** (`arch3_model.py`) is an encoder-decoder. The encoder is bidirectional and has 10 layers; the decoder is causal, has 4 layers and cross-attends to the encoder. The 704 / 320 split comes from the 95th percentile lengths of prompts (688 tokens) and continuations (172 tokens). The vocabulary gains 64 tokens: a `<cont>` and a `<denoise>` mode token and 62 span sentinels. Training mixes 80% continuation updates with 20% T5 span corruption updates (15% of tokens, mean span 3), and the last 10% of updates use task-shaped 512-word to 128-word examples.
- **Arch 4** (`arch4_model.py`) is a 16-layer decoder with multi-head latent attention: keys and values come from a 320-dimensional latent (d/2), with a separate 32-dimensional RoPE key shared by all heads. Each head is QK-normalised for fp16 stability and has a sigmoid output gate. Layers 2 and 8 (counted from 0) add an Engram memory: hashed bigram and trigram tables (4 hash heads per order, 32 dimensions, about 82k rows each) read by a context gate. The memory holds 42.9M of the 139.1M parameters and starts as an exact no-op (zero-initialised value path and convolution). Tokens with the same normalised text share one n-gram id; invalid UTF-8 byte tokens keep their own id, which fixes a collision that had mapped 134 byte tokens to one key.

## Methodology

### Pretraining

- AdamW, betas (0.9, 0.95), eps 1e-8, weight decay 0.1 on 2-D weights only, gradient clipping 1.0.
- 4 sequences x 8 accumulation steps = 32 sequences = 32,768 tokens per update.
- Peak learning rate 6e-4, minimum 6e-5, 500 warmup updates.
- fp16 autocast with a gradient scaler on a Kaggle T4. No dropout.
- Init N(0, 0.02); residual output projections use std 0.02 / sqrt(2 x layers).
- Arch 1 and Arch 2: cosine decay over the whole run (`pretrain.py`).
- Arch 3: the cosine horizon is measured during warmup and fixed so training fits 18 hours (31,455 updates). Fp16 overflow updates are skipped and do not advance the schedule (`pretrain_arch3.py`).
- Arch 4: warmup-stable-decay. The learning rate stays at its peak until the validation loss plateaus or the time cap is near, then decays over 20% of the stable length. The last 10% of updates use task-shaped examples with loss on the continuation only. Engram tables have their own AdamW group with 5x the learning rate and no weight decay (`pretrain_arch4.py`).
- Arch 3 and Arch 4 save full checkpoints (model, optimiser, scaler, samplers, random states) and resume exactly across Kaggle sessions.

Arch 2 was also continued for 329M tokens on unread documents (Codabench 0.8351). The continued pretraining run is not part of this folder. Its outputs are used only in the final consensus step (see below): its plain sampling output as a voter, and a retrieval fine-tune of it (same `retrieval_finetune.py` and `predict.py`, with the continued checkpoint) as a candidate.

### Decoding

- The plain submissions of all four models use one sampling setting: temperature 0.4, top-p 0.92, top-k 50, repetition penalty 1.15 over the last 100 generated tokens. Prompts keep their most recent tokens: 1024 for Arch 1 and Arch 2, 703 after the `<cont>` token for Arch 3, 704 for Arch 4. Generation stops at 128 complete words.
- A 40-setting sweep of single-sample decoding for Arch 2 found temperature 0.5, top-p 0.95, top-k 20, repetition penalty 1.2 (called `T35` in the code). It was only 0.0003 above the original setting on held-out prompts, below the seed spread of 0.0019.
- Minimum Bayes risk (MBR) decoding (`generate_mbr.py`): Arch 2 samples 16 continuations per prompt with `T35`. Among the samples with exactly 128 words, the one with the highest mean ROUGE-1/2 F1 against the others is kept. Each sample has its own seeded generator, so its text depends only on the prompt, the setting and the seed, even inside a batch.

### Retrieval and retrieval fine-tuning

- `retrieval.py` builds two 12-word anchors per prompt: its last 12 words, and the 12 words ending 50 words earlier. It searches them verbatim in the four training files (Aho-Corasick over whitespace-normalised documents). For each match in another document, the 128 words that continue the prompt there are kept, up to 5 passages per prompt. 40.0% of test prompts and 45.5% of held-out prompts get at least one passage.
- Arch 2 as trained does not copy from a passage placed in its context: on 182 matched held-out prompts, retrieval without fine-tuning dropped BERTScore from 0.8817 to 0.8471.
- `retrieval_finetune.py` fine-tunes the original Arch 2 to use passages. Input layout: [prompt without its last 12 words][last 12 words + passage][last 12 words], cut to the last 704 tokens. 12,000 windows of 512 + 128 words come from Supreme Court and High Court training documents (held-out documents excluded). In 25% of them the passage comes from another document, so the model also learns to ignore a wrong passage. Loss is on the continuation only. AdamW, learning rate 1e-4 with 20 warmup updates and cosine decay to 1e-5, 32 examples per update, one pass (375 updates, about 16 minutes on a T4), no dropout.
- `predict.py` writes the final submission. For each test prompt with a passage, the fine-tuned model samples 16 continuations with `T35`, spread over its passages, and MBR picks one. The other prompts keep the plain MBR x16 output.

There is no distillation and no ensembling of different models in Phase 1. Distillation experiments start in Phase 2.

## Files

| File | What it does |
|---|---|
| `tokenizer.py` | Assignment 1 BPE tokenizer: word-frequency training with incremental pair counts, merge-rank encoding with a word cache, save and load. |
| `train_tokenizer.py` | Trains the 16,384-token tokenizer on the first 15M words of each source and writes `legal_bpe_16384.json`. |
| `arch1_model.py` | Arch 1 model (GPT-2 style decoder). |
| `arch2_model.py` | Arch 2 model (LLaMA style decoder). |
| `arch3_model.py` | Arch 3 model (encoder-decoder). |
| `arch4_model.py` | Arch 4 model (latent attention, QK-norm, output gate, Engram memory) with cached step-by-step generation. |
| `pretrain.py` | Data pipeline, pretraining and test generation for Arch 1 or Arch 2, chosen by `ARCH`. |
| `pretrain_arch3.py` | Data pipeline, mixed-objective pretraining and test generation for Arch 3. |
| `pretrain_arch4.py` | Data pipeline, warmup-stable-decay pretraining, held-out BERTScore checks and test generation for Arch 4. |
| `heldout_set.py` | Replays Arch 2's data pipeline and samples the 400 held-out prompts from documents Arch 2 never read. |
| `generate_mbr.py` | Batched KV-cached sampling for Arch 2, MBR selection and the MBR x16 test submission. Also imported by the scripts below. |
| `retrieval.py` | 12-word anchor search over the training files; saves up to 5 passages per prompt. |
| `retrieval_finetune.py` | Builds the 12,000 retrieval examples and fine-tunes Arch 2 on them. |
| `predict.py` | Final submission: retrieval fine-tuned Arch 2 + MBR x16 on prompts with a passage, plain MBR x16 elsewhere. Optional held-out check with `RUN_HELDOUT = True`. |

## How to run

Dependencies: Python 3, `torch`, `numpy`, `ijson`, `matplotlib`, `pyahocorasick` (imported as `ahocorasick`). `bert_score` is needed only for the held-out BERTScore checks (`pretrain_arch4.py` milestones, `RUN_HELDOUT` in `predict.py`). Training needs a CUDA GPU; all runs used a Kaggle T4 (16 GB, fp16).

Path constants are at the top of each script. By default everything runs from this folder:

- `DATA_DIR = "data"`: `train_supreme_court.json`, `train_high_courts.json`, `train_acts.json`, `train_tribunals.json`, `test.json`.
- `TOKENIZER_PATH = "legal_bpe_16384.json"`.
- Outputs go to `outputs/<name>/`.

Order of commands:

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

- `pretrain.py`: set `ARCH` at the top. Each run writes `model.pt`, `metrics.json`, `loss_curves.png`, `prediction.json` and `submission.zip`. Arch 1 took two Kaggle sessions; rerunning the script resumes from `checkpoint.pt`.
- `pretrain_arch3.py` and `pretrain_arch4.py` cache their tokenised data in their output folder and resume from their latest checkpoint when rerun in a new session. Both write `model.pt`, `prediction.json` and `submission.zip`; `pretrain_arch4.py` also writes `metrics.json`, `run_log.json` and `loss_curves.png`.
- `heldout_set.py` must run before `retrieval.py` and `retrieval_finetune.py`: the held-out documents are skipped by the fine-tune, and their own matches are ignored by the retrieval search.
- `predict.py` reads `outputs/mbr16/prediction.json` for the prompts without a passage, so `generate_mbr.py` must run first.
- Every `submission.zip` holds one `prediction.json` that maps each test id to `{"text": <128 words>}`.

## Codabench submissions covered by this code

| Submission | Codabench BERTScore F1 | How to produce it |
|---|:--:|---|
| Arch 1, plain sampling | 0.8299 | `pretrain.py` with `ARCH = "arch1"` writes `outputs/arch1/submission.zip` |
| Arch 2, plain sampling | 0.8334 | `pretrain.py` with `ARCH = "arch2"` writes `outputs/arch2/submission.zip` |
| Arch 3, plain sampling | 0.8275 | `pretrain_arch3.py` writes `outputs/arch3/submission.zip` |
| Arch 4, plain sampling | 0.8365 | `pretrain_arch4.py` writes `outputs/arch4/submission.zip` |
| Arch 2 + MBR x16 | 0.8402 | `generate_mbr.py` (`T35`, `N_SAMPLES = 16`, `TEST_SEED = 9000000`) writes `outputs/mbr16/submission.zip` |
| Arch 2 + retrieval fine-tune + MBR x16 | 0.8681 | `retrieval.py`, `retrieval_finetune.py`, then `predict.py` (`T35`, `N_SAMPLES = 16`, `TEST_SEED = 52000000`) writes `outputs/final/submission.zip` |
| **Consensus over finished runs (best, on the leaderboard)** | **0.8688** | `consensus.py` on the run outputs listed at its top writes `submission_consensus.zip` |

### Final consensus step (`consensus.py`)

The best submission (0.8688) applies the MBR rule to whole runs. For each prompt it keeps the output that has the highest mean ROUGE-1/ROUGE-2 F1 agreement with the other outputs:
- Prompts with a retrieved passage (400): candidates are the retrieval fine-tunes of the original and the continued Arch 2 (MBR x16) and a second-seed run; their greedy outputs are extra voters. Ties keep the continued model's output.
- Other prompts (600): candidates are 18 Arch 2 MBR runs (pool sizes 12 to 48, different seeds) and the MBR x16 output; the Arch 4 and continued Arch 2 sampling outputs are extra voters. Ties keep the MBR x40 run.

Every submitted word is generated by a Phase 1 model: retrieved text is only part of the input, and this step only chooses among finished outputs. Rebuilt from the saved run outputs, `consensus.py` matches the submitted file on all 600 ordinary prompts and on 356 of 400 retrieval prompts; the other 44 differ because the original model's greedy outputs (a voter file) were not kept.

Two other Phase 1 submissions are not reproduced by this folder: Arch 2 after continued pretraining (0.8351) and Arch 2 + MBR over 8 samples (0.8388).
