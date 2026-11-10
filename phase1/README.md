# Phase 1: Legal Language Model Pretraining

## Task

The input is a 512 word prompt from an Indian legal document. The model has to write the next 128 words. Codabench scores the output with BERTScore F1 (roberta-large).

## My final system

My final system is my Arch 2 model with MBR decoding.

1. I trained my own BPE tokenizer from Assignment 1 (16,384 tokens) on the training files.
2. I trained four language models from scratch, each between 50M and 300M parameters. Arch 2, a LLaMA style decoder with 61.6M parameters, is the one I used for the final system.
3. For each test prompt, Arch 2 samples 16 continuations with the sampling setting called `T35` in the code (temperature 0.5, top p 0.95, top k 20, repetition penalty 1.2). This was the best of 40 settings I tried for Arch 2.
4. MBR decoding: for each continuation I compute its mean ROUGE-1/2 F1 against the other 15, and I keep the one that agrees most with the others. If some continuations have exactly 128 words, only those take part.
5. The kept continuation is cut to exactly 128 words.

Leaderboard score: 0.8402.

## Models I tried

| Phase 1 model | Technique | Codabench |
|---|---|---|
| Arch 1: GPT 2 style decoder, 97.6M | plain sampling | 0.8299 |
| Arch 2: LLaMA style decoder, 61.6M | plain sampling | 0.8334 |
| Arch 3: encoder decoder, 86.1M | plain sampling | 0.8275 |
| Arch 4: latent attention decoder with n gram memory, 139.1M | plain sampling | 0.8365 |
| Arch 2 | MBR over 16 samples (leaderboard) | 0.8402 |

All models are written by hand in PyTorch, starting from my Assignment 3 code. On my held out prompts, MBR scored higher than single samples, and MBR over 16 samples scored higher than over 4 or 8.

## Architectures

All four models are written by hand in PyTorch and share one layout: token embedding, positions, N blocks of (norm, attention, norm, feed-forward) with residual connections, a final norm and an LM head tied to the embedding. Context 1024, vocabulary 16,384.

| | Arch 1 | Arch 2 | Arch 3 | Arch 4 |
|---|---|---|---|---|
| Type | GPT-2 style decoder | LLaMA style decoder | encoder-decoder | deep decoder |
| Layers x width | 12 x 768 | 12 x 640 | 10 encoder + 4 decoder, 640 | 16 x 640 |
| Attention | 12 heads | grouped query, 10 query and 2 key/value heads | 10 heads, plus cross attention | latent attention (latent 320, separate 32-dim RoPE part), QK norm, sigmoid output gate |
| Positions | sinusoidal | RoPE | RoPE | RoPE |
| Norm | LayerNorm | RMSNorm | RMSNorm | RMSNorm |
| Feed-forward | GELU, 3072 | SwiGLU, 1706 | SwiGLU, 1712 | SwiGLU, 1712 |
| Extra | | | inputs up to 704 tokens, outputs up to 320 | hashed bigram and trigram memory at layers 2 and 8 (42.9M) |
| Parameters | 97.6M | 61.6M | 86.1M | 139.1M |

Arch 3 is trained 80% on continuation and 20% on T5 span corruption, with the last 10% of training on 512 word prompts and 128 word targets. The Arch 4 memory hashes the last 2 and 3 tokens into learned tables (4 hashes each), a gate compares the looked up vector with the hidden state, and the result is added to the residual stream. Its value projection starts at zero, so it has no effect at the start of training.

## Training setup

- Data: the four official sources, streamed, deduplicated, documents under 50 words dropped, 5% of documents held out, and the same token budget for each source.
- AdamW with betas (0.9, 0.95), weight decay 0.1 on matrices only, gradient clipping 1.0, no dropout.
- 32 sequences of 1024 tokens per update (batch 4, accumulation 8), fp16 on one Kaggle T4.
- Arch 1 and 2: peak learning rate 6e-4, 500 warmup updates, cosine decay to 6e-5. Arch 2 trained on 300M tokens (about 9,150 updates), Arch 1 on 735M.
- Arch 3: the same optimizer, 695M encoder and 275M decoder tokens.
- Arch 4: warmup, then a constant learning rate until the validation loss stops improving, then decay. The memory tables have their own group with 5x the learning rate and no weight decay. 361M tokens.
- Generation for the final system: `T35` sampling, 16 samples per prompt, MBR over the samples that reach 128 words.

## Files

| File | What it does |
|---|---|
| `tokenizer.py` | my Assignment 1 BPE tokenizer |
| `train_tokenizer.py` | trains the tokenizer and writes `legal_bpe_16384.json` |
| `legal_bpe_16384.json` | the trained tokenizer |
| `arch1_model.py` to `arch4_model.py` | the four models |
| `pretrain.py` | trains Arch 1 or Arch 2 and writes test outputs |
| `pretrain_arch3.py` | trains Arch 3 and writes test outputs |
| `pretrain_arch4.py` | trains Arch 4, checks it on held out prompts, writes test outputs |
| `generate_mbr.py` | samples 16 outputs per prompt with Arch 2 and keeps the MBR choice |
| `ablations/` | small runs that test parts of Arch 2 and Arch 4 (see its own README) |
| `experiments/` | code for my other submissions, listed under Other submissions below |

## How to run

Dependencies: Python 3, torch, numpy, ijson, matplotlib, and bert_score for the held out check in `pretrain_arch4.py`. I trained on a Kaggle T4 GPU.

Set these constants at the top of the scripts if your paths differ:

1. `DATA_DIR = "data"`, the folder with `train_supreme_court.json`, `train_high_courts.json`, `train_acts.json`, `train_tribunals.json` and `test.json`.
2. `TOKENIZER_PATH = "legal_bpe_16384.json"`.
3. `ARCH` in `pretrain.py`, set to `"arch1"` or `"arch2"`.
4. Outputs go to `outputs/<name>/`.

Run in this order:

```
python train_tokenizer.py
python pretrain.py                 # once with ARCH = "arch1", once with ARCH = "arch2"
python pretrain_arch3.py
python pretrain_arch4.py
python generate_mbr.py
```

The training scripts continue from their last checkpoint if they are run again. `generate_mbr.py` needs `outputs/arch2/model.pt` and writes `outputs/mbr16/submission.zip`. Its random seeds are fixed (`TEST_SEED = 9000000`), so each test prompt always gets the same 16 samples. Every zip holds one `prediction.json` that maps each test id to its 128 word continuation.

## Submissions this code reproduces

| Submission | Output folder | Codabench |
|---|---|---|
| Arch 1, plain sampling | `outputs/arch1/` | 0.8299 |
| Arch 2, plain sampling | `outputs/arch2/` | 0.8334 |
| Arch 3, plain sampling | `outputs/arch3/` | 0.8275 |
| Arch 4, plain sampling | `outputs/arch4/` | 0.8365 |
| Arch 2 + MBR over 16 samples (leaderboard) | `outputs/mbr16/` | 0.8402 |

## Other submissions

These are my other Phase 1 Codabench submissions, all made with my own models from the test prompts. Run the commands from this folder.

| Submission | Codabench | How to reproduce |
|---|---|---|
| Arch 1, a second sampling run with the same settings | 0.8291 | `python -m experiments.resample` |
| Arch 2 trained further on 329M new tokens, plain sampling | 0.8351 | `python -m experiments.continue_arch2` |
| Arch 2, one T35 sample | 0.8344 | `generate_mbr.py` with `N_SAMPLES = 1` |
| Arch 2, one T35 sample with another seed | 0.8354 | `python -m experiments.mbr_pool`, folder `single_t35_b` |
| Arch 2, MBR over 4 T35 samples | 0.8378 | `generate_mbr.py` with `N_SAMPLES = 4` |
| Arch 2, MBR over T35 samples 5 to 8 | 0.8369 | `python -m experiments.mbr_pool`, folder `mbr4b_t35` |
| Arch 2, MBR over 8 T35 samples (first MBR run) | 0.8388 | `python -m experiments.mbr8` |
| Arch 2, MBR over 8 T35 samples | 0.8390 | `generate_mbr.py` with `N_SAMPLES = 8` |
| Arch 2, MBR over T35 samples 9 to 16 | 0.8387 | `python -m experiments.mbr_pool`, folder `mbr8b_t35` |
| Arch 2, MBR over 12 T35 samples | 0.8396 | `generate_mbr.py` with `N_SAMPLES = 12` |
| Arch 2, MBR over T35 samples 5 to 16 | 0.8395 | `python -m experiments.mbr_pool`, folder `mbr12b_t35` |
| Arch 2, MBR over 1 sample of each setting | 0.8368 | `python -m experiments.mbr_pool`, folder `mix4` |
| Arch 2, MBR over 2 samples of each setting | 0.8383 | `python -m experiments.mbr_pool`, folder `mix8` |
| Arch 2, MBR over 3 samples of each setting | 0.8387 | `python -m experiments.mbr_pool`, folder `mix12` |
| Arch 2, MBR over 4 samples of each setting | 0.8388 | `python -m experiments.mbr_pool`, folder `mix16` |
| Arch 2, MBR over 8 T35 samples and 4 of each other setting | 0.8390 | `python -m experiments.mbr_pool`, folder `mix20` |
| Arch 2, MBR over all 28 samples | 0.8400 | `python -m experiments.mbr_pool`, folder `all28` |
| Arch 3 and Arch 4, the same files sent a second time | 0.8275, 0.8365 | `pretrain_arch3.py`, `pretrain_arch4.py` |

When changing `N_SAMPLES` in `generate_mbr.py`, I also change `OUTPUT_DIR` so the 16 sample run is kept. `experiments.mbr_pool` samples 28 continuations per prompt with Arch 2: 16 with T35, 4 with T36 (temperature 0.3, top p 0.85, top k 50, repetition penalty 1.2), 4 with the plain Arch 2 setting, and 4 with T34 (temperature 0.6, top p 0.95, top k 20, repetition penalty 1.1). It writes one zip per row to `outputs/pool/<folder>/`. Its T35 samples use the same seeds as `generate_mbr.py`. `experiments.continue_arch2` needs `outputs/arch2/model.pt` and `outputs/arch2/checkpoint.pt` from `pretrain.py`.
