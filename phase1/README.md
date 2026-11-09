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
