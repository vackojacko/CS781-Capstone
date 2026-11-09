# Phase 1: Legal Language Model Pretraining

## Task

The input is a 512-word prompt from an Indian legal document. The model has to write the next 128 words. Codabench scores the output with BERTScore F1 (roberta-large).

## My final system

- I trained four language models from scratch, each between 50M and 300M parameters, with my own BPE tokenizer from Assignment 1 (16,384 tokens).
- My Arch 2 model, a LLaMA style decoder, was used for the later steps.
- MBR decoding: Arch 2 writes 16 continuations for each prompt, and I keep the one that agrees most with the others (ROUGE-1/2 F1).
- Retrieval: for 40% of the test prompts, 12 words near the end of the prompt appear word for word in the training files. I give the model the 128 words that follow them there.
- I fine-tuned Arch 2 so it can use this passage, and also ignore it when it is wrong.
- The best score is 0.8688. For each prompt, it picks the output that agrees most with outputs from my other finished runs.

## Models I tried

| Phase 1 model | Technique | Codabench score |
|---|---|:--:|
| Arch 1: GPT-2 style decoder, 97.6M | plain sampling | 0.8299 |
| Arch 2: LLaMA style decoder, 61.6M | plain sampling | 0.8334 |
| Arch 3: encoder-decoder, 86.1M | plain sampling | 0.8275 |
| Arch 4: latent-attention decoder with n-gram memory, 139.1M | plain sampling | 0.8365 |
| Arch 2 | MBR over 16 samples | 0.8402 |
| Arch 2 | retrieval fine-tune + MBR over 16 samples | 0.8681 |
| Arch 2 and other runs | vote over finished runs | 0.8688 |

All models are written by hand in PyTorch, starting from my Assignment 3 code. The MBR runs use the sampling setting called `T35` in the code, which was the best of 40 settings I tried for Arch 2.

## Files

| File | What it does |
|---|---|
| `tokenizer.py` | my Assignment 1 BPE tokenizer |
| `train_tokenizer.py` | trains the tokenizer and writes `legal_bpe_16384.json` |
| `legal_bpe_16384.json` | the trained tokenizer |
| `arch1_model.py` to `arch4_model.py` | the four models |
| `pretrain.py` | trains Arch 1 or Arch 2 and writes test outputs |
| `pretrain_arch3.py` | trains Arch 3 and writes test outputs |
| `pretrain_arch4.py` | trains Arch 4, checks it on held-out prompts, writes test outputs |
| `heldout_set.py` | picks 400 held-out prompts for tuning and checking |
| `generate_mbr.py` | samples 16 outputs with Arch 2 and keeps the MBR choice |
| `retrieval.py` | finds matching passages in the training files |
| `retrieval_finetune.py` | fine-tunes Arch 2 to use the passages |
| `predict.py` | makes the retrieval + MBR submission |
| `consensus.py` | makes the best submission by voting over finished runs |
| `ablations/` | small runs that test parts of Arch 2 and Arch 4 |

## How to run

Set these constants at the top of the scripts if your paths differ:

- `DATA_DIR = "data"`, the folder with `train_supreme_court.json`, `train_high_courts.json`, `train_acts.json`, `train_tribunals.json` and `test.json`.
- `TOKENIZER_PATH = "legal_bpe_16384.json"`.
- `ARCH` in `pretrain.py`, set to `"arch1"` or `"arch2"`.
- Outputs go to `outputs/<name>/`.

Run in this order:

```
python train_tokenizer.py
python pretrain.py                 # once with ARCH = "arch1", once with ARCH = "arch2"
python pretrain_arch3.py
python pretrain_arch4.py
python heldout_set.py
python generate_mbr.py
python retrieval.py
python retrieval_finetune.py
python predict.py
python consensus.py
```

- The training scripts continue from their last checkpoint if they are run again. I trained on a Kaggle T4 GPU.
- `generate_mbr.py`, `retrieval_finetune.py` and `predict.py` need `outputs/arch2/model.pt`.
- In `predict.py`, `RUN_HELDOUT = True` also runs a held-out BERTScore check.
- `consensus.py` reads the saved run files listed at its top and writes `submission_consensus.zip`.

Dependencies: Python 3, torch, numpy, ijson, matplotlib, pyahocorasick, and bert_score for the held-out checks.

## Submissions this code reproduces

| Submission | Score |
|---|:--:|
| Arch 1, plain sampling (`outputs/arch1/`) | 0.8299 |
| Arch 2, plain sampling (`outputs/arch2/`) | 0.8334 |
| Arch 3, plain sampling (`outputs/arch3/`) | 0.8275 |
| Arch 4, plain sampling (`outputs/arch4/`) | 0.8365 |
| Arch 2 + MBR x16 (`outputs/mbr16/`) | 0.8402 |
| Arch 2 + retrieval fine-tune + MBR x16 (`outputs/final/`) | 0.8681 |
| Vote over finished runs (`submission_consensus.zip`) (best) | 0.8688 |

## Notes

- One voter file for the vote was not kept. Because of this, `consensus.py` matches the submitted file on 956 of 1,000 prompts.
- Two other runs are not made by this folder: Arch 2 trained on 329M more tokens (0.8351) and Arch 2 + MBR over 8 samples (0.8388). The vote uses outputs from the first one.
