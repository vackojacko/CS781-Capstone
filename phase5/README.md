# Phase 5: Legal Statute Identification (LSI)

## Task

The input is the facts of an Indian court case, with names and statute citations masked. The output is
the set of Indian Penal Code sections that apply, out of 100 labels (ILSI). The metric is macro-F1 over the 100 labels. A submission is a zip with
`prediction.json`, a list of `{"id": ..., "pred": [label ids]}`.

My best submission is #93 with a Codabench score of 0.4732. The leaderboard shows #76 (0.4727) as the
selected entry.

## Data

- Train 42,750, dev 10,181, test 13,019 cases, about 3.7 labels per case. Test has no labels.
- `data.clean` turns `<ENTITY>` into `X`, `<SECTION>` into `Section` and `<ACT>` into `Act`. This cuts 19%
  of the tokens.
- Tokenizer: my own Phase 1 BPE (16,384 tokens, `tokenizer.PieceTokenizer`).
- A case longer than 20,000 tokens keeps its first 18,976 and last 1,024 tokens.
- Chunks (`data.case_chunks`): windows of 1024 tokens (704 for Arch 3) with a quarter overlap, at most k
  per case (k = 4 in training). A longer case keeps its first k - 1 chunks and its last chunk.

## Models

The Phase 1 models are in `../phase1`. Arch 3 is used as an encoder only, and Arch 4 is the bidirectional
version from Phase 2 (`../phase2/adapt.py mntp`).

`model.CaseModel` reads a whole case (all chunks) per example. The `attn` head is my Assignment 2 Bahdanau
attention with one query per statute (size 128). The `pool` head is the Assignment 4 Q2 classifier. The
loss is BCE or asymmetric loss (ASL, gamma 4, clip 0.05).

## Fine-tuning runs

All techniques come from the Assignment 4 skeletons (`peft_methods.py`). Trainable counts include the head.

| Run | Model | Technique | Head, loss | Trainable | Peak LR | Epochs |
|---|---|---|---|---|---|---|
| F6 | Arch 1 | full, fp32 | pooled, BCE | 97.7M | 3e-5 | 1 |
| F5 | Arch 3 | adapters, size 64 | attn, BCE | 1.2M | 5e-4 | 3 |
| F1 | Arch 4 | LoRMA_+ r 16, alpha 32, all block linears | attn, ASL | 4.5M | 3e-4 | 2 |
| F1B | Arch 4 | LoRMA_+ r 16, alpha 32 | attn, BCE | 4.5M | 3e-4 | 3 |
| F1C | Arch 4 | LoRMA_+ r 16, alpha 32 | pooled, BCE | 4.2M | 3e-4 | 3 |
| F3 | Arch 4 | QLoRA: NF4 base (block 64), LoRA r 16, alpha 32 | attn, BCE | 4.0M | 3e-4 | 2 |
| F4 | Arch 2 | partial: top 8 of 12 blocks + final norm | attn, BCE | 34.4M | 5e-5 | 3 |
| F8 | Arch 2 | full | attn, BCE | 62.0M | 3e-5 | 3 |
| F7 | Arch 2 | F4 + distillation from F1 | attn, BCE, KD | 34.4M | 5e-5 | 3 |
| G1, G2 | Arch 2 | partial, lr 5e-5 and 1e-4 | attn, BCE | 34.4M | 5e-5, 1e-4 | 6 |
| G3 | Arch 2 | full | attn, BCE | 62.0M | 3e-5 | 6 |
| G6 | Arch 2 | partial, up to 8 chunks | attn, BCE | 34.4M | 5e-5 | 6 |
| G4 | Arch 2 | G1 + distillation from F1 (best single model) | attn, BCE, KD | 34.4M | 5e-5 | 6 |

Shared settings (`train.train_cases`): 16 cases per update, AdamW, weight decay 0.01, head LR 1e-3, 6%
warmup, clipping 1.0, fp16 (Arch 1 in fp32). Names ending in `_s1` to `_s4` only change the seed.

Distillation (`distill.py`): the teacher F1 saves its logits on all training cases once
(`teacher_train.npz`). The student follows the F4 recipe. The Assignment 4 Q6 loss is used per label as a
2-class problem, with tau 2 and kd_weight 0.5.

## Decision rule and ensemble

- `thresholds.py`: each label gets its own threshold from a grid from 0.02 to 0.89, pulled toward one
  global threshold (shrink 50). Every case gets at least one label.
- Dev numbers use 2-fold cross-fitting on dev. The final thresholds use all of dev.
- `ensemble.py` averages the logits of 12 runs at k = 4: F1, F1B, F1B seed 1, F3, G1, G2, G2 seed 1, and
  G4 with its seeds 1 to 4.
- `knn_votes.py` finds the 50 nearest train and dev cases of each dev and test case (TF-IDF over BPE ids,
  fitted on train only). A dev case never finds itself.
- The neighbours' labels, weighted by similarity to the power 8, are blended in: 0.2 votes and 0.8 model,
  or 0.7 votes and 0.3 model when the closest neighbour has similarity 0.8 or more.

## Files

- `tokenizer.py`, `data.py`: tokenizer, cleaning, tokens in `data/lsi_tokens/`, case chunks.
- `model.py`: backbone loading, heads, `CaseModel`, losses.
- `peft_methods.py`: adapters, LoRMA_+, NF4 and `QLoRALinear`, partial and full fine-tuning.
- `distill.py`, `thresholds.py`: distillation loss, thresholds and cross-fitting.
- `train.py`: the run configs (`RUNS`) and one full run. `predict.py`: zip writer.
- `knn_votes.py`, `ensemble.py`: neighbours and the best submission.

## How to run

Needs Python 3, PyTorch, NumPy, SciPy, scikit-learn, and a GPU for training (I used one Kaggle T4 per run).
Paths are relative to this folder:

| File | Constants |
|---|---|
| `data.py` | `DATA_DIR = "data/lsi"`, `TOKEN_DIR = "data/lsi_tokens"`, `TOKENIZER_PATH = "tokenizer/legal_bpe_16384.json"` |
| `train.py` | `MODEL_DIRS` (`models/arch1`, `models/arch2`, `models/arch3`, `models/arch4_mntp`), `TEACHER_PATH`, `OUT_DIR = "runs"` |
| `knn_votes.py` | `TOKEN_DIR`, `OUT_FILE = "runs/knn_bpe.npz"` |
| `ensemble.py` | `RUNS_DIR`, `KNN_FILE`, `OUT_PATH` |

Each model folder holds `model.pt` and `model_arch.py` (copy of `../phase1/archN_model.py`). Arch 3 and 4
also need the Phase 1 `metrics.json`, and Arch 4 needs `arch4_engram_compress.pt`.

1. `python data.py` tokenizes all splits.
2. `python train.py f1_arch4_lorma_attn_asl` trains the teacher and writes its `teacher_train.npz`.
3. `python train.py <run name>` for every other run in `RUNS`. Each writes `log.txt`, `metrics.json`,
   `scores.npz` and a submission zip to `runs/<run name>/`.
4. `python knn_votes.py` writes `runs/knn_bpe.npz`.
5. `python ensemble.py` writes `submissions/ens12_knn_bpe_k50_p8_a0.2_g0.8.zip` and prints the dev
   macro-F1 of the 12-run base (0.4435) and of the blend (0.4807).

A single run's submission can be rebuilt with `python predict.py runs/<run name>/scores.npz out.zip`
(k = 4 chunks). Every run first runs the LoRMA, NF4 and distillation checks.

## Codabench submissions

Dev is the cross-fitted macro-F1. Single runs use `predict.py` with k = 4, which matches the zip from
`train.py` except for F1 (`train.py` picks k = 8, dev 0.3964 against 0.3963). The kNN rows are
`ensemble.py` with `K_NEIGHBOURS` (k), `POWER` (p), `BLEND` and the gate (`GATE` / `BLEND_GATE`). The
defaults give #93.

| # | Run (`train.py <run>`) | Dev | Codabench |
|---|---|---|---|
| 01 | `f6_arch1_full_pool_bce` | 0.100 | 0.0997 |
| 02 | `f5_arch3_adapter_attn_bce` | 0.314 | 0.3161 |
| 03 | `f1_arch4_lorma_attn_asl` | 0.396 | 0.3942 |
| 04 | `f3_arch4_qlora_attn_bce` | 0.400 | no score (still Preparing) |
| 05 | `f8_arch2_full_attn_bce` | 0.400 | 0.4031 |
| 06 | `f1b_s1_arch4_lorma_attn_bce` | 0.408 | 0.4085 |
| 07 | `f4_arch2_partial_attn_bce` | 0.408 | 0.4075 |
| 08 | `f7_arch2_kd_from_f1` | 0.412 | 0.4087 |
| 22 | `g4_arch2_kd_e6` | 0.420 | 0.4211 |
| 29 | `g3_arch2_full_e6` | 0.399 | 0.4008 |
| 30 | `f1c_arch4_lorma_pool_bce` | 0.400 | 0.3968 |
| 31 | `f4s1_arch2_partial_attn_bce_s1` | 0.403 | 0.4017 |
| 32 | `g1_arch2_partial_e6` | 0.403 | 0.3993 |
| 33 | `g6_arch2_partial_e6_k8` | 0.403 | 0.4014 |
| 34 | `g5_arch2_partial_e6_lr1e4_s1` | 0.403 | 0.4050 |
| 35 | `f4s2_arch2_partial_attn_bce_s2` | 0.405 | 0.4055 |
| 36 | `g2_arch2_partial_e6_lr1e4` | 0.406 | 0.4077 |
| 37 | `f1b_arch4_lorma_attn_bce` | 0.406 | 0.4062 |
| 38 | `f7s1_arch2_kd_from_f1_s1` | 0.411 | 0.4091 |

| # | k | p | blend | gate | Dev | Codabench |
|---|---|---|---|---|---|---|
| 63 | 30 | 8 | 0.25 | 0.8/0.7 | 0.483 | 0.4724 |
| 64 | 30 | 8 | 0.25 | none (`GATE = 2`) | 0.480 | 0.4707 |
| 65 | 30 | 8 | 0.25 | 0.9/0.6 | 0.482 | 0.4722 |
| 75 | 20 | 8 | 0.2 | 0.8/0.7 | 0.482 | 0.4727 |
| 76 | as #75, with a 1-3-gram index added (not in this code) | | | | 0.482 | 0.4727 |
| 85 | 50 | 8 | 0.3 | 0.8/0.7 | 0.481 | no score recorded |
| 86 | 20 | 8 | 0.25 | 0.8/0.7 | 0.481 | 0.4710 |
| 87 | 30 | 12 | 0.25 | 0.8/0.7 | 0.481 | 0.4688 |
| 89 | 30 | 8 | 0.2 | 0.8/0.7 | 0.481 | 0.4720 |
| 90 | 30 | 8 | 0.3 | 0.8/0.7 | 0.481 | 0.4703 |
| 92 | 50 | 8 | 0.25 | 0.9/0.6 | 0.481 | 0.4723 |
| 93 (best, defaults) | 50 | 8 | 0.2 | 0.8/0.7 | 0.481 | 0.4732 |
| 95 | 50 | 8 | 0.25 | 0.8/0.7 | 0.481 | 0.4720 |
| 97 | 20 | 8 | 0.3 | 0.8/0.7 | 0.480 | 0.4700 |
| 98 | 20 | 8 | 0.2 | 0.9/0.6 | 0.480 | 0.4729 |
| 99 | 20 | 8 | 0.25 | 0.9/0.6 | 0.480 | 0.4721 |

Notes:

- Dev of #63 to #65 is the mean over three random halvings of dev. `ensemble.py` prints only the first
  halving (0.482, 0.479, 0.481), which all other rows use.
- Other submissions (model-only ensembles, best 0.4451, and other kNN variants) are not covered here.
- 28% of test cases also appear in the Phase 1 pretraining corpus with the section numbers visible. I did
  not use that text.
