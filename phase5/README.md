# Phase 5: Legal Statute Identification (LSI)

## Task and metric

The input is the facts section of an Indian court case. Names are masked as `<ENTITY>` and every statute citation is
masked as `<SECTION>` and `<ACT>`. The output is the set of Indian Penal Code sections applied to the case, out of 100
labels (ILSI dataset, IL-TUR packaging). The metric is macro-F1 over the 100 labels, so every statute counts equally.

A submission is a zip with one file, `prediction.json`, a list of `{"id": ..., "pred": [label ids]}`.

## Data and preprocessing

- Train 42,750, dev 10,181, test 13,019 cases. About 3.7 labels per case, every case has at least one.
- All 100 labels occur in train and dev. The rarest label has 272 train cases. Label frequencies span a factor of 43.
- Test has no labels. Only the official train and dev labels are used.
- Cleaning (`data.clean`): `<ENTITY>` becomes `X`, `<SECTION>` becomes `Section`, `<ACT>` becomes `Act`, and
  whitespace is collapsed. `<ENTITY>` costs 7 of my tokens and also hides ordinary words, so this cuts 19% of tokens.
- Tokenizer: my own Phase 1 BPE (16,384 tokens), applied to the letter, digit and punctuation pieces of each word as
  in Phase 2 (`tokenizer.PieceTokenizer`). Only the last piece of a word gets the end-of-word marker.
- A case longer than 20,000 tokens keeps its first 18,976 and its last 1,024 tokens.
- Chunks (`data.case_chunks`): windows of 1024 tokens (Arch 3: 703 tokens plus one prefix token, 704 in total) from
  the start of the facts, with a quarter overlap. At most k chunks per case (k = 4 in training); a longer case keeps
  its first k - 1 chunks and its last chunk. The median case is about 650 tokens, and about 90% of cases fit in 4
  chunks.

## Models

All four Phase 1 models are used (Arch 1 97.6M, Arch 2 61.6M, Arch 3 86.1M and Arch 4 139.1M parameters). Their code and full architecture table are in `../phase1` (see `../phase1/README.md`). Their task heads and fine-tuning code are here.

Only the 10-layer encoder of Arch 3 is used. Arch 4 is the bidirectional version from Phase 2 (causal mask removed, 800 updates of masked next-token prediction,
`../phase2/adapt.py mntp`). The Arch 3 encoder is bidirectional by design and gets one prefix token (id 16384).
Arch 1 and Arch 2 stay causal.

### Whole-case model (`model.CaseModel`)

One training example is one whole case: all its chunks go through the backbone together and the loss is on the
case's label set. No chunk is labelled on its own.

- Label attention head (`head="attn"`): the Assignment 2 Bahdanau attention with one learned query per statute
  (100 queries, attention size 128) over every token state of the case, then one output vector and bias per label.
- Pooled head (`head="pool"`): the Assignment 4 Q2 classifier. Each chunk gives one vector (last token for a causal
  model, mean of tokens for a bidirectional one), the chunk vectors are averaged and a linear layer gives 100 logits.
- The output bias starts at each label's log prior odds from the training set.
- Loss: binary cross-entropy over 100 sigmoids (`bce`), or the asymmetric loss (`asl`, gamma 4, clip 0.05) for F1.

## Training and decisions

### Fine-tuning per Phase 1 model

All techniques come from the Assignment 4 skeletons (`peft_methods.py`). Trainable counts include the head.

| Run | Model | Technique | Head / loss | Trainable | Peak LR | Epochs |
|---|---|---|---|---|---|---|
| F6 | Arch 1 | full fine-tuning, fp32 | pooled / BCE | 97.7M | 3e-5 | 1 |
| F5 | Arch 3 encoder | bottleneck adapters (size 64) after every encoder block | attention / BCE | 1.2M | 5e-4 | 3 |
| F1 | Arch 4 bidirectional | LoRMA_+ (rank 16, alpha 32) on every linear layer of the blocks | attention / ASL | 4.5M | 3e-4 | 2 |
| F1B | Arch 4 bidirectional | LoRMA_+ (rank 16, alpha 32) | attention / BCE | 4.5M | 3e-4 | 3 |
| F1C | Arch 4 bidirectional | LoRMA_+ (rank 16, alpha 32) | pooled / BCE | 4.2M | 3e-4 | 3 |
| F3 | Arch 4 bidirectional | QLoRA: packed 4-bit NF4 base (block 64) + LoRA rank 16, alpha 32 | attention / BCE | 4.0M | 3e-4 | 2 |
| F4 | Arch 2 | partial fine-tuning: top 8 of 12 blocks and the final norm | attention / BCE | 34.4M | 5e-5 | 3 |
| F8 | Arch 2 | full fine-tuning | attention / BCE | 62.0M | 3e-5 | 3 |
| F7 | Arch 2 | partial (as F4) + distillation from F1 | attention / BCE + KD | 34.4M | 5e-5 | 3 |
| G1, G2 | Arch 2 | partial, lr 5e-5 and 1e-4 | attention / BCE | 34.4M | 5e-5, 1e-4 | 6 |
| G3 | Arch 2 | full fine-tuning | attention / BCE | 62.0M | 3e-5 | 6 |
| G6 | Arch 2 | partial, trained on up to 8 chunks | attention / BCE | 34.4M | 5e-5 | 6 |
| G4 | Arch 2 | partial + distillation from F1 (best single model) | attention / BCE + KD | 34.4M | 5e-5 | 6 |

Shared settings (`train.train_cases`): one case per forward pass, 16 cases per update, AdamW with weight decay 0.01,
head learning rate 1e-3, 6% linear warmup then linear decay to 0, gradient clipping 1.0, fp16 autocast with a
gradient scaler (Arch 1 in fp32, since it did not learn in fp16 in Phase 3). Dropout stays as in the Phase 1 config
(0 for Arch 2, 0.1 for Arch 1). Every `eval_every` updates the model is scored on a fixed random dev subset of
`eval_n` cases with one global threshold, and the best trainable weights are kept. Each run has a time cap
(`max_minutes`). Seeds 1 to 4 are the same configs with another seed.

### Distillation (`distill.py`)

The teacher is F1 (Arch 4 LoRMA_+). After training it writes its logits on every one of the 42,750 training cases
once (`teacher_train.npz`). The student uses exactly F4's recipe (Arch 2, partial fine-tuning, same data and updates).
The Assignment 4 Q6 loss is applied per label as a 2-class problem with logits [0, z], so the softmax is the sigmoid
and the CE term is BCE: `(1 - kd_weight) CE + kd_weight tau^2 KL`, with tau 2 and kd_weight 0.5.
Result (dev, cross-fitted): student F7 0.412, the same model without distillation (F4) 0.408, teacher F1 0.396.
With 6 epochs the student (G4) reaches 0.420.

### Decision rule (`thresholds.py`)

- Per-label thresholds on a grid from 0.02 to 0.89, shrunk toward one global threshold:
  `t' = (n t + 50 g) / (n + 50)` with n the number of dev positives of the label.
- Every case gets at least one label (the one with the highest p / t if none clears its threshold).
- The dev number I report uses 2-fold cross-fitting: thresholds tuned on one half of dev, scored on the other half, and
  swapped. The final thresholds use all of dev and are applied to test.

### Ensemble and kNN label votes (`ensemble.py`, `knn_votes.py`)

- Base: the logits of 12 runs at k = 4 chunks are averaged, then passed through a sigmoid. Members: 4 Arch 4 runs
  (F1, F1B, F1B seed 1, F3), 3 six-epoch Arch 2 partial runs (G1, G2, G2 seed 1) and 5 six-epoch distilled students
  (G4 and seeds 1 to 4).
- Neighbours: TF-IDF over the Phase 1 BPE token ids (unigrams and bigrams, min_df 2, up to 400,000 features,
  sublinear tf), fitted on train only. Every dev and test case gets its 50 most similar train and dev cases by
  cosine similarity. A dev case never retrieves itself. A test case is only a query.
- Votes: the label vectors of the k = 50 nearest cases, weighted by similarity^8.
- Gated blend: `p = 0.8 p_model + 0.2 votes`, and `p = 0.3 p_model + 0.7 votes` when the top-1 similarity is at
  least 0.8.
- Per-label thresholds are then tuned on dev as above.

## Code layout

| File | What it does |
|---|---|
| `tokenizer.py` | Phase 1 BPE tokenizer (loading and encoding only) and the piece tokenizer from Phase 2 |
| `data.py` | cleaning, tokenizing every split into `data/lsi_tokens/{train,dev,test}.npz`, loading them, case chunks |
| `model.py` | loads a Phase 1 backbone without its LM head, the pooled classifier, Bahdanau attention, `CaseModel`, BCE and ASL losses |
| `peft_methods.py` | bottleneck adapters, LoRMA_+ (with the dense vs reordered check), NF4 quantisation and `QLoRALinear`, partial and full fine-tuning, `attach_peft` |
| `distill.py` | Assignment 4 distillation loss and its checks, per-label 2-class form, loading the teacher cache |
| `thresholds.py` | macro-F1, global and per-label thresholds, at least one label, cross-fitting, the dev report |
| `predict.py` | writes and checks a Codabench zip; also makes a single run's submission from its `scores.npz` |
| `train.py` | the run configs (`RUNS`) and one full run: fine-tune, predict dev and test, score, write the submission |
| `knn_votes.py` | builds the TF-IDF index over BPE ids and the top-50 neighbours; vote and gated blend functions |
| `ensemble.py` | the best submission: 12-run logit average + gated kNN votes + per-label thresholds |

## How to run

Dependencies: Python 3, PyTorch, NumPy, SciPy, scikit-learn. Training needs a GPU (I used one Kaggle T4 per run);
the tokenizing, kNN and ensemble steps run on CPU.

Paths are relative to this folder. Set them at the top of the files if your layout differs:

- `data.py`: `DATA_DIR = "data/lsi"` (`train.json`, `dev.json`, `test.json` with `id`, `text`, `labels`),
  `TOKEN_DIR = "data/lsi_tokens"`, `TOKENIZER_PATH = "tokenizer/legal_bpe_16384.json"` (the Phase 1 tokenizer).
- `train.py`: `MODEL_DIRS` (`models/arch1`, `models/arch2`, `models/arch3`, `models/arch4_mntp`), `TEACHER_PATH`,
  `OUT_DIR = "runs"`. Each model folder holds `model.pt`, the model code as `model_arch.py` (copy the matching
  `../phase1/archN_model.py`), and for Arch 3 and Arch 4 the `metrics.json` with `model_config` written by Phase 1
  (Arch 4 also `arch4_engram_compress.pt`). `models/arch4_mntp` is the bidirectional Arch 4 from Phase 2.
- `knn_votes.py`: `TOKEN_DIR`, `OUT_FILE = "runs/knn_bpe.npz"`. `ensemble.py`: `RUNS_DIR`, `KNN_FILE`, `OUT_PATH`.

Order:

1. `python data.py` tokenizes all splits (CPU, 4 processes).
2. `python train.py f1_arch4_lorma_attn_asl` trains the teacher. It also writes `runs/f1_arch4_lorma_attn_asl/teacher_train.npz`.
3. `python train.py <run name>` for every other run in `RUNS` (the students need step 2 first). Each run writes
   `runs/<run name>/log.txt`, `metrics.json`, `scores.npz` (dev and test logits), `submission_<run name>_k<k>.zip`
   (k with the best cross-fitted dev score) and, for runs with fewer than 30M trainable parameters, `trainable_state.pt`.
4. `python knn_votes.py` writes `runs/knn_bpe.npz` (top-50 neighbours of every dev and test case).
5. `python ensemble.py` writes `submissions/ens12_knn_bpe_k50_p8_a0.2_g0.8.zip` and prints the cross-fitted dev
   macro-F1 of the 12-run base (0.4435) and of the blend (0.4807).

A single run's submission can also be rebuilt from its saved scores:
`python predict.py runs/<run name>/scores.npz out.zip` (per-label thresholds on dev, k = 4 chunks).

Every run starts with the LoRMA, NF4 and distillation checks and stops if one fails.

## Codabench submissions covered by this code

Dev is the cross-fitted per-label macro-F1. Single runs: `python train.py <run name>`, then
`python predict.py runs/<run name>/scores.npz out.zip` (k = 4 chunks). This equals the zip that `train.py` writes,
except for F1, where `train.py` picks k = 8 (dev 0.3964 against 0.3963 at k = 4) and #03 used k = 4.
kNN rows: `ensemble.py` with the constants shown (`K_NEIGHBOURS`, `POWER`, `BLEND`, `GATE`, `BLEND_GATE`); the
default constants give #93.

| # | Command / config | Dev | Codabench |
|---|---|---|---|
| 01 | `train.py f6_arch1_full_pool_bce` | 0.100 | 0.0997 |
| 02 | `train.py f5_arch3_adapter_attn_bce` | 0.314 | 0.3161 |
| 03 | `train.py f1_arch4_lorma_attn_asl` | 0.396 | 0.3942 |
| 04 | `train.py f3_arch4_qlora_attn_bce` | 0.400 | no score (still Preparing) |
| 05 | `train.py f8_arch2_full_attn_bce` | 0.400 | 0.4031 |
| 06 | `train.py f1b_s1_arch4_lorma_attn_bce` | 0.408 | 0.4085 |
| 07 | `train.py f4_arch2_partial_attn_bce` | 0.408 | 0.4075 |
| 08 | `train.py f7_arch2_kd_from_f1` | 0.412 | 0.4087 |
| 22 | `train.py g4_arch2_kd_e6` | 0.420 | 0.4211 |
| 29 | `train.py g3_arch2_full_e6` | 0.399 | 0.4008 |
| 30 | `train.py f1c_arch4_lorma_pool_bce` | 0.400 | 0.3968 |
| 31 | `train.py f4s1_arch2_partial_attn_bce_s1` | 0.403 | 0.4017 |
| 32 | `train.py g1_arch2_partial_e6` | 0.403 | 0.3993 |
| 33 | `train.py g6_arch2_partial_e6_k8` | 0.403 | 0.4014 |
| 34 | `train.py g5_arch2_partial_e6_lr1e4_s1` | 0.403 | 0.4050 |
| 35 | `train.py f4s2_arch2_partial_attn_bce_s2` | 0.405 | 0.4055 |
| 36 | `train.py g2_arch2_partial_e6_lr1e4` | 0.406 | 0.4077 |
| 37 | `train.py f1b_arch4_lorma_attn_bce` | 0.406 | 0.4062 |
| 38 | `train.py f7s1_arch2_kd_from_f1_s1` | 0.411 | 0.4091 |
| 63 | `ensemble.py`, k 30, p 8, blend 0.25, gate 0.8 / 0.7 | 0.483 | 0.4724 |
| 64 | `ensemble.py`, k 30, p 8, blend 0.25, no gate (`GATE = 2`) | 0.480 | 0.4707 |
| 65 | `ensemble.py`, k 30, p 8, blend 0.25, gate 0.9 / 0.6 | 0.482 | 0.4722 |
| 75 | `ensemble.py`, k 20, p 8, blend 0.2, gate 0.8 / 0.7 | 0.482 | 0.4727 |
| 76 | as #75, but the BPE and 1-3-gram neighbour indexes are combined (the 1-3-gram index is not in this code) | 0.482 | 0.4727 |
| 85 | `ensemble.py`, k 50, p 8, blend 0.3, gate 0.8 / 0.7 | 0.481 | no score recorded |
| 86 | `ensemble.py`, k 20, p 8, blend 0.25, gate 0.8 / 0.7 | 0.481 | 0.4710 |
| 87 | `ensemble.py`, k 30, p 12, blend 0.25, gate 0.8 / 0.7 | 0.481 | 0.4688 |
| 89 | `ensemble.py`, k 30, p 8, blend 0.2, gate 0.8 / 0.7 | 0.481 | 0.4720 |
| 90 | `ensemble.py`, k 30, p 8, blend 0.3, gate 0.8 / 0.7 | 0.481 | 0.4703 |
| 92 | `ensemble.py`, k 50, p 8, blend 0.25, gate 0.9 / 0.6 | 0.481 | 0.4723 |
| **93** | **`ensemble.py` (defaults: k 50, p 8, blend 0.2, gate 0.8 / 0.7)** | **0.481** | **0.4732 (best)** |
| 95 | `ensemble.py`, k 50, p 8, blend 0.25, gate 0.8 / 0.7 | 0.481 | 0.4720 |
| 97 | `ensemble.py`, k 20, p 8, blend 0.3, gate 0.8 / 0.7 | 0.480 | 0.4700 |
| 98 | `ensemble.py`, k 20, p 8, blend 0.2, gate 0.9 / 0.6 | 0.480 | 0.4729 |
| 99 | `ensemble.py`, k 20, p 8, blend 0.25, gate 0.9 / 0.6 | 0.480 | 0.4721 |

My best submission is #93 with a Codabench score of 0.4732, but the leaderboard shows #76 (0.4727) as the selected entry.

Notes:

- The dev numbers of #63 to #65 are the mean over three random halvings of dev. `ensemble.py` prints the first
  halving only (0.482, 0.479 and 0.481). All other rows use the first halving.
- Not covered here: the model-only ensembles (#09 to #21, #23 to #28, #39 to #59, best 0.4451), the kNN runs on the 8-run base
  (#60 to #62), the other 1-3-gram index runs of the final grid, the dev-weighted neighbours (#66, #67) and the models also
  trained on half of dev (#72). The head and loss screening runs (pooled, BiLSTM, weighted BCE on 12,000 cases) are
  not included either.
- The 28% of test cases that also appear in the Phase 1 pretraining corpus with the section numbers visible were
  not used.
