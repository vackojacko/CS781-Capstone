# Phase 4: Court Judgment Prediction (CJPE)

## Task

Each input is the full text of an Indian Supreme Court judgment with the final decision removed (ILDC).
The model predicts 1 if the appeal was accepted and 0 if it was rejected. The metric is macro-F1 over the
1,517 test cases. A submission is `submission.zip` with `prediction.json`, holding
`[{"id": ..., "pred": 0 or 1}, ...]`.

Best Codabench score: 0.8154, the average of three Arch 4 LoRMA runs with the hierarchical input.

## Data

| | train | dev | test |
|---|---|---|---|
| rows | 37,387 | 1,988 | 1,517 |
| unique cases used | 32,305 | 994 | 1,517 |
| label 0 / 1 | 59 / 41 | 50 / 50 | balanced |

- Repeated case ids are dropped in train and dev.
- `test.json` has labels, but the loader sets every test label to -1, so they are never used.
- `data.align_train` rewrites train into the lowercased dev and test format.
- Tokenizer: my Phase 1 BPE (16,384 tokens), used as in Phases 2 and 3 (`tokenizer.PieceTokenizer`).
- Train cases keep their last 5,600 words and then their last 6,400 tokens. Dev and test are kept whole
  (up to 50,000 tokens).
- Chunks are 1024 tokens with stride 768, counted back from the end of the case, where the outcome is.
  Arch 3 chunks are 703 tokens plus a continuation marker.
- 99.5% of test cases also appear in the Phase 1 pretraining file with their decision. I do not use that.

## Models and method

The Phase 1 models are in `../phase1`. Arch 1 and Arch 2 are used as causal decoders. For Arch 3 I use only
the encoder (60.6M with adapters). Arch 4 is the bidirectional version from Phase 2.

- Chunk classifier (`model.SentimentClassifier`): backbone, pooled vector, `Linear(d, 2)`.
- Aggregator (`model.Aggregator`): a BiLSTM (my Assignment 2 LSTM, hidden 128) over the last 8 chunks.

Stage 1 fine-tunes the chunk classifier. Each epoch, every training case gives one random chunk from its
last 2 chunks (last 3 for E6). Settings: cross-entropy, AdamW with weight decay 0.01, head LR 1e-3, 6%
warmup, clipping 1.0, 32 chunks per update (4 x 8, Arch 1: 2 x 16), fp16 (Arch 1 in fp32). Dev is scored
every 400 updates and the best weights are kept.

| Run | Model | Technique (`peft_methods.py`) | Settings | Trainable |
|---|---|---|---|---|
| E1 | Arch 4 bidirectional | LoRMA+ on every `nn.Linear` of every block | r 16, alpha 32, B = 0 at start, lr 3e-4, 2 epochs | 4,146,434 of 143.2M |
| E2 | Arch 2 | gradual unfreezing | top 4 blocks + final norm + head, top 8 at 25% of updates, all at 50%, lr 5e-5, 2 epochs | 17.0M, 34.1M, then 61.6M |
| E3 | Arch 3 encoder | bottleneck adapter after every encoder block | size 64, GELU, up-projection zero at start, lr 5e-4, 3 epochs | 827,522 of 60.6M |
| E4 | Arch 1 | full fine-tuning, fp32 | lr 3e-5, batch 2 x 16, 1 epoch, 200-minute cap | 97.6M |
| E6 | Arch 4 bidirectional | LoRMA+ as E1 | trained on the last 3 chunks, seed 1 | 4,146,434 |
| E10 | Arch 4 bidirectional | LoRMA+ | r 32, alpha 64, 3 epochs, seed 2 | 8,291,586 |
| E11 | Arch 4 bidirectional | LoRMA+ as E1 | seed 3 | 4,146,434 |
| E5 | Arch 2 student | E2 recipe with distillation | see below | 17.0M to 61.6M |
| E12 | Arch 4 student | E1 recipe with distillation | see below | 4,146,434 |

The LoRMA and distillation checks from Assignment 4 are `peft_methods.check_lorma()` and
`distill.check_distillation_loss()`.

Distillation: E1, E6 and E2 are the teachers. Each saves its logits for the last two chunks of the same
16,000 training cases to `runs/<run>/teacher_train.npz`. The student loss is
`0.5 CE + 0.5 tau^2 KL` on the average teacher logits, with tau 2. E5 (Arch 2, 61.6M) scored 0.7638, which
is 95% of the teacher ensemble's 0.8048 (#40) with 18% of its parameters (61.6M against 339.8M).

Stage 2 freezes the model. The aggregator trains on the last 8 chunks of 6,000 training cases (3,000 for
Arch 1) for 15 epochs (AdamW, lr 1e-3, batch 64).

### Input variants

Each run scores six inputs with the same model: `first` chunk, `selected` (the 2 chunks with the most cue
words in `train.CUES`), `last` chunk, `tail2` (last 2 chunks), `whole` (all chunks) and `hier` (the
aggregator).

### Threshold and ensemble

Train is 59/41 but dev and test are balanced, so every submission uses the threshold with the best dev
macro-F1 among 181 dev score quantiles (5% to 95%). The
prior-shift rule scored lower (E1 `hier`: 0.7814 against 0.8038).

`ensemble.py` standardises the `hier` scores of E1, E10 and E11 with each run's dev mean and standard
deviation, averages them and tunes the threshold on dev. Dev 0.8147 (threshold 0.079), 49.0% of test cases
predicted 1, Codabench 0.8154.

## Files

| File | What it does |
|---|---|
| `data.py` | loads and cleans the splits, writes `tokens/{train,dev,test}.npz`, chunk helpers. Run it once. |
| `tokenizer.py` | Phase 1 BPE applied to the pieces of each word |
| `backbone.py` | loads a Phase 1 checkpoint without its LM head, makes Arch 4 bidirectional, keeps the Arch 3 encoder |
| `model.py` | pooling, chunk classifier, Assignment 2 LSTM, BiLSTM aggregator |
| `peft_methods.py` | adapters, LoRMA+ with its checks, `unfreeze_top`, `attach_peft` |
| `distill.py` | distillation loss with its checks, teacher caches (`TEACHER_FILES`) |
| `predict.py` | variant scores, dev threshold search, submission zip writer |
| `train.py` | the run configs (`CONFIGS`) and one full run |
| `ensemble.py` | the best submission from E1, E10 and E11 |

## How to run

Needs Python 3, PyTorch, NumPy, scikit-learn, and a GPU for training (I used one Kaggle T4 per run). Run
every command from this folder.

| File | Path constants |
|---|---|
| `data.py` | `DATA_DIR = "data/cjpe"` (`train.json`, `dev.json`, `test.json`), `BPE_PATH = "data/legal_bpe_16384.json"`, `TOKEN_DIR = "tokens"` |
| `train.py` | `OUT_DIR = "runs"`, `MODEL_DIRS` (one folder per architecture, Arch 4 is the bidirectional Phase 2 checkpoint) |
| `distill.py` | `TEACHER_FILES`, the caches written by E1, E6 and E2 |
| `ensemble.py` | `RUNS_DIR`, `OUT_PATH`, `MEMBERS`, `VARIANT` |

Each model folder needs `model.pt` and its `../phase1` model file as `model_arch.py`. Arch 3 and 4 also
need the Phase 1 `metrics.json`.

```
python data.py                                   # tokens/train.npz, dev.npz, test.npz
python train.py e1_arch4_lorma                   # teacher; writes runs/e1_arch4_lorma/teacher_train.npz
python train.py e6_arch4_lorma_tail3_s1          # teacher
python train.py e2_arch2_unfreeze                # teacher
python train.py e3_arch3_adapter
python train.py e4_arch1_full
python train.py e10_arch4_lorma_r32_e3_s2
python train.py e11_arch4_lorma_s3
python train.py e5_arch2_kd                      # needs the three teacher caches
python train.py e12_arch4_lorma_kd               # needs the three teacher caches
python ensemble.py                               # submission_ens_a4_tail2_hier.zip
```

Each `train.py <run>` writes `log.txt`, `metrics.json`, `scores.npz` (dev and test scores of all six
variants), `aggregator.pt` and the `tail2` and `hier` submission zips to `runs/<run>/`.

The other variants (`first`, `last`, `selected`, `whole`) are written from `scores.npz`, for example:

```
python -c "import sys, numpy as np; from predict import report, write_submission; run, v = sys.argv[1], sys.argv[2]; z = np.load('runs/' + run + '/scores.npz'); r = report(z['dev_' + v], z['dev_y']); write_submission(run + '__' + v + '.zip', list(z['test_ids']), z['test_' + v] > r['thr'])" e1_arch4_lorma first
```

## Codabench submissions covered by this code

Dev is macro-F1 on the 994 dev cases. "#" is the submission number. "variant" means the `python -c`
line above with that run and variant. "train" means `python train.py` with that run.

| # | Submission | Dev | Codabench | Produced by |
|---|---|---|---|---|
| 01 | Arch 1 E4, first | 0.5133 | 0.4752 | variant `e4_arch1_full first` |
| 02 | Arch 1 E4, whole | 0.5315 | 0.4894 | variant `e4_arch1_full whole` |
| 03 | Arch 1 E4, hier | 0.5483 | 0.5108 | train `e4_arch1_full` |
| 04 | Arch 3 E3, first | 0.5813 | 0.5618 | variant `e3_arch3_adapter first` |
| 05 | Arch 2 E2, first | 0.6143 | 0.5866 | variant `e2_arch2_unfreeze first` |
| 06 | Arch 3 E3, tail2 | 0.6196 | 0.6021 | train `e3_arch3_adapter` |
| 07 | Arch 4 E1, first | 0.6203 | 0.6044 | variant `e1_arch4_lorma first` |
| 08 | Arch 3 E3, hier | 0.6582 | 0.6154 | train `e3_arch3_adapter` |
| 09 | Arch 2 E2, selected | 0.7122 | 0.7168 | variant `e2_arch2_unfreeze selected` |
| 10 | Arch 4 E1, selected | 0.7409 | 0.7396 | variant `e1_arch4_lorma selected` |
| 11 | Arch 4 E6, selected | 0.7485 | 0.7498 | variant `e6_arch4_lorma_tail3_s1 selected` |
| 13 | Arch 2 E2, last | 0.7555 | 0.7436 | variant `e2_arch2_unfreeze last` |
| 14 | Arch 2 E2, whole | 0.7560 | 0.7473 | variant `e2_arch2_unfreeze whole` |
| 16 | Arch 2 E2, hier | 0.7686 | 0.7646 | train `e2_arch2_unfreeze` |
| 17 | Arch 4 E6, tail2 | 0.7744 | 0.7755 | train `e6_arch4_lorma_tail3_s1` |
| 18 | Arch 4 E1, tail2 | 0.7786 | 0.7751 | train `e1_arch4_lorma` |
| 19 | Arch 4 E1, last | 0.7816 | 0.7408 | variant `e1_arch4_lorma last` |
| 20 | Arch 4 E6, whole | 0.7875 | 0.7698 | variant `e6_arch4_lorma_tail3_s1 whole` |
| 21 | Arch 4 E1, whole | 0.7877 | 0.7665 | variant `e1_arch4_lorma whole` |
| 27 | Arch 4 E1, hier | 0.8038 | 0.7902 | train `e1_arch4_lorma` |
| 37 | Arch 4 E6, hier (best single model) | 0.8075 | 0.8048 | train `e6_arch4_lorma_tail3_s1` |
| 41 | Arch 2 student E5, whole | 0.7433 | 0.7502 | variant `e5_arch2_kd whole` |
| 42 | Arch 2 student E5, tail2 | 0.7461 | 0.7607 | train `e5_arch2_kd` |
| 43 | Arch 2 student E5, hier | 0.7515 | 0.7638 | train `e5_arch2_kd` |
| 44 | Arch 4 student E12, tail2 | 0.7645 | 0.7487 | train `e12_arch4_lorma_kd` |
| 45 | Arch 4 student E12, hier | 0.7837 | 0.7772 | train `e12_arch4_lorma_kd` |
| 46 | Arch 4 student E12, whole | 0.7865 | 0.7551 | train `e12_arch4_lorma_kd` (best variant on dev) |
| 48 | Arch 4 E11, hier | 0.7958 | 0.7896 | train `e11_arch4_lorma_s3` |
| 50 | Arch 4 E10, hier | 0.8099 | 0.7947 | train `e10_arch4_lorma_r32_e3_s2` |
| 53 | Ensemble E1 + E10 + E11, hier (best) | 0.8147 | 0.8154 | `python ensemble.py` |

The other submissions of the 80 are ensembles of other member lists and runs not set up here (E7, E13
to E20). None scored above #53.
