# Phase 4: Court Judgment Prediction (CJPE)

## Task

The input is the full text of an Indian Supreme Court judgment with the final decision removed. The model
predicts 1 if the appeal was accepted and 0 if it was rejected. The metric is macro-F1 over the 1,517 test
cases.

## My final system

- I split each case into chunks of 1024 tokens, counted back from the end, because the decision is
  usually discussed near the end.
- I fine-tune a Phase 1 model to label single chunks, using one of the last 2 chunks of each training case.
- I then freeze the model and train a small BiLSTM (my Assignment 2 LSTM) over the last 8 chunks of a case.
- Train is 59/41 but dev and test are balanced, so I pick the decision threshold that gives the best dev
  macro-F1.
- My best submission averages three Arch 4 LoRMA+ runs (E1, E10, E11). It scored 0.8154 on Codabench.

## Models I tried

All four Phase 1 models are in `../phase1`. For Arch 3 I use only the encoder. Arch 4 is the
bidirectional version from Phase 2. The score is the best single run for each model.

| Phase 1 model | Technique | Codabench |
|---|---|---|
| Arch 1 | full fine-tuning | 0.5108 |
| Arch 3 | bottleneck adapters | 0.6154 |
| Arch 2 | gradual unfreezing | 0.7646 |
| Arch 4 | LoRMA+ | 0.8048 |
| Arch 2 student (E5) | distillation from E1, E6 and E2 | 0.7638 |
| Arch 4 student (E12) | distillation from E1, E6 and E2 | 0.7772 |

The Arch 2 student keeps 95% of the teacher score (0.8048) with 18% of the parameters.

## Files

| File | What it does |
|---|---|
| `data.py` | loads and cleans the data and saves the tokens |
| `tokenizer.py` | my Phase 1 BPE tokenizer |
| `backbone.py` | loads a Phase 1 model without its LM head |
| `model.py` | chunk classifier and BiLSTM |
| `peft_methods.py` | adapters, LoRMA+ and gradual unfreezing |
| `distill.py` | distillation loss and teacher files |
| `predict.py` | threshold search and submission zip |
| `train.py` | run settings and one full run |
| `ensemble.py` | builds the best submission |

## How to run

Run every command from this folder. Set these paths at the top of the files first:

- `data.py`: `DATA_DIR = "data/cjpe"`, `BPE_PATH = "data/legal_bpe_16384.json"`, `TOKEN_DIR = "tokens"`
- `train.py`: `OUT_DIR = "runs"` and `MODEL_DIRS`, one folder per model
- `distill.py`: `TEACHER_FILES`, written by E1, E6 and E2
- `ensemble.py`: `RUNS_DIR`, `OUT_PATH`, `MEMBERS`, `VARIANT`

Each model folder needs `model.pt` and its `../phase1` model file saved as `model_arch.py`. Arch 3 and
Arch 4 also need the Phase 1 `metrics.json`.

```
python data.py
python train.py e1_arch4_lorma
python train.py e6_arch4_lorma_tail3_s1
python train.py e2_arch2_unfreeze
python train.py e3_arch3_adapter
python train.py e4_arch1_full
python train.py e10_arch4_lorma_r32_e3_s2
python train.py e11_arch4_lorma_s3
python train.py e5_arch2_kd
python train.py e12_arch4_lorma_kd
python ensemble.py
```

E1, E6 and E2 must run before E5 and E12, because they save the teacher outputs. Each run saves its
scores and its `tail2` and `hier` submission zips in `runs/<run>/`. The other inputs (`first`, `last`,
`selected`, `whole`) can be written from the saved scores, for example:

```
python -c "import sys, numpy as np; from predict import report, write_submission; run, v = sys.argv[1], sys.argv[2]; z = np.load('runs/' + run + '/scores.npz'); r = report(z['dev_' + v], z['dev_y']); write_submission(run + '__' + v + '.zip', list(z['test_ids']), z['test_' + v] > r['thr'])" e1_arch4_lorma first
```

Dependencies: Python 3, PyTorch, NumPy, scikit-learn, and a GPU for training (I used one Kaggle T4 per run).

## Submissions this code reproduces

| Submission | Codabench |
|---|---|
| #03 Arch 1 E4, hier | 0.5108 |
| #08 Arch 3 E3, hier | 0.6154 |
| #16 Arch 2 E2, hier | 0.7646 |
| #27 Arch 4 E1, hier | 0.7902 |
| #37 Arch 4 E6, hier | 0.8048 |
| #43 Arch 2 student E5, hier | 0.7638 |
| #45 Arch 4 student E12, hier | 0.7772 |
| #48 Arch 4 E11, hier | 0.7896 |
| #50 Arch 4 E10, hier | 0.7947 |
| #53 E1 + E10 + E11 (best) | 0.8154 |

The `first`, `last`, `selected`, `whole` and `tail2` submissions of these runs (#01 to #46) also come from
this code.

## Notes

My other submissions are ensembles of other runs or runs not kept here (E7, E13 to E20). None scored
above #53.
