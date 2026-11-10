# Phase 4: Court Judgment Prediction (CJPE)

## Task

The input is the full text of an Indian Supreme Court judgment with the final decision removed. The model
predicts 1 if the appeal was accepted and 0 if it was rejected. The metric is macro-F1 over the 1,517 test
cases.

## My final system

- I split each case into chunks of 1024 tokens, counted back from the end, because the decision is
  usually discussed near the end.
- I fine-tune a Phase 1 model to label single chunks, using one of the last 2 or 3 chunks of each training case.
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
| `experiments/` | code for my other submissions, listed under Other submissions below |

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
| #53 E1 + E10 + E11 (leaderboard) | 0.8154 |

The `first`, `last`, `selected`, `whole` and `tail2` submissions of these runs (#01 to #46) also come from
this code.

## Other submissions

I made 79 scored Phase 4 submissions. The table above covers the main ones; this table covers the rest. Every command runs from `code/phase4`. Each run saves `runs/<run>/scores.npz`, and the scripts in `experiments/` read those files and write the zips to `submissions/`. The ensembles need the scores of all their members, so I first run the extra runs E7, E13, E14 and E15 to E20 with `python -m experiments.extra_runs <run>` (the run names are in `experiments/extra_runs.py`).

| Submission | Codabench | How to reproduce |
|---|---|---|
| #01 E4 first, #02 E4 whole | 0.4752, 0.4894 | `python train.py e4_arch1_full`, then `python -m experiments.single_variant e4_arch1_full first` (or `whole`) |
| #04 E3 first, #06 E3 tail2 | 0.5618, 0.6021 | `python train.py e3_arch3_adapter` (it also writes the tail2 zip), then `python -m experiments.single_variant e3_arch3_adapter first` |
| #05 E2 first, #09 E2 selected, #13 E2 last, #14 E2 whole | 0.5866, 0.7168, 0.7436, 0.7473 | `python train.py e2_arch2_unfreeze`, then `python -m experiments.single_variant e2_arch2_unfreeze <input>` |
| #07 E1 first, #10 E1 selected, #18 E1 tail2, #19 E1 last, #21 E1 whole | 0.6044, 0.7396, 0.7751, 0.7408, 0.7665 | `python train.py e1_arch4_lorma`, then `python -m experiments.single_variant e1_arch4_lorma <input>` |
| #11 E6 selected, #17 E6 tail2, #20 E6 whole | 0.7498, 0.7755, 0.7698 | `python train.py e6_arch4_lorma_tail3_s1`, then `python -m experiments.single_variant e6_arch4_lorma_tail3_s1 <input>` |
| #12 E7 tail2, #15 E7 hier (Arch 2 on the last 3 chunks) | 0.7452, 0.7613 | `python -m experiments.extra_runs e7_arch2_unfreeze_tail3_s1` (it writes both zips) |
| #41 E5 whole, #42 E5 tail2 | 0.7502, 0.7607 | `python train.py e5_arch2_kd`, then `python -m experiments.single_variant e5_arch2_kd whole` (or `tail2`) |
| #44 E12 tail2, #46 E12 whole | 0.7487, 0.7551 | `python train.py e12_arch4_lorma_kd`, then `python -m experiments.single_variant e12_arch4_lorma_kd tail2` (or `whole`) |
| #47 E13 hier (Arch 4 on the last 3 chunks, seed 4) | 0.7791 | `python -m experiments.extra_runs e13_arch4_lorma_tail3_s4` |
| #49 E14 hier (Arch 4 on the last 3 chunks, seed 5) | 0.7712 | `python -m experiments.extra_runs e14_arch4_lorma_tail3_s5` |
| #22 E1 + E6, tail2 and hier | 0.7970 | `python -m experiments.ensembles r1_e1_e6_tail2_hier` |
| #23 E1 + E2 + E3 + E4 + E6, hier | 0.7619 | `python -m experiments.ensembles r1_e1_e2_e3_e4_e6_hier` |
| #24 E2 + E6 + E7, hier | 0.7972 | `python -m experiments.ensembles r1_e2_e6_e7_hier` |
| #25 E1 + E4 + E6, hier | 0.7691 | `python -m experiments.ensembles r1_e1_e4_e6_hier` |
| #26 E1 + E2 + E6, tail2 and hier | 0.8003 | `python -m experiments.ensembles r1_e1_e2_e6_tail2_hier` |
| #29 E1 + E2, hier | 0.7889 | `python -m experiments.ensembles r1_e1_e2_hier` |
| #30 E1 + E6 + E7, tail2 and hier | 0.8007 | `python -m experiments.ensembles r1_e1_e6_e7_tail2_hier` |
| #31 E1 + E2 + E6 + E7, tail2 and hier | 0.7961 | `python -m experiments.ensembles r1_e1_e2_e6_e7_tail2_hier` |
| #32 E6 + E7, hier | 0.8011 | `python -m experiments.ensembles r1_e6_e7_hier` |
| #33 E1 + E7, hier | 0.7934 | `python -m experiments.ensembles r1_e1_e7_hier` |
| #34 E2 + E6, tail2 and hier | 0.7933 | `python -m experiments.ensembles r1_e2_e6_tail2_hier` |
| #35 E2 + E6, hier | 0.7954 | `python -m experiments.ensembles r1_e2_e6_hier` |
| #36 E1 + E2 + E6 + E7, hier | 0.7963 | `python -m experiments.ensembles r1_e1_e2_e6_e7_hier` |
| #38 E1 + E6 + E7, hier | 0.8033 | `python -m experiments.ensembles r1_e1_e6_e7_hier` |
| #39 E1 + E6, hier | 0.8059 | `python -m experiments.ensembles r1_e1_e6_hier` |
| #40 E1 + E2 + E6, hier | 0.8048 | `python -m experiments.ensembles r1_e1_e2_e6_hier` |
| #51 E6 + E13 + E14 (Arch 4 on the last 3 chunks), hier | 0.8028 | `python -m experiments.ensembles r2_a4_tail3_hier` |
| #52 six Arch 4 runs (E1, E6, E10, E11, E13, E14) + E12, hier | 0.8105 | `python -m experiments.ensembles r2_a4x6_kd_hier` |
| #54 the six Arch 4 runs + E12 + E2, hier | 0.8027 | `python -m experiments.ensembles r2_a4x6_kd_a2_hier` |
| #55 the six Arch 4 runs + E2, hier | 0.8061 | `python -m experiments.ensembles r2_a4x6_a2_hier` |
| #56 the six Arch 4 runs, hier | 0.8101 | `python -m experiments.ensembles r2_a4x6_hier` |
| #57 the six Arch 4 runs, hier and whole | 0.8021 | `python -m experiments.ensembles r2_a4x6_hier_whole` |
| #58 E1 + E6 + E10 + E14, hier | 0.8021 | `python -m experiments.ensembles r2_a4_top4_hier` |
| #59 E1 + E6 + E10 + E14, hier and whole | 0.8049 | `python -m experiments.ensembles r2_a4_top4_hier_whole` |
| #60 all 12 Arch 4 runs, hier and tail2 | 0.8028 | `python -m experiments.ensembles r3_all_a4_hier_tail2` |
| #61 the 6 new seeds E15 to E20, hier | 0.8087 | `python -m experiments.ensembles r3_new_seeds_hier` |
| #62 all 12 Arch 4 runs + E2 + E7 + E5 + E12, hier | 0.8075 | `python -m experiments.ensembles r3_all_a4_a2x2_kd_hier` |
| #63 all 12 Arch 4 runs, hier, whole, tail2 and last | 0.7983 | `python -m experiments.ensembles r3_all_a4_allinputs` |
| #64 the 9 Arch 4 runs on the last 2 chunks, hier and tail2 | 0.8035 | `python -m experiments.ensembles r3_tail2_hier_tail2` |
| #65 all 12 Arch 4 runs, hier counted twice and tail2 once | 0.8100 | `python -m experiments.ensembles r3_all_a4_hier_x2_tail2` |
| #66 all 12 Arch 4 runs + E2, hier | 0.8059 | `python -m experiments.ensembles r3_all_a4_a2_hier` |
| #67 the 9 Arch 4 runs on the last 2 chunks, hier | 0.8074 | `python -m experiments.ensembles r3_tail2_hier` |
| #68 the 10 Arch 4 runs with the best dev hier score | 0.8065 | `python -m experiments.ensembles r3_top10_hier` |
| #69 all 12 Arch 4 runs, hier and whole | 0.8042 | `python -m experiments.ensembles r3_all_a4_hier_whole` |
| #70 all 12 Arch 4 runs, hier | 0.8074 | `python -m experiments.ensembles r3_all_a4_hier` |
| #71 the 8 Arch 4 runs with the best dev hier score | 0.8092 | `python -m experiments.ensembles r3_top8_hier` |
| #72 all 12 Arch 4 runs + E12, hier | 0.8059 | `python -m experiments.ensembles r3_all_a4_kd_hier` |
| #73 all 12 Arch 4 runs, hier and whole, median | 0.8015 | `python -m experiments.ensembles r3_all_a4_hier_whole_median` |
| #74 all 12 Arch 4 runs, hier, median | 0.8093 | `python -m experiments.ensembles r3_all_a4_hier_median` |
| #75 E15 to E20, hier and whole | 0.8088 | `python -m experiments.ensembles r3_new_seeds_hier_whole` |
| #76 the 9 Arch 4 runs on the last 2 chunks, hier and whole | 0.8088 | `python -m experiments.ensembles r3_tail2_hier_whole` |
| #77 the 9 Arch 4 runs on the last 2 chunks, hier, median | 0.8087 | `python -m experiments.ensembles r3_tail2_hier_median` |
| #78 the 6 Arch 4 runs with the best dev hier score | 0.8153 | `python -m experiments.ensembles r3_top6_hier` |
| #79 the 4 Arch 4 runs with the best dev hier score | 0.8088 | `python -m experiments.ensembles r3_top4_hier` |
| #80 the members of #53 and #78 together (8 Arch 4 runs), hier | 0.8082 | `python -m experiments.ensembles final_union_hier` |

`python -m experiments.ensembles all` writes every ensemble in this table at once.
