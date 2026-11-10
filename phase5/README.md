# Phase 5: Legal Statute Identification (LSI)

## Task

The input is the facts of an Indian court case, with names and statute citations masked. The output is
the set of Indian Penal Code sections that apply, out of 100 sections. The metric is macro-F1 over the
100 sections.

## My final system

1. I replace the mask tags with plain words and tokenize with my Phase 1 BPE tokenizer.
2. Each model reads up to 4 chunks of 1024 tokens per case and scores all 100 sections. The head is my
   Assignment 2 attention, with one query per section (one Arch 4 run uses mean pooling instead).
3. I average the logits of 9 fine-tuned runs and apply a sigmoid. The 9 runs are:
   five Arch 4 runs, F1 (LoRMA+, ASL loss), F1B (LoRMA+, BCE loss), F1B seed 1, F1C (LoRMA+, mean pooling)
   and F3 (QLoRA); and four Arch 2 runs trained for 6 epochs, G1 (partial fine-tuning), G2 (partial
   fine-tuning, lr 1e-4), G5 (G2 with seed 1) and G4 (student distilled from F1).
4. Each section gets its own threshold, tuned on dev and pulled toward one global threshold when the
   section has few positives. Every case gets at least one section.

This ensemble scores 0.442 on dev (thresholds tuned on one half and scored on the other) and 0.4450 on
Codabench, which is my best Phase 5 score (submission #26).

## Models I tried

All four Phase 1 models are in `../phase1`. For Arch 3 I use only the encoder. Arch 4 is the
bidirectional version from Phase 2. The score is the best single run for each row.

| Phase 1 model | Technique | Codabench |
|---|---|---|
| Arch 1 | full fine-tuning | 0.0997 |
| Arch 3 | bottleneck adapters | 0.3161 |
| Arch 2 | full fine-tuning | 0.4031 |
| Arch 2 | partial fine-tuning (top 8 blocks) | 0.4077 |
| Arch 4 | LoRMA+ | 0.4085 |
| Arch 4 | QLoRA | no score |
| Arch 2 student (F7) | distillation from F1, 3 epochs | 0.4091 |
| Arch 2 student (G4) | distillation from F1, 6 epochs | 0.4211 |

G4 is my best single model. The teacher F1 (Arch 4 LoRMA+) scored 0.3942, so the students did better
than their teacher. Averaging logits worked a little better than averaging probabilities (0.4450 against
0.4432 for the same 9 runs). Adding the 6 epoch full fine-tuning run and the 8 chunk run lowered dev.

## Files

| File | What it does |
|---|---|
| `data.py` | loads and cleans the data, saves the tokens, makes the chunks |
| `tokenizer.py` | my Phase 1 BPE tokenizer |
| `model.py` | loads a Phase 1 model and adds the heads and losses |
| `peft_methods.py` | adapters, LoRMA+, QLoRA, partial and full fine-tuning |
| `distill.py` | distillation loss |
| `thresholds.py` | per-section thresholds |
| `train.py` | run settings and one full run |
| `predict.py` | submission zip from one run |
| `ensemble.py` | builds the best submission from the 9 runs |
| `experiments/` | code for my other submissions, listed under Other submissions below |

## How to run

Run every command from this folder. Set these paths at the top of the files first:

1. `data.py`: `DATA_DIR = "data/lsi"`, `TOKEN_DIR = "data/lsi_tokens"`, `TOKENIZER_PATH = "tokenizer/legal_bpe_16384.json"`
2. `train.py`: `MODEL_DIRS`, `TEACHER_PATH`, `OUT_DIR = "runs"`
3. `ensemble.py`: `RUNS_DIR`, `OUT_PATH`

Each model folder needs `model.pt` and its `../phase1` model file saved as `model_arch.py`. Arch 3 and
Arch 4 also need the Phase 1 `metrics.json`, and Arch 4 needs `arch4_engram_compress.pt`.

```
python data.py
python train.py f1_arch4_lorma_attn_asl
python train.py <run name>
python ensemble.py
```

F1 runs first because it is the teacher. Then run `train.py` for the other 8 runs in `MEMBERS` in
`ensemble.py` (the other runs in `RUNS` in `train.py` are the ones I tried and did not keep). Each run
saves its scores and a submission zip in `runs/<run name>/`. The last command writes
`submissions/ens9_a4x5_g4_logit.zip`.

To rebuild the zip of one run: `python predict.py runs/<run name>/scores.npz out.zip`

Dependencies: Python 3, PyTorch, NumPy, SciPy, scikit-learn, and a GPU for training (I used one Kaggle
T4 per run).

## Submissions this code reproduces

| Submission | Codabench |
|---|---|
| #01 Arch 1 full fine-tuning | 0.0997 |
| #02 Arch 3 adapters | 0.3161 |
| #03 Arch 4 LoRMA+ (F1) | 0.3942 |
| #05 Arch 2 full fine-tuning | 0.4031 |
| #06 Arch 4 LoRMA+, seed 1 (F1B seed 1) | 0.4085 |
| #22 Arch 2 student G4 | 0.4211 |
| #26 9 runs, logit average (leaderboard) | 0.4450 |

The other single-run submissions come from the same `train.py` runs. The other logit average
ensembles on the leaderboard come from `ensemble.py` with a different `MEMBERS` list.

## Other submissions

The table above covers my main Phase 5 submissions, and this table covers the rest. Every command runs from `code/phase5`. Every run is in `RUNS` in `train.py`, so I first train the members with `python train.py <run>`; each run saves `runs/<run>/scores.npz`, and the scripts in `experiments/` read those files and write the zips to `submissions/`.

| Submission | Codabench | How to reproduce |
|---|---|---|
| #04 Arch 4 QLoRA (F3) | no score | `python train.py f3_arch4_qlora_attn_bce`, then `python predict.py runs/f3_arch4_qlora_attn_bce/scores.npz out.zip` |
| #07 Arch 2 partial fine-tuning (F4) | 0.4075 | `python train.py f4_arch2_partial_attn_bce`, then `python predict.py runs/f4_arch2_partial_attn_bce/scores.npz out.zip` |
| #08 Arch 2 student (F7) | 0.4087 | `python train.py f7_arch2_kd_from_f1`, then `python predict.py runs/f7_arch2_kd_from_f1/scores.npz out.zip` |
| #29 to #38 single runs: G3, F1C, F4 seed 1, G1, G6, G5, F4 seed 2, G2, F1B, F7 seed 1 | 0.4008, 0.3968, 0.4017, 0.3993, 0.4014, 0.4050, 0.4055, 0.4077, 0.4062, 0.4091 | `python train.py <run>`, then `python predict.py runs/<run>/scores.npz out.zip` (runs `g3_arch2_full_e6`, `f1c_arch4_lorma_pool_bce`, `f4s1_arch2_partial_attn_bce_s1`, `g1_arch2_partial_e6`, `g6_arch2_partial_e6_k8`, `g5_arch2_partial_e6_lr1e4_s1`, `f4s2_arch2_partial_attn_bce_s2`, `g2_arch2_partial_e6_lr1e4`, `f1b_arch4_lorma_attn_bce`, `f7s1_arch2_kd_from_f1_s1`) |
| #09 4 Arch 2 runs, prob average | 0.4225 | `python -m experiments.ensembles 09_ens_a2x4` |
| #10 5 Arch 4 runs, prob average | 0.4212 | `python -m experiments.ensembles 10_ens_a4x5` |
| #11 QLoRA + Arch 2 partial, prob average | 0.4230 | `python -m experiments.ensembles 11_ens_a4qlora_a2` |
| #12 6 attention + BCE runs, prob average | 0.4302 | `python -m experiments.ensembles 12_ens_bce_attn6` |
| #13 9 runs + Arch 3, prob average | 0.4292 | `python -m experiments.ensembles 13_ens_all9_a3` |
| #14 4 best runs, prob average | 0.4305 | `python -m experiments.ensembles 14_ens_top4` |
| #15 9 runs, prob average | 0.4309 | `python -m experiments.ensembles 15_ens_all9` |
| #16 11 runs, prob average | 0.4286 | `python -m experiments.ensembles 16_ens_all11` |
| #17 9 runs, logit average | 0.4324 | `python -m experiments.ensembles 17_all9_logit` |
| #18 9 runs, Arch 4 and Arch 2 groups weighted equally | 0.4279 | `python -m experiments.ensembles 18_all9_groups` |
| #19 9 runs, threshold shrinkage 20 | 0.4308 | `python -m experiments.ensembles 19_all9_prob_shrink20` |
| #20 9 runs, no threshold shrinkage | 0.4302 | `python -m experiments.ensembles 20_all9_prob_shrink0` |
| #21 11 runs, groups weighted equally | 0.4299 | `python -m experiments.ensembles 21_all11_groups` |
| #23 13 runs, prob average | 0.4395 | `python -m experiments.ensembles 23_all13_prob` |
| #24 13 runs, logit average | 0.4400 | `python -m experiments.ensembles 24_all13_logit` |
| #25 members of #26, prob average | 0.4432 | `python -m experiments.ensembles 25_a4x5_g_prob` |
| #27 Arch 4 and Arch 2 group weight tuned on dev | 0.4386 | `python -m experiments.group_weights` |
| #28 greedy ensemble selection | 0.4432 | `python -m experiments.greedy` |
| #39 3 students, logit average | 0.4183 | `python -m experiments.ensembles 39_ens_kd3_logit` |
| #40 5 Arch 4 runs, logit average | 0.4216 | `python -m experiments.ensembles 40_ens_a4x5_logit` |
| #41 4 LoRMA+ runs, logit average | 0.4202 | `python -m experiments.ensembles 41_ens_a4_lorma_only_logit` |
| #42 6 three-epoch Arch 2 runs, logit average | 0.4236 | `python -m experiments.ensembles 42_ens_f2x6_logit` |
| #43 5 Arch 4 runs + 3 students, prob average | 0.4263 | `python -m experiments.ensembles 43_ens_a4_kd3_prob` |
| #44 5 Arch 4 runs + 3 students, logit average | 0.4264 | `python -m experiments.ensembles 44_ens_a4_kd3_logit` |
| #45 4 six-epoch Arch 2 runs, logit average | 0.4362 | `python -m experiments.ensembles 45_ens_g4x_logit` |
| #46 6 six-epoch Arch 2 runs, logit average | 0.4343 | `python -m experiments.ensembles 46_ens_g6x_logit` |
| #47 12 Arch 2 runs, logit average | 0.4345 | `python -m experiments.ensembles 47_ens_a2_all12_logit` |
| #48 5 Arch 4 runs + G4 + G5, logit average | 0.4390 | `python -m experiments.ensembles 48_ens_a4_kd_g5_logit` |
| #49 all 17 runs, logit average | 0.4388 | `python -m experiments.ensembles 49_ens_a4_all_logit` |
| #50 members of #26 + F7, prob average | 0.4435 | `python -m experiments.ensembles 50_ens_a4_g4_kd_prob` |
| #51 5 Arch 4 runs + 6 six-epoch Arch 2 runs, prob average | 0.4423 | `python -m experiments.ensembles 51_ens_a4_g6_prob` |
| #52 members of #26 + F7, logit average | 0.4442 | `python -m experiments.ensembles 52_ens_a4_g4_kd_logit` |
| #53 5 Arch 4 runs + 6 six-epoch Arch 2 runs, logit average | 0.4415 | `python -m experiments.ensembles 53_ens_a4_g6_logit` |
| #54 members of #26 without F1 | 0.4449 | `python -m experiments.ensembles 54_ens_a4_g4_minus_asl_logit` |
| #55 members of #26 with G4 counted twice | 0.4450 | `python -m experiments.ensembles 55_ens_a4_g4_double_kd_logit` |
| #56 members of #26 without F1C | 0.4451 | `python -m experiments.ensembles 56_ens_a4_g4_minus_pool_logit` |
| #57 members of #26 + F7 + F7 seed 1 | 0.4431 | `python -m experiments.ensembles 57_ens_a4_g4_f7s_logit` |
| #58 members of #26 without F3 | 0.4449 | `python -m experiments.ensembles 58_ens_a4_g4_minus_qlora_logit` |
| #59 members of #26 without F1 and F1C | 0.4446 | `python -m experiments.ensembles 59_ens_attn_bce_a4_g4_logit` |

`python -m experiments.ensembles all` writes every ensemble in this table at once.
