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
| #06 Arch 4 LoRMA+ (F1B) | 0.4085 |
| #22 Arch 2 student G4 | 0.4211 |
| #26 9 runs, logit average (leaderboard) | 0.4450 |

The other single-run submissions come from the same `train.py` runs. The other logit average
ensembles on the leaderboard come from `ensemble.py` with a different `MEMBERS` list.
