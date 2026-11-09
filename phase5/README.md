# Phase 5: Legal Statute Identification (LSI)

## Task

The input is the facts of an Indian court case, with names and statute citations masked. The output is
the set of Indian Penal Code sections that apply, out of 100 sections. The metric is macro-F1 over the
100 sections.

## My final system

- I replace the mask tags with plain words and tokenize with my Phase 1 BPE tokenizer.
- Each model reads up to 4 chunks of 1024 tokens per case and scores all 100 sections. In most runs the head is my
  Assignment 2 attention, with one query per section.
- I average the scores of 12 fine-tuned runs (Arch 4 and Arch 2, including the distilled students).
- For each case I find the 50 most similar train and dev cases with TF-IDF and mix their sections into the
  scores.
- Each section gets its own threshold, tuned on dev, and every case gets at least one section.
- My report uses leaderboard entry #76 (0.4727, rank 2). #75 has the same score and is made by this
  code. My highest score is #93 (0.4732), which is what the code makes by default.

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
than their teacher.

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
| `knn_votes.py` | finds the similar cases |
| `ensemble.py` | builds the best submission |

## How to run

Run every command from this folder. Set these paths at the top of the files first:

- `data.py`: `DATA_DIR = "data/lsi"`, `TOKEN_DIR = "data/lsi_tokens"`, `TOKENIZER_PATH = "tokenizer/legal_bpe_16384.json"`
- `train.py`: `MODEL_DIRS`, `TEACHER_PATH`, `OUT_DIR = "runs"`
- `knn_votes.py`: `TOKEN_DIR`, `OUT_FILE = "runs/knn_bpe.npz"`
- `ensemble.py`: `RUNS_DIR`, `KNN_FILE`, `OUT_PATH`

Each model folder needs `model.pt` and its `../phase1` model file saved as `model_arch.py`. Arch 3 and
Arch 4 also need the Phase 1 `metrics.json`, and Arch 4 needs `arch4_engram_compress.pt`.

```
python data.py
python train.py f1_arch4_lorma_attn_asl
python train.py <run name>
python knn_votes.py
python ensemble.py
```

F1 runs first because it is the teacher. Then run `train.py` for every other run in `RUNS` in
`train.py`. Each run saves its scores and a submission zip in `runs/<run name>/`. The last command
writes `submissions/ens12_knn_bpe_k50_p8_a0.2_g0.8.zip`.

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
| #06 Arch 4 LoRMA+ (F1B, seed 1) | 0.4085 |
| #22 Arch 2 student G4 | 0.4211 |
| #75 12 runs with 20 similar cases | 0.4727 |
| #93 12 runs with 50 similar cases (best) | 0.4732 |

The other single-run submissions (#04 to #38) come from the same `train.py` runs. The other blends with
similar cases (#63 to #99) come from `ensemble.py` with different values of `K_NEIGHBOURS`, `POWER`,
`BLEND`, `GATE` and `BLEND_GATE`.

## Notes

- #76 is #75 with one more similarity index that is not in this code. It has the same score.
- 28% of test cases also appear in the Phase 1 pretraining text with the section numbers visible. I did
  not use that text.
