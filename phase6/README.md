# Phase 6: Legal summarization (IN-Abs)

## Task

The input is one Indian court judgment and the output is a short summary of it, like its headnote. There are 7,030 training judgments and 100 test judgments. The main metric is BERTScore F1 against the real headnote, and ROUGE-1 and ROUGE-L are also reported.

## My final system

- It is extractive: it picks whole sentences from the judgment and does not write new text.
- Each sentence gets a word score: how close it is to the main topic of its judgment (TF-IDF centrality), and how often its words also appear in training headnotes.
- Each sentence also gets a model score: the average of seven sentence scoring models built on my Phase 1 models.
- The final score is 0.48 times the word score plus 0.52 times the model score.
- I take the top sentences until the summary has 525 words. I skip sentences shorter than 6 words or longer than 90 words.
- Leaderboard score: 0.8527 BERTScore F1 (ROUGE-1 0.530, ROUGE-L 0.280).

Each sentence scoring model reads the judgment with a fine-tuned Phase 1 model and gives every sentence a probability of being in the summary. The training labels come from a greedy search that picks the sentences that best match the start of the real headnote in ROUGE.

## Models I tried

| Phase 1 model | Technique | Codabench |
|---|---|---|
| Arch 2 | LoRMA+ | 0.8422 |
| Arch 4 (bidirectional) | LoRA | 0.8426 |
| Arch 3 (encoder) | Pfeiffer adapter | 0.8394 |
| Arch 1 | Houlsby adapters | 0.8381 |
| Arch 3 | Pfeiffer adapter, distilled from Arch 2 | 0.8415 |
| Arch 2 | Abstractive summary (writes new text) | 0.8382 |

For distillation, each student model learns half from the training labels and half from a teacher of a different architecture. All six students did better on dev than the same model without a teacher. The best single model on dev was Arch 4 distilled from Arch 2 (0.8507).

The abstractive model often mixed up facts, so I kept the extractive system.

## Training setup

- Best single model: bidirectional Arch 4 with LoRA, distilled, learning rate 3e-4, 8 judgments per update, 2 epochs, dropout 0.1.
- Training labels are the sentences that best match the start of the real headnote (greedy ROUGE).
- Summaries take the highest scoring sentences up to 525 words and keep them in document order.
- All runs were on one Kaggle T4 GPU with fp16 mixed precision. Task runs use AdamW with weight decay 0.01, gradient clipping 1.0, a linear warmup then linear decay, and learning rate 1e-3 for the task head.

## Files

| File | What it does |
|---|---|
| `tokenizer.py` | My Phase 1 BPE tokenizer |
| `data.py` | Splits judgments into sentences, makes training labels and the dev set |
| `oracle.py` | ROUGE scores and the greedy label search |
| `model.py` | Sentence scoring model on top of a Phase 1 model |
| `peft_methods.py` | LoRMA+, LoRA, Pfeiffer and Houlsby |
| `train.py` | Trains one sentence scoring model (S1 to S5) |
| `distill.py` | Trains the six distilled students (k1 to k6) |
| `centrality.py` | The word score |
| `blend.py` | Mixes the scores and picks the sentences |
| `predict.py` | Builds the final submission from seven saved runs |
| `generate.py` | The abstractive Arch 2 model |
| `experiments/` | code for my other submissions, listed under Other submissions below |

## How to run

Set these paths at the top of the files:

- `data.py`: `TRAIN_JSON`, `TEST_JSON`, `TOKENIZER_PATH`, `OUT_DIR`
- `model.py`: `MODEL_DIRS`
- `train.py` and `distill.py`: `OUT_DIR` or `RUNS_DIR` (default `work/runs`)
- `predict.py`: `RUNS_DIR`, `OUT_ZIP`
- `generate.py`: `ARCH2_DIR`, `SELECTOR`, `OUT_DIR`

Each model folder in `MODEL_DIRS` holds `model.pt`, `metrics.json` and the matching model file from `../phase1` saved as `model_arch.py`. The Arch 4 folder holds the bidirectional Arch 4 from Phase 2.

Then run these in order:

```
python data.py
python train.py s1_arch2_lorma
python train.py s2_arch4_lora
python train.py s3_arch3_pfeiffer
python train.py s4_arch1_houlsby
python train.py s5_arch2_lorma_s1
python distill.py k1_arch3_pfeiffer_kd
python distill.py k3_arch2_lorma_kd_a4
python distill.py k4_arch4_lora_kd_a2
python distill.py k5_arch3_pfeiffer_kd_a4
python predict.py
```

The students need S1 and S2 to be trained first. `predict.py` prints the dev BERTScore and writes `work/submission_final.zip`. The students `k2_arch1_houlsby_kd` and `k6_arch1_houlsby_kd_a4` are optional because they are not in the final mix. `python generate.py` trains the abstractive model after S1.

Dependencies: Python 3, torch, numpy, scikit-learn, peft, bert_score. Training needs a GPU (I used one Kaggle T4 per run).

## Submissions this code reproduces

| Submission | Codabench |
|---|---|
| 02: Arch 2 + LoRMA+ (S1) | 0.8422 |
| 08: Arch 4 + LoRA (S2) | 0.8426 |
| 03: Arch 3 + Pfeiffer (S3) | 0.8394 |
| 04: Arch 1 + Houlsby (S4) | 0.8381 |
| 22: Arch 3 + Pfeiffer distilled from S1 (k1) | 0.8415 |
| 17: Abstractive Arch 2 | 0.8382 |
| 68: centrality and seven models, no headnote word score | 0.8525 |
| 82: final system | 0.8527 (leaderboard) |

## Other submissions

I made 81 scored Phase 6 submissions. The table above covers the main ones and this table covers the rest. Every command runs from `code/phase6` and writes `work/experiments/submission_<number>.zip`. The blends read the saved `scores.npz` of their models, so I first train S1 to S5 with `train.py`, k1 to k6 with `distill.py`, and P2 (Arch 4 + LoRA on labels from the whole headnote, with no sentence position features) with `python -m experiments.whole_headnote_run`. In the table, "local" centrality uses the judgment's own IDF, "global" uses IDF from all training sentences, and w is the weight on the model average.

| Submission | Codabench | How to reproduce |
|---|---|---|
| #01 local centrality, 600 words (baseline) | 0.8487 | `python -m experiments.raw_centrality 01` |
| #13 and #14 local centrality, 500 and 700 words | 0.8478, 0.8488 | `python -m experiments.raw_centrality 13` (or `14`) |
| #15 global centrality, 600 words | 0.8496 | `python -m experiments.raw_centrality 15` |
| #16 local + global centrality, 600 words | 0.8503 | `python -m experiments.raw_centrality 16` |
| #05 and #06 S1 alone, 600 and 750 words | 0.8409, 0.8408 | `python -m experiments.blends 05` (or `06`) |
| #07 S1 with local centrality, w 0.5, 600 words | 0.8488 | `python -m experiments.blends 07` |
| #09 S1 with local centrality, w 0.8, 500 words | 0.8473 | `python -m experiments.blends 09` |
| #10 and #11 S2 with local centrality, w 0.8 at 500 words and w 0.6 at 600 words | 0.8468, 0.8480 | `python -m experiments.blends 10` (or `11`) |
| #12 S1 to S4 with local centrality, w 0.8, 500 words | 0.8480 | `python -m experiments.blends 12` |
| #18 to #21 S1 + S2 with global or local + global centrality, w 0 to 0.3, 600 words | 0.8490, 0.8483, 0.8488, 0.8500 | `python -m experiments.blends 18` (or `19`, `20`, `21`) |
| #23 to #27 S1 + S2 + k1, w 0.3 to 0.5, 600 words, with and without the header skip | 0.8517, 0.8516, 0.8506, 0.8496, 0.8505 | `python -m experiments.blends 23` (to `27`) |
| #28 to #30 five models (S1, S2, S5, k1, k2), global, w 0.5 to 0.7, 600 words | 0.8517, 0.8504, 0.8480 | `python -m experiments.blends 28` (to `30`) |
| #31 to #33 the five models + P2, w 0.5 to 0.7, 600 words | 0.8510, 0.8514, 0.8480 | `python -m experiments.blends 31` (to `33`) |
| #34 to #36 the five models, model average rescaled to 0..1, w 0.4 and 0.5, 500 and 600 words | 0.8511, 0.8516, 0.8496 | `python -m experiments.blends 34` (to `36`) |
| #37 to #41 the five models, w 0.5 and 0.55, 550 to 650 words | 0.8513, 0.8518, 0.8510, 0.8505, 0.8517 | `python -m experiments.blends 37` (to `41`) |
| #42 to #44, #48, #51, #53, #54, #57 nine models (S1, S2, S5, k1 to k6), w 0.45 to 0.6, 550 and 600 words | 0.8514, 0.8514, 0.8515, 0.8516, 0.8516, 0.8520, 0.8514, 0.8512 | `python -m experiments.blends <number>` |
| #45 to #47, #49, #50, #52, #55, #56 seven models (the final seven), w 0.45 to 0.6, 550 and 600 words | 0.8519, 0.8515, 0.8510, 0.8519, 0.8518, 0.8522, 0.8513, 0.8515 | `python -m experiments.blends <number>` |
| #58 to #63 seven models, w 0.5, redundancy penalty 0.1 to 0.3, 550 and 600 words | 0.8517, 0.8511, 0.8512, 0.8514, 0.8517, 0.8516 | `python -m experiments.blends 58` (to `63`) |
| #64, #69, #70, #72, #76 to #78 seven models, global or local + global, w 0.48 and 0.52, 525 and 575 words | 0.8522, 0.8504, 0.8517, 0.8503, 0.8519, 0.8504, 0.8499 | `python -m experiments.blends <number>` |
| #65 to #67, #71, #73 to #75, #79 nine models, global or local + global, w 0.48 and 0.52, 525 and 575 words | 0.8515, 0.8505, 0.8499, 0.8516, 0.8519, 0.8502, 0.8500, 0.8517 | `python -m experiments.blends <number>` |
| #81 seven models, w 0.55, 525 words | 0.8522 | `python -m experiments.blends 81` |

From #23 on, the case header is never picked, except in #24 and #27. The exact settings of every submission are in `RECIPES` at the top of `experiments/blends.py` and `experiments/raw_centrality.py`. `python -m experiments.blends all` writes all the blends at once.
