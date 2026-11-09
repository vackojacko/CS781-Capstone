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
