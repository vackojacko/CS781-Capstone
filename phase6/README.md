# Phase 6: Legal summarization (IN-Abs)

## Task

- Input: one judgment. Output: its headnote. Data: 7,030 training and 100 test judgments.
- Submission: a zip with `prediction.json`, a list of `{"id": ..., "summary": ...}`, one per test id.
- Metric: BERTScore F1 (`bert_score.score(pred, gold, lang="en")`, roberta-large), plus ROUGE-1 and ROUGE-L F1.

BERTScore only reads the first 510 roberta tokens (about 400 words), so my summaries try to match the start of a headnote.

My best submission scored 0.8527. It picks sentences using centrality, seven salience models and headnote propensity, up to 525 words.

## Data (`data.py`, `oracle.py`)

- Dev set: 300 training judgments held out with `np.random.RandomState(0)`. All dev numbers use them.
- Sentences are encoded with my Phase 1 BPE (16,384 tokens), at most 128 tokens each.
- Oracle labels: a greedy oracle adds the sentence that raises ROUGE-1 F1 + ROUGE-2 F1 most against the first 420 headnote words, up to 500 words.
- Case header sentences (found by `header_end`) are never picked.
- Training reads only the first 200 sentences, which hold 98.1% of oracle sentences.

## Salience model (`model.py`, `peft_methods.py`)

A frozen Phase 1 model reads windows of whole sentences (1024 tokens, 703 for Arch 3). Mean-pooled sentence vectors go through my Assignment 2 BiLSTM (hidden 256) and a linear layer, giving one probability per sentence. Only 3 random windows per judgment keep gradients, to fit a 16 GB T4. The head has 1.84M parameters (2.11M for Arch 1).

| Model | Technique | Setting | Trainable |
|---|---|---|---|
| Arch 2 | LoRMA+ (my Phase 2 layer) | r 16, alpha 16, dropout 0.05, every linear layer | 3.99M |
| Arch 4 (bidirectional) | LoRA (peft) | r 16, alpha 16, dropout 0.05, attention and FFN | 5.46M |
| Arch 3 (encoder) | Pfeiffer adapter | bottleneck 64 | 2.67M |
| Arch 1 | Houlsby adapters | bottleneck 64 | 4.49M |

## Training (`train.py`)

| Setting | Value |
|---|---|
| Loss | BCE against oracle labels, `pos_weight = 4` |
| Optimizer | AdamW, weight decay 0.01 |
| Learning rate | 3e-4 for LoRMA and LoRA, 1e-3 for adapters, 1e-3 for the head |
| Batch | 1 judgment, gradients accumulated over 8, clipped to norm 1.0 |
| Schedule | 50 warmup steps, linear decay to 5% |
| Epochs | 2, at most 330 minutes |
| Dropout | 0.1 backbone, 0.2 sentence vectors |

Each run's own summary length (450, 600 or 750 words) is picked by dev BERTScore.

## Distillation (`distill.py`)

Each student learns from a teacher (S1 or S2) of a different architecture.

```
loss = 0.5 x BCE(student, oracle labels) + 0.5 x BCE(student, teacher probabilities)
```

| Student | Teacher | Dev without KD | Dev with KD |
|---|---|---|---|
| k1: Arch 3 + Pfeiffer | Arch 2 (S1) | 0.8459 | 0.8482 |
| k2: Arch 1 + Houlsby | Arch 2 (S1) | 0.8430 | 0.8480 |
| k3: Arch 2 + LoRMA, seed 2 | Arch 4 (S2) | 0.8496 | 0.8497 |
| k4: Arch 4 + LoRA, seed 2 | Arch 2 (S1) | 0.8497 | 0.8507 (best single model) |
| k5: Arch 3 + Pfeiffer, seed 1 | Arch 4 (S2) | 0.8459 | 0.8484 |
| k6: Arch 1 + Houlsby, seed 1 | Arch 4 (S2) | 0.8430 | 0.8454 |

## Final score (`centrality.py`, `blend.py`, `predict.py`)

The final score mixes word statistics and the average of seven models almost equally:

```
score = 0.48 x (0.35 x centrality + 0.65 x propensity) + 0.52 x mean probability of seven models
```

- Centrality: closeness of a sentence's TF-IDF vector to its judgment's mean vector. Propensity: how often its words also appear in training headnotes.
- The seven models: S1, S2, S5, k1, k3, k4 and k5.
- Sentences are taken from the top score until 525 words, skipping ones under 6 or over 90 words.

The abstractive variant (`generate.py`) mixed up facts and scored 0.8420 on the first 100 dev judgments, against 0.8491 for extraction from the same S1 scores.

## Files

| File | What it does |
|---|---|
| `tokenizer.py` | My Phase 1 BPE (from Assignment 1) and the Phase 2 piece tokenizer |
| `data.py` | Splitting, tokenizing, oracle labels, dev split. Writes `work/p6_{train,dev,test}.pkl.gz` |
| `oracle.py` | ROUGE-N F1 and greedy oracle labels |
| `model.py` | The BiLSTM salience model on a Phase 1 model |
| `peft_methods.py` | LoRMA+, LoRA, Pfeiffer and Houlsby |
| `train.py` | Trains one salience model (S1 to S5), saves `scores.npz` and a zip |
| `distill.py` | The six students (k1 to k6) |
| `centrality.py` | TF-IDF centrality and headnote propensity |
| `blend.py` | Final score and sentence selection |
| `predict.py` | Builds the final submission (#82) from seven `scores.npz` files |
| `generate.py` | Abstractive Arch 2 variant (g1) |

## How to run

Dependencies: Python 3, `torch`, `numpy`, `scikit-learn`, `peft`, `bert_score`. Training needs a CUDA GPU (I used one Kaggle T4 per run).

| File | Path constants |
|---|---|
| `data.py` | `TRAIN_JSON = data/Phase_6_Summ/Train/train.json`, `TEST_JSON = data/Phase_6_Summ/Test/test.json`, `TOKENIZER_PATH = data/legal_bpe_16384.json`, `OUT_DIR = work` |
| `model.py` | `MODEL_DIRS` |
| `train.py`, `distill.py`, `predict.py` | `OUT_DIR` / `RUNS_DIR = work/runs` |
| `generate.py` | `ARCH2_DIR`, `SELECTOR`, `OUT_DIR` |

Each of `models/arch1`, `models/arch2`, `models/arch3` and `models/arch4_mntp` (bidirectional Arch 4 from Phase 2, plus `arch4_engram_compress.pt` if present) holds `model.pt`, `metrics.json` and the `../phase1` model file saved as `model_arch.py`.

```
python data.py                                  # work/p6_train.pkl.gz, p6_dev.pkl.gz, p6_test.pkl.gz
python train.py s1_arch2_lorma                  # teacher 1, also scores the training set
python train.py s2_arch4_lora                   # teacher 2, also scores the training set
python train.py s3_arch3_pfeiffer
python train.py s4_arch1_houlsby
python train.py s5_arch2_lorma_s1
python distill.py k1_arch3_pfeiffer_kd          # students need S1 and S2 first
python distill.py k3_arch2_lorma_kd_a4
python distill.py k4_arch4_lora_kd_a2
python distill.py k5_arch3_pfeiffer_kd_a4
python distill.py k2_arch1_houlsby_kd            # optional, not in the final ensemble
python distill.py k6_arch1_houlsby_kd_a4         # optional, not in the final ensemble
python predict.py                               # prints dev BERTScore, writes work/submission_final.zip
python generate.py                              # abstractive variant, after S1
```

Each run writes `work/runs/<tag>/scores.npz` and `work/runs/<tag>/submission_<tag>.zip`.

## Codabench submissions

I made 82 submissions and 81 got a score. The rest were grids of the same blend and baselines (see the report appendix).

| # | Submission | Dev | Codabench | Command / config |
|---|---|---|---|---|
| 02 | Arch 2 + LoRMA (S1) | 0.8496 | 0.8422 | `python train.py s1_arch2_lorma` (450 words) |
| 08 | Arch 4 + LoRA (S2) | 0.8497 | 0.8426 | `python train.py s2_arch4_lora` (450 words) |
| 03 | Arch 3 + Pfeiffer (S3) | 0.8459 | 0.8394 | `python train.py s3_arch3_pfeiffer` (450 words) |
| 04 | Arch 1 + Houlsby (S4) | 0.8430 | 0.8381 | `python train.py s4_arch1_houlsby` (450 words) |
| 22 | Arch 3 + Pfeiffer, distilled from S1 (k1) | 0.8482 | 0.8415 | `python distill.py k1_arch3_pfeiffer_kd` |
| 17 | Abstractive Arch 2 (g1) | 0.8420 (first 100 dev) | 0.8382 | `python generate.py` |
| 68 | 0.48 centrality + 0.52 mean of 7 models, 525 words | 0.8539 | 0.8525 | `python predict.py` with `W_PROPENSITY = 0`, `MIN_WORDS = 0`, `MAX_WORDS = 100000` in `blend.py` |
| 82 | Final (best): #68 + headnote propensity 0.65 + sentence length filter | 0.8573 | 0.8527 | `python predict.py` (default constants) |

The final submission #82 scored BERTScore F1 0.8527, ROUGE-1 0.530 and ROUGE-L 0.280.
