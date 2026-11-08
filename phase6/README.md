# Phase 6: Legal summarization (IN-Abs)

## Task and metric

- Input: one Indian Supreme Court judgment. Output: one summary (headnote) per judgment.
- Data: 7,030 training judgments with reference headnotes, 100 test judgments without them.
- Submission: a zip with `prediction.json` at the root, a list of `{"id": ..., "summary": ...}`, one entry per test id.
- Metric: BERTScore F1 (main), ROUGE-1 F1 and ROUGE-L F1 (secondary). The scorer calls `bert_score.score(pred, gold, lang="en")`, which uses roberta-large.
- BERTScore cuts both texts to their first 510 roberta tokens. 78% of the headnotes are longer than that, so only about the first 400 words of each text are scored. So my summaries are about 525 words and try to match how a headnote starts.

## Data and preprocessing (`data.py`)

- Dev set: 300 training judgments held out with `np.random.RandomState(0)`. The same 300 are used for every dev number of this phase. The other 6,730 are used for training.
- Sentence splitting: a rule-based splitter. It splits on `.`, `?` or `!` followed by a capital letter, an opening bracket or a quote, but not after abbreviations such as "No.", "Mr." or "Ltd." and not after a word of 2 characters or less with its full stop ("v.", "S."). Text pieces of 3 characters or less are dropped.
- Tokenizing: every sentence is encoded with my own Phase 1 BPE (16,384 tokens) using the piece tokenizer from Phase 2 (letters, digits and punctuation are split before BPE). At most 128 tokens are kept per sentence. An empty sentence gets the separator token.
- Oracle labels (`oracle.py`): the standard greedy oracle (Nallapati et al., 2017). Starting from no sentences, it adds the sentence that most raises ROUGE-1 F1 + ROUGE-2 F1 against the first 420 words of the headnote, and stops when no sentence helps or 500 words are chosen. Chosen sentences get label 1, all others 0.
- Case header: a judgment starts with the case number, the parties and the advocates, up to "... delivered by ...". `header_end` finds that sentence (searching the first 40 sentences), and those sentences are never selected.
- Facts used in the design (training set): judgment median 3,128 words, headnote median 638 words (ratio 0.22). Oracle sentences sit at relative positions 0.19, 0.38 and 0.65 (quartiles), and 98.1% of them are inside the first 200 sentences, so training reads only the first 200 sentences of each judgment.

## The four backbones

The Phase 1 model code lives in `../phase1` (`arch1_model.py` to `arch4_model.py`), and the full architecture table is in `../phase1/README.md`. All four were trained from scratch with my own BPE, context 1024, and have 97.6M (Arch 1), 61.6M (Arch 2), 86.1M (Arch 3) and 139.1M (Arch 4) parameters.

- Arch 1: my Assignment 3 decoder scaled up, with fixed sinusoidal positions.
- Arch 2: grouped-query attention with 10 query heads and 2 key/value heads, which removes 80% of the key and value parameters.
- Arch 3: 704 encoder and 320 decoder positions, trained with continuation and T5 span corruption. Phase 6 uses only its 10-layer encoder. A continuation token is put in front of every window, so its windows hold 703 sentence tokens.
- Arch 4: multi-head latent attention with a sigmoid output gate and a 42.9M hashed bigram/trigram memory at layers 2 and 8. Phase 6 uses the bidirectional Arch 4 from Phase 2 (causal mask removed, then 800 updates of masked next-token training).

## Approach

### Sentence salience model (`model.py`)

1. Sentences are packed whole into windows of up to 1024 tokens (703 for Arch 3). Each window goes through the frozen Phase 1 backbone (with its fine-tuning technique). A sentence vector is the mean of its token states.
2. Dropout 0.2, then two position features (relative position and its square) are added to each sentence vector.
3. A bidirectional LSTM (my Assignment 2 `LSTMCell` and `RecurrentEncoder`, hidden size 256) reads the sentences of the judgment in order.
4. A linear layer gives one logit per sentence. The probability is the sigmoid of the logit.

To fit a 16 GB T4, only 3 randomly chosen windows per judgment keep gradients during training. The other windows run without gradients, so every sentence still gets a vector. Backbones run in fp16, except Arch 1 (fp32).

### Fine-tuning technique per Phase 1 model (`peft_methods.py`)

The backbone is frozen. Only the technique's parameters and the BiLSTM head (1.84M, 2.11M for Arch 1) are trained.

| Phase 1 model | Technique | Setting | Trainable |
|---|---|---|---|
| Arch 2 | LoRMA+ (my Phase 2 layer): `W' = (I + (alpha/r) B A) W0`, A starts at zero | r 16, alpha 16, dropout 0.05, every linear layer of every block | 3.99M |
| Arch 4 (bidirectional) | LoRA (peft library): `W' = W0 + (alpha/r) B A` | r 16, alpha 16, dropout 0.05, attention and FFN projections | 5.46M |
| Arch 3 (encoder) | Pfeiffer adapter: `y + up(gelu(down(y)))` after every FFN, `up` starts at zero | bottleneck 64 | 2.67M |
| Arch 1 | Houlsby adapters: the same bottleneck after attention and after the FFN | bottleneck 64 | 4.49M |

### Training (`train.py`)

- Loss: binary cross-entropy against the oracle labels, with `pos_weight = 4` (about 10 positive sentences among about 93).
- AdamW, weight decay 0.01. Learning rate 3e-4 for LoRMA and LoRA, 1e-3 for the adapters, 1e-3 for the head.
- One judgment per forward pass, gradients accumulated over 8 judgments, clipped to norm 1.0.
- 50 warmup steps, then linear decay to 5% of the learning rate. 2 epochs, at most 330 minutes.
- Backbone dropout 0.1. After each epoch, a fast dev check (ROUGE-1 F1 of the top 600 words against the first 420 headnote words) picks the best epoch.
- After training, every dev and test sentence is scored and saved in `scores.npz`. The summary length for the run's own submission (450, 600 or 750 words) is chosen by real BERTScore on dev.

### Distillation (`distill.py`)

A student's loss is `0.5 x BCE(student, oracle labels) + 0.5 x BCE(student, teacher probabilities)`. The teachers are S1 (Arch 2 + LoRMA) and S2 (Arch 4 + LoRA), which also score every training sentence. Every student learns from a teacher of another architecture.

| Student | Teacher | Dev without KD | Dev with KD |
|---|---|---|---|
| k1: Arch 3 + Pfeiffer | Arch 2 (S1) | 0.8459 | 0.8482 |
| k2: Arch 1 + Houlsby | Arch 2 (S1) | 0.8430 | 0.8480 |
| k3: Arch 2 + LoRMA, seed 2 | Arch 4 (S2) | 0.8496 | 0.8497 |
| k4: Arch 4 + LoRA, seed 2 | Arch 2 (S1) | 0.8497 | 0.8507 (best single model) |
| k5: Arch 3 + Pfeiffer, seed 1 | Arch 4 (S2) | 0.8459 | 0.8484 |
| k6: Arch 1 + Houlsby, seed 1 | Arch 4 (S2) | 0.8430 | 0.8454 |

### Final ensemble and selection (`centrality.py`, `blend.py`, `predict.py`)

The score of a sentence is

    0.48 x (0.35 x centrality + 0.65 x propensity) + 0.52 x mean probability of seven salience models

- Centrality: TF-IDF fitted on all training sentences (min_df 2, sublinear tf). A sentence's centrality is the dot product of its TF-IDF vector with the mean vector of its judgment, scaled to 0 to 1 within the judgment.
- Headnote propensity of a word: of the training judgments that contain the word, the share whose headnote also contains it, smoothed toward the average with 10 pseudo-judgments. A sentence's propensity is the IDF-weighted mean over its words, scaled to 0 to 1 within the judgment.
- The seven models: Arch 2 + LoRMA (seeds 0 and 1), Arch 4 + LoRA, and the students k1, k3, k4 and k5. The two Arch 1 runs are left out.
- Header sentences get score -1.
- Selection: sentences are taken from the highest score down, skipping sentences under 6 or over 90 words, until 525 words are reached. They are printed in document order.

### Abstractive variant (`generate.py`)

Arch 2 with its language-model head and LoRMA (2 epochs, lr 3e-4, batch 4) learns to write the first 400 tokens of the headnote from the oracle sentences (up to 600 tokens), with the loss only on the summary tokens. At test time it reads the top sentences of S1 and decodes greedily, never repeating a 3-token sequence. It is scored on the first 100 dev judgments. It wrote fluent text but mixed up facts, so it scored below extraction (0.8420 against 0.8491 for extraction from the same S1 scores).

## Files

| File | What it does |
|---|---|
| `tokenizer.py` | My Phase 1 BPE tokenizer (from Assignment 1), load, encode and decode only, plus the Phase 2 piece tokenizer. |
| `data.py` | Sentence splitter, header detection, tokenizing, oracle labels, the 300-judgment dev split; writes `work/p6_{train,dev,test}.pkl.gz`. |
| `oracle.py` | ROUGE-N F1 from n-gram counts and the greedy oracle labels. |
| `model.py` | Loads a Phase 1 backbone, sentence windows, the Assignment 2 BiLSTM, the salience model. |
| `peft_methods.py` | LoRMA+, LoRA, Pfeiffer and Houlsby adapters on the frozen backbone. |
| `train.py` | Trains one salience model (S1 to S5), saves `scores.npz` and a submission zip. |
| `distill.py` | The six distilled students (k1 to k6) and loading of the teacher probabilities. |
| `centrality.py` | Global-IDF TF-IDF centrality and headnote propensity. |
| `blend.py` | Final sentence score and sentence selection. |
| `predict.py` | Builds the final submission (#82) from the seven saved `scores.npz` files. |
| `generate.py` | The abstractive Arch 2 variant (g1). |

## Running the code

Dependencies: Python 3, `torch`, `numpy`, `scikit-learn`, `peft`, `bert_score` (downloads roberta-large). Training needs a CUDA GPU (I used one Kaggle T4 per run). `data.py`, `predict.py` and the blend run on CPU apart from BERTScore.

Paths (relative to this folder, set at the top of each file):

- `data.py`: `TRAIN_JSON = data/Phase_6_Summ/Train/train.json`, `TEST_JSON = data/Phase_6_Summ/Test/test.json`, `TOKENIZER_PATH = data/legal_bpe_16384.json` (my Phase 1 tokenizer), `OUT_DIR = work`.
- `model.py`: `MODEL_DIRS`. Each of `models/arch1`, `models/arch2`, `models/arch3` and `models/arch4_mntp` holds `model.pt`, its `metrics.json` and the model file from `../phase1` saved as `model_arch.py` (for example `../phase1/arch2_model.py` becomes `models/arch2/model_arch.py`). `models/arch4_mntp` holds the bidirectional Arch 4 checkpoint from Phase 2 and, if present, `arch4_engram_compress.pt`.
- `train.py`, `distill.py`, `predict.py`: `OUT_DIR` / `RUNS_DIR = work/runs`. `generate.py`: `ARCH2_DIR`, `SELECTOR`, `OUT_DIR`.

Order of commands:

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

Each training run writes `work/runs/<tag>/scores.npz` (sentence probabilities for dev and test, and for S1 and S2 also train) and `work/runs/<tag>/submission_<tag>.zip`. `predict.py` needs the seven `scores.npz` files of S1, S2, S5, k1, k3, k4 and k5. Given the same `scores.npz` files, `predict.py` gives the same zip every time; retraining on a GPU can change the scores slightly.

## Codabench submissions covered by this code

82 submissions were made in this phase and 81 returned a score. The others were weight, length and redundancy grids of the same blend and centrality-only baselines (see the report appendix). Dev is the 300 held-out judgments unless noted.

| # | Submission | Dev | Codabench | Command / config |
|---|---|---|---|---|
| 02 | Arch 2 + LoRMA (S1) | 0.8496 | 0.8422 | `python train.py s1_arch2_lorma` (450 words) |
| 08 | Arch 4 + LoRA (S2) | 0.8497 | 0.8426 | `python train.py s2_arch4_lora` (450 words) |
| 03 | Arch 3 + Pfeiffer (S3) | 0.8459 | 0.8394 | `python train.py s3_arch3_pfeiffer` (450 words) |
| 04 | Arch 1 + Houlsby (S4) | 0.8430 | 0.8381 | `python train.py s4_arch1_houlsby` (450 words) |
| 22 | Arch 3 + Pfeiffer, distilled from S1 (k1) | 0.8482 | 0.8415 | `python distill.py k1_arch3_pfeiffer_kd` |
| 17 | Abstractive Arch 2 (g1) | 0.8420 (first 100 dev) | 0.8382 | `python generate.py` |
| 68 | 0.48 centrality + 0.52 mean of 7 models, 525 words | 0.8539 | 0.8525 | `python predict.py` with `W_PROPENSITY = 0`, `MIN_WORDS = 0`, `MAX_WORDS = 100000` in `blend.py` |
| 82 | **Final: #68 + headnote propensity 0.65 + sentence-length filter (best)** | 0.8573 | **0.8527** | `python predict.py` (default constants) |

The final submission #82 scored BERTScore F1 0.8527, ROUGE-1 0.530 and ROUGE-L 0.280.
