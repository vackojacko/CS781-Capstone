# Phase 7: Prior Case Retrieval (PCR)

## Task

- Data: IL-PCR. Queries are judgments with `<CITATION>`, `<NAME>` and `<ORG>` hidden. Every query is also in its own candidate pool.
- For every query I return exactly 7 candidate ids. Metric: micro-F1@7 over all queries. A perfect ranking reaches only 0.618 on dev.

```
P = hits / (7 x queries),  R = hits / relevant cases,  F1 = 2PR / (P + R)
```

Queries / pool size: train 827 / 4,320, dev 118 / 1,023, test 237 / 1,727. Median relevant per query: 4 (train), 5 (dev).

My best submission scored 0.4832 with no trained model: TF-IDF on 4 to 6 word phrases, citation contexts, co-citation, CSLS, a word view (weight 0.75) and neighbour votes.

## Data

- Files in `data/`: `train.json`, `train_candidates.json`, `dev.json`, `dev_candidates.json`, `test.json`, `test_candidates.json`.
- `data.clean` removes placeholders and lowercases. `data.context_list` turns each `<CITATION>` sentence, plus one sentence on each side, into a short query.
- `lexical.remove_self` gives a query's own entry -1e9 (worth +0.022 F1@7). `data.load` never reads test labels.

## Lexical system (best submission)

1. Full view: TF-IDF cosine over 4 to 6 word phrases, fitted on candidates and queries (min_df 2, sublinear TF).
2. Best context view: a candidate's best match over the citation contexts.
3. Word view: TF-IDF over words and word pairs (max 300,000 features).
4. Co-citation: with n labelled judgments citing both query and candidate, the score adds log(1 + n).
5. Neighbour votes: the 5 most similar labelled queries vote for the cases they cite.
6. CSLS (k = 4) lowers candidates that look similar to every query. r_c and r_q are the mean of the 4 highest similarities.

```
CSLS(q, c) = 2 sim(q, c) - r_c - r_q
```

The final score (`rerank.final_score`) adds the three views, each standardised per query (`z`), the co-citation boost and four times the votes.

```
z(CSLS(full)) + z(CSLS(best context)) + 0.75 z(CSLS(words)) + log(1 + co-citations) + 4 x votes
```

Co-citation and votes use train labels for dev, and train and dev labels for test.

## Dense retrievers (experiments)

None of the four dense Phase 1 models improved the lexical system. TF-IDF on 4 to 6 word phrases alone gets 0.447 on dev.

- A judgment vector is the mean of at most 2 chunk vectors (first and last).
- Training: 5,308 (query, precedent) pairs, hard negatives from the top 20 TF-IDF candidates, InfoNCE over 20 x cosine, AdamW, weight decay 0.01, accumulation 4, clipping 1.0.
- LoRMA and QLoRA (4-bit NF4) use rank 16, alpha 32. Distillation (D5) uses the A4 loss with tau = 2 and teacher logits of 30 x TF-IDF cosine.

| Run | Model | Technique | LR | Chunks / negatives | Epochs | Trainable | Dev F1@7, dense alone |
|---|---|---|---|---|---|---|---|
| d3_arch3_partial | Arch 3 encoder | partial fine-tuning (top 4 blocks) | 1e-4 | 2 / 4 | best of 2 | 19.71M | 0.145 |
| d4_arch2_ia3 | Arch 2 | IA3 | 3e-3 | 2 / 4 | best of 2 | 0.02M | 0.055 |
| d1_arch2_lorma | Arch 2 | LoRMA | 3e-4 | 2 / 4 | best of 2 | 2.15M | 0.018 |
| d5_arch2_lorma_kd | Arch 2 | LoRMA + distillation from TF-IDF | 3e-4 | 2 / 4 | best of 2 | 2.15M | 0.010 |
| d2b_arch4_qlora | Arch 4 bidirectional | QLoRA | 3e-4 | 1 / 2 | up to 2 | 3.68M | not logged |
| d6b_arch4_lorma | Arch 4 bidirectional | LoRMA | 3e-4 | 1 / 2 | best of 2 | 4.15M | 0.104 |
| d7_arch1_lorma | Arch 1 | LoRMA | 3e-4 | 1 / 2 | 1 | 2.66M | 0.006 |

Runs stop after 150 minutes (45 for Arch 1). The "b" runs use 1 chunk and 2 negatives because the first Arch 4 runs ran out of memory.

A dense run is added as `w x z(CSLS(dense))` (submissions 8, 9 and 13). Arch 1 LoRMA with `W_DENSE = 0.1` (dev 0.4914) was not scored.

## Files

| File | What it does |
|---|---|
| `data.py` | Loading, cleaning, citation contexts |
| `lexical.py` | TF-IDF views, self removal, CSLS, micro-F1@7 |
| `citation_graph.py` | Co-citation counts and neighbour votes |
| `rerank.py` | Final score and its weights (`K_CSLS`, `W_WORDS`, `W_VOTES`) |
| `predict.py` | Prints dev F1@7, writes the zip |
| `tokenizer.py` | My Phase 1 BPE on letter, digit and punctuation pieces |
| `legal_bpe_16384.json` | The Phase 1 BPE merges |
| `prep_dense.py` | Writes `p7_data.pkl.gz` (BPE ids, TF-IDF top 50) |
| `dense_model.py` | Phase 1 backbone, pooling, chunking |
| `peft_methods.py` | LoRMA+, QLoRA with NF4, IA3, partial fine-tuning |
| `distill.py` | The A4 distillation loss |
| `train_dense.py` | Trains one dense run from `CONFIGS`, saves score matrices |

## How to run

Dependencies: Python 3, numpy, scikit-learn, torch (`peft` for IA3). Dense training needs a CUDA GPU (I used a Kaggle T4). Run commands from this folder.

| File | Constants |
|---|---|
| `data.py` | `DATA_DIR` |
| `predict.py` | `DATA_DIR`, `OUT_DIR`, `DENSE_FILE`, `W_DENSE`, `VOTE_K` |
| `prep_dense.py` | `DATA_DIR`, `TOKENIZER_PATH`, `PREP_FILE` |
| `train_dense.py` | `PREP_FILE`, `MODELS_DIR`, `OUT_DIR`, `DEVICE` |

1. Best submission (CPU, no training):
   ```
   python predict.py
   ```
   It prints `dev F1@7 0.4936 ...` and writes `submissions/p7_final_dev0.4936.zip`.
2. Dense runs (optional, GPU):
   ```
   python prep_dense.py
   python train_dense.py d3_arch3_partial
   ```
   `MODELS_DIR` needs `arch1`, `arch2`, `arch3` and `arch4_mntp`, each with `model.pt` and the `../phase1` model file as `model_arch.py` (Arch 3 and 4 also `metrics.json`, Arch 4 also `arch4_engram_compress.pt`). Output goes to `dense_runs/<run>/`.
3. To add a dense run, set `DENSE_FILE = "dense_runs/<run>/scores.npz"` and `W_DENSE` in `predict.py`, then rerun it.

## Submissions

Phase 7 has 14 scored submissions. Unlisted settings keep their defaults (`W_WORDS = 0.75`, `W_VOTES = 4.0`, `K_CSLS = 4`, `VOTE_K = 5`, no dense file).

| # | Recipe | Settings (`rerank.py` / `predict.py`) | Dev F1@7 | Codabench F1@7 |
|---|---|---|---|---|
| 1 | CSLS (full, best context) + co-citation | `W_WORDS = 0`, `W_VOTES = 0` | 0.4792 | 0.4643 |
| 2 | #1 + 4 x neighbour votes | `W_WORDS = 0` | 0.4858 | 0.4719 |
| 6 | #2 + 0.5 x word view | `W_WORDS = 0.5` | 0.4925 | 0.4826 |
| 8 | #2 + 0.25 x Arch 2 IA3 dense | `W_WORDS = 0`, `DENSE_FILE` = d4_arch2_ia3, `W_DENSE = 0.25` | 0.4869 | 0.4719 |
| 9 | #2 + 0.25 x Arch 4 QLoRA dense | `W_WORDS = 0`, `DENSE_FILE` = d2b_arch4_qlora, `W_DENSE = 0.25` | 0.4858 | 0.4725 |
| 10 (best) | #2 + 0.75 x word view | defaults | 0.4936 | 0.4832 |
| 11 | #2 + 1.0 x word view | `W_WORDS = 1.0` | 0.4925 | 0.4820 |
| 13 | #6 + 0.25 x Arch 4 QLoRA dense | `W_WORDS = 0.5`, `DENSE_FILE` = d2b_arch4_qlora, `W_DENSE = 0.25` | 0.4925 | 0.4802 |
| 14 | #10 with 6 x votes | `W_VOTES = 6.0` | 0.4925 | 0.4808 |

Submissions 3, 4, 5, 7 and 12 used views not in this folder. They scored 0.4572, 0.4572, 0.4814, 0.4790 and 0.4814.
