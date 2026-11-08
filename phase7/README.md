# Phase 7: Prior Case Retrieval (PCR)

## Task and metric

- Data: IL-PCR. A query is an Indian Supreme Court judgment. The citations inside it are hidden as `<CITATION>`, names as `<NAME>` and organisations as `<ORG>`.
- Each split has its own pool of candidate judgments. Every query is also inside its own pool.
- For every query I return exactly 7 candidate ids, best first.
- Metric: micro-F1@7. Hits, picks and relevant cases are summed over all queries. P = hits / (7 x queries), R = hits / relevant cases, F1 = 2PR / (P + R).
- Because I always pick 7, a perfect ranking reaches only 0.618 on dev.

| split | queries | pool | relevant per query |
|---|---|---|---|
| train | 827 | 4,320 | median 4 |
| dev | 118 | 1,023 | median 5 |
| test | 237 | 1,727 | hidden |

## Data and text cleaning

- Files: `train.json`, `train_candidates.json`, `dev.json`, `dev_candidates.json`, `test.json`, `test_candidates.json` in `data/`. Each judgment is a list of sentences.
- Lexical system (`data.clean`): the `<NAME>`, `<ORG>` and `<CITATION>` placeholders are removed, spaces are collapsed and the text is lowercased. The TF-IDF token pattern keeps only alphanumeric words (`[a-z0-9]+`).
- Citation contexts (`data.context_list`): every sentence that contains `<CITATION>`, plus one sentence on each side, becomes its own short query.
- Self removal (`lexical.remove_self`): a query's own entry in its pool gets a score of -1e9, so it is never picked. This alone was worth +0.022 F1@7.
- Dense models (`data.model_text`, `tokenizer.py`): placeholders become "X" or "citation", and a judgment longer than 4,000 words keeps its first 3,000 and last 1,000 words. The text is encoded with my Phase 1 BPE (16,384 tokens, from Assignment 1, file `legal_bpe_16384.json`). Each word is split into letter, digit and punctuation pieces before BPE.
- Test labels are never read: `data.load` sets the relevant list of every test query to empty.

## Models

The final ranking is lexical and needs no training. The four Phase 1 models (Arch 1 is my A3 decoder, 97.6M; Arch 2 61.6M; Arch 3 86.1M; Arch 4 139.1M) were fine-tuned as dense retrievers for the Phase 7 experiments. Their code lives in `../phase1`, and the full architecture table is in `../phase1/README.md`.

How each one is used here (`dense_model.py`):

- Arch 1 and Arch 2: causal decoders. The LM head is removed. A chunk vector is the hidden state of the last token.
- Arch 3: only the 10-layer encoder is used, the decoder is removed. Each chunk starts with one continuation marker token, so a chunk holds 703 text tokens (704 positions). A chunk vector is the mean of the encoder states.
- Arch 4: the bidirectional version from Phase 2 (causal mask removed and adapted with masked next-token prediction, code in `../phase2/adapt.py`). A chunk vector is the mean of the hidden states.

## Methodology

### Lexical system (the submitted ranking)

1. Full query view: TF-IDF cosine over 4 to 6 word phrases. The vocabulary is fitted on candidates and queries together, with minimum document frequency 2 and sublinear TF. A passage quoted in a precedent and in its citing query is then kept.
2. Best context view: each citation context is scored as its own query with the same vectorizer, and a candidate keeps its best match.
3. Word view: TF-IDF over single words and word pairs (at most 300,000 features), fitted the same way.
4. Co-citation: n = number of labelled judgments that cite both the query and the candidate. The score adds log(1 + n).
5. Neighbour votes: the 5 most similar labelled queries (TF-IDF over words and word pairs) vote for the cases they cite, each vote weighted by that similarity.
6. Hubness correction (CSLS, k = 4): CSLS(q, c) = 2 sim(q, c) - r_c - r_q, where r_c is the mean of the 4 highest similarities of candidate c to the queries, and r_q the same for the query. It needs no labels.
7. Final score (`rerank.final_score`):
   `z(CSLS(full)) + z(CSLS(best context)) + 0.75 z(CSLS(words)) + log(1 + co-citations) + 4 x votes`.
   `z` standardises each query's scores to mean 0 and standard deviation 1. The top 7 are returned.

For dev scores, co-citation and votes use the train labels only. For the test file they use the train and dev labels (both official).

### Dense retrievers (experiments)

- One vector per judgment: the BPE ids are cut into chunks with a quarter overlap, at most 2 chunks are kept (the first and the last), each chunk gives one vector, and the judgment vector is their mean scaled to length 1. Similarity is the dot product.
- Training data: every (train query, true precedent) pair, 5,308 pairs.
- Hard negatives: sampled from the query's 20 best TF-IDF candidates (4 to 6 word phrases, fitted on candidates only) that are not precedents.
- Loss: InfoNCE. Logits are 20 x cosine(query, candidate) over the precedent and its negatives, with cross-entropy on the precedent.
- Optimiser: AdamW, constant learning rate, weight decay 0.01, gradients accumulated over 4 pairs, gradient norm clipped to 1.0. fp16 autocast, except Arch 1 in fp32.
- After each epoch the dev F1@7 of the dense scores alone is measured and the best epoch is kept.
- LoRMA (from A4): LoRMA+, W' = (alpha / r B A + I) W0, on every linear layer of every block, rank 16, alpha 32, B starts at zero.
- QLoRA (from A4): every linear layer of every block is stored as packed 4-bit NF4 codes (blocks of 64, fp16 absmax) and frozen, with a trainable LoRA on top, rank 16, alpha 32.
- IA3: learned scaling vectors on the key and value projections and the FFN output projection (the `peft` library).
- Partial fine-tuning: only the top 4 encoder blocks and the final encoder norm of Arch 3 train.
- Distillation (D5): the A4 loss, 0.5 x CE + 0.5 x tau^2 x KL(softmax(teacher / tau) || softmax(student / tau)), tau = 2. The classes are the same candidates (precedent and negatives). The teacher logits are 30 x their TF-IDF cosine scores, so the Arch 2 student learns the teacher's whole ranking.

| run | model | technique | LR | chunks / negatives | epochs | trainable | dev F1@7 dense alone |
|---|---|---|---|---|---|---|---|
| d3_arch3_partial | Arch 3 encoder | partial fine-tuning (top 4 blocks) | 1e-4 | 2 / 4 | best of 2 | 19.71M | 0.145 |
| d4_arch2_ia3 | Arch 2 | IA3 | 3e-3 | 2 / 4 | best of 2 | 0.02M | 0.055 |
| d1_arch2_lorma | Arch 2 | LoRMA | 3e-4 | 2 / 4 | best of 2 | 2.15M | 0.018 |
| d5_arch2_lorma_kd | Arch 2 | LoRMA + distillation from TF-IDF | 3e-4 | 2 / 4 | best of 2 | 2.15M | 0.010 |
| d2b_arch4_qlora | Arch 4 bidirectional | QLoRA | 3e-4 | 1 / 2 | up to 2 | 3.68M | not logged |
| d6b_arch4_lorma | Arch 4 bidirectional | LoRMA | 3e-4 | 1 / 2 | best of 2 | 4.15M | 0.104 |
| d7_arch1_lorma | Arch 1 | LoRMA | 3e-4 | 1 / 2 | 1 | 2.66M | 0.006 |

Runs stop after 150 minutes (45 for Arch 1). The first Arch 4 runs (d2_arch4_qlora and d6_arch4_lorma, 2 chunks and 3 negatives, not kept in `train_dense.py`) ran out of GPU memory, so they were rerun with 1 chunk and 2 negatives (the "b" runs). For reference, TF-IDF 4 to 6 word phrases alone reach 0.447 on dev. No dense model improved the lexical system, so the best submission does not use one.

### Ensembling

A dense run can be added to the final score as `w x z(CSLS(dense))` with `DENSE_FILE` and `W_DENSE` in `predict.py`. This was tried for Arch 2 IA3 and Arch 4 QLoRA (submissions 8, 9 and 13). Arch 1 LoRMA with `W_DENSE = 0.1` was also tried (dev F1@7 0.4914), but that submission was not scored.

## Files

| file | what it does |
|---|---|
| `data.py` | loads a split, cleans text, builds citation contexts and the dense model input text |
| `lexical.py` | TF-IDF vectorizer, the three lexical views, self removal, z-score, CSLS, top 7, micro-F1@7, candidate-only TF-IDF for hard negatives |
| `citation_graph.py` | co-citation counts and neighbour votes from a labelled graph |
| `rerank.py` | the final score and its weights (`K_CSLS`, `W_WORDS`, `W_VOTES`) |
| `predict.py` | builds all signals for dev and test, prints dev F1@7, checks the test predictions and writes the zip |
| `tokenizer.py` | my Phase 1 BPE on letter, digit and punctuation pieces |
| `legal_bpe_16384.json` | the Phase 1 BPE merges |
| `prep_dense.py` | BPE ids of every judgment and the TF-IDF top 50 per query, saved to `p7_data.pkl.gz` |
| `dense_model.py` | loads a Phase 1 checkpoint as a backbone, the A4 classifier with pooling, chunking |
| `peft_methods.py` | LoRMA+, QLoRA with NF4, IA3 (through `peft`), partial fine-tuning |
| `distill.py` | the A4 distillation loss |
| `train_dense.py` | trains one dense run from `CONFIGS`, saves dev and test score matrices |

## How to run

Dependencies: Python 3, numpy, scikit-learn, torch. `peft` is needed only for the IA3 run. Dense training needs a CUDA GPU (I used a Kaggle T4). The lexical system runs on CPU.

Path constants: `DATA_DIR` in `data.py`, `predict.py` and `prep_dense.py`. `OUT_DIR`, `DENSE_FILE`, `W_DENSE` and `VOTE_K` in `predict.py`. `TOKENIZER_PATH` and `PREP_FILE` in `prep_dense.py`. `PREP_FILE`, `MODELS_DIR`, `OUT_DIR` and `DEVICE` in `train_dense.py`. Run every command from this folder.

1. Best submission (lexical, CPU, no training):
   ```
   python predict.py
   ```
   Input: the six JSON files in `data/`. It prints `dev F1@7 0.4936 ...` and writes `submissions/p7_final_dev0.4936.zip` with `prediction.json` (test query id to a list of 7 candidate ids).
2. Dense runs (optional, GPU):
   ```
   python prep_dense.py
   python train_dense.py d3_arch3_partial
   ```
   `MODELS_DIR` must hold one folder per model: `arch1`, `arch2`, `arch3` and `arch4_mntp`. Each folder holds `model.pt` and the architecture file from `../phase1` saved as `model_arch.py`; Arch 3 and Arch 4 also need their `metrics.json` (model config), and Arch 4 its `arch4_engram_compress.pt`. Output: `dense_runs/<run>/scores.npz` (dev and test score matrices) and `metrics.json`. Any other run name from `CONFIGS` works the same way.
3. To add a dense run to the ranking, set `DENSE_FILE = "dense_runs/<run>/scores.npz"` and `W_DENSE` in `predict.py`, then run `python predict.py` again.

## Submissions made with this code

Phase 7 has 14 scored submissions. Settings not listed keep their defaults (`W_WORDS = 0.75`, `W_VOTES = 4.0`, `K_CSLS = 4`, `VOTE_K = 5`, no dense file).

| # | recipe | settings (`rerank.py` / `predict.py`) | dev F1@7 | Codabench F1@7 |
|---|---|---|---|---|
| 1 | CSLS (full, best context) + co-citation | `W_WORDS = 0`, `W_VOTES = 0` | 0.4792 | 0.4643 |
| 2 | #1 + 4 x neighbour votes | `W_WORDS = 0` | 0.4858 | 0.4719 |
| 6 | #2 + 0.5 x word view | `W_WORDS = 0.5` | 0.4925 | 0.4826 |
| 8 | #2 + 0.25 x Arch 2 IA3 dense | `W_WORDS = 0`, `DENSE_FILE` = d4_arch2_ia3, `W_DENSE = 0.25` | 0.4869 | 0.4719 |
| 9 | #2 + 0.25 x Arch 4 QLoRA dense | `W_WORDS = 0`, `DENSE_FILE` = d2b_arch4_qlora, `W_DENSE = 0.25` | 0.4858 | 0.4725 |
| **10 (best)** | **#2 + 0.75 x word view** | **defaults** | **0.4936** | **0.4832** |
| 11 | #2 + 1.0 x word view | `W_WORDS = 1.0` | 0.4925 | 0.4820 |
| 13 | #6 + 0.25 x Arch 4 QLoRA dense | `W_WORDS = 0.5`, `DENSE_FILE` = d2b_arch4_qlora, `W_DENSE = 0.25` | 0.4925 | 0.4802 |
| 14 | #10 with 6 x votes | `W_VOTES = 6.0` | 0.4925 | 0.4808 |

Submissions 3, 4, 5, 7 and 12 used views that are not in this folder (no CSLS, merged citation contexts, word 2 to 3 grams). Their scores are 0.4572, 0.4572, 0.4814, 0.4790 and 0.4814.
