# Phase 7: Prior case retrieval (IL-PCR)

## Task

The input is a query judgment where the citations, names and organisations are hidden. The output is the 7 earlier cases from a candidate pool that the query most likely cites. The metric is micro-F1@7 over all queries, and even a perfect ranking only reaches 0.618 on dev, because I must always return exactly 7 cases while queries cite anywhere from 1 to more than 30.

## My final system

- It starts from word statistics. It compares each query with each candidate using TF-IDF on 4 to 6 word phrases, on the sentences around each hidden citation, and on single words and word pairs.
- It adds a bonus when other training judgments cite both the query and the candidate (co-citation).
- The 5 most similar training queries vote for the cases they cite.
- It adds the score of my fine-tuned Arch 4 bi-encoder (run `d2b_arch4_qlora`, trained with QLoRA). It turns each judgment into one vector and scores a pair by how close the two vectors are.
- It uses CSLS to push down candidates that look similar to every query, and it removes the query itself from its own pool.
- Leaderboard score: 0.4802 micro-F1@7 (submission 13), dev F1@7 0.4925.

The final score adds the phrase and citation sentence scores, 0.5 times the word score, the co-citation bonus, 4 times the votes and 0.25 times the Arch 4 QLoRA score. The bonus and votes use training labels for dev, and training and dev labels for test.

## Models I tried

I trained my Phase 1 models as dense retrievers. Each one turns a judgment into one vector and learns to put a query close to the cases it cites. Alone they are much weaker than the word statistics, so I only use one of them, Arch 4 QLoRA, as a small extra score.

| Phase 1 model | Technique | Dev F1@7 alone | Codabench when added to my system |
|---|---|---|---|
| Arch 3 (encoder) | partial fine-tuning (top 4 blocks) | 0.145 | not scored |
| Arch 2 | IA3 | 0.055 | 0.4719 |
| Arch 2 | LoRMA | 0.018 | not scored |
| Arch 2 | LoRMA, distilled from TF-IDF scores | 0.010 | not scored |
| Arch 4 (bidirectional) | QLoRA | not logged | 0.4725 and 0.4802 |
| Arch 4 (bidirectional) | LoRMA | 0.104 | not scored |
| Arch 1 | LoRMA | 0.006 | not scored |

For comparison, TF-IDF on 4 to 6 word phrases alone gets 0.447 on dev.

## Files

| File | What it does |
|---|---|
| `data.py` | Loads and cleans the data, finds the sentences around citations |
| `lexical.py` | TF-IDF scores, CSLS, self removal and micro-F1@7 |
| `citation_graph.py` | Co-citation counts and neighbour votes |
| `rerank.py` | The final score and its weights |
| `predict.py` | Prints dev F1@7 and writes the submission zip |
| `tokenizer.py` | My Phase 1 BPE tokenizer |
| `prep_dense.py` | Tokenizes the data for the dense runs |
| `dense_model.py` | Loads a Phase 1 model and makes judgment vectors |
| `peft_methods.py` | LoRMA, QLoRA, IA3 and partial fine-tuning |
| `distill.py` | The distillation loss |
| `train_dense.py` | Trains one dense run |

## How to run

Set these paths at the top of the files:

- `data.py`: `DATA_DIR`
- `predict.py`: `DATA_DIR`, `OUT_DIR`, `DENSE_FILE`, `W_DENSE`
- `prep_dense.py`: `DATA_DIR`, `TOKENIZER_PATH`, `PREP_FILE`
- `train_dense.py`: `PREP_FILE`, `MODELS_DIR`, `OUT_DIR`, `DEVICE`

`DATA_DIR` holds `train.json`, `train_candidates.json`, `dev.json`, `dev_candidates.json`, `test.json` and `test_candidates.json`. Run all commands from this folder.

1. Train the Arch 4 QLoRA run (needs a GPU):
   ```
   python prep_dense.py
   python train_dense.py d2b_arch4_qlora
   ```
   `MODELS_DIR` needs the folders `arch1`, `arch2`, `arch3` and `arch4_mntp` from my Phase 1 and Phase 2 models. Output goes to `dense_runs/<run>/`, so this run writes `dense_runs/d2b_arch4_qlora/scores.npz`. The other run names are listed in `CONFIGS` in `train_dense.py`.
2. Final submission:
   ```
   python predict.py
   ```
   It prints dev F1@7 0.4925 and writes `submissions/p7_final_dev0.4925.zip`, which is submission 13.
3. To run without a dense model, set `DENSE_FILE = None` in `predict.py`. To use another dense run, set `DENSE_FILE = "dense_runs/<run>/scores.npz"` and `W_DENSE`.

Dependencies: Python 3, numpy, scikit-learn, torch, and peft for IA3.

## Submissions this code reproduces

Phase 7 has 14 scored submissions. The defaults are `W_WORDS = 0.5` and `W_VOTES = 4.0` in `rerank.py`, and `DENSE_FILE` set to the `d2b_arch4_qlora` scores with `W_DENSE = 0.25` in `predict.py`. Rows without a dense model use `DENSE_FILE = None`.

| Submission | Codabench |
|---|---|
| 1: phrases and citation sentences with co-citation (`W_WORDS = 0`, `W_VOTES = 0`) | 0.4643 |
| 2: 1 plus neighbour votes (`W_WORDS = 0`) | 0.4719 |
| 6: 2 plus word view at 0.5 | 0.4826 |
| 8: 2 plus Arch 2 IA3 dense at 0.25 (`W_WORDS = 0`) | 0.4719 |
| 9: 2 plus Arch 4 QLoRA dense at 0.25 (`W_WORDS = 0`) | 0.4725 |
| 10: 2 plus word view at 0.75 (`W_WORDS = 0.75`) | 0.4832 |
| 11: 2 plus word view at 1.0 (`W_WORDS = 1.0`) | 0.4820 |
| 13: 6 plus Arch 4 QLoRA dense at 0.25 (defaults, my final system) | 0.4802 |
| 14: 10 with votes at 6 (`W_WORDS = 0.75`, `W_VOTES = 6.0`) | 0.4808 |
