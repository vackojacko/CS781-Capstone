# CS781 Capstone: In-House Legal Intelligence System

Abeer Gupta (230031), Codabench username a_230031.

This folder has the code of my best models for the seven capstone phases.
I trained every model from scratch with my own BPE tokenizer, on the official course data only.
Phase 1 trains four language models. Phases 2 to 7 fine-tune all four for one legal task each.

| Phase | Task | Metric | Best score | Rank | Folder |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining | BERTScore F1 | 0.8688 | 2 | `phase1` |
| 2 | Legal NER | strict macro-F1 | 0.6872 | 3 | `phase2` |
| 3 | Rhetorical role prediction | macro-F1 | 0.6345 | 6 | `phase3` |
| 4 | Court judgment prediction | macro-F1 | 0.8154 | 11 | `phase4` |
| 5 | Legal statute identification | macro-F1 | 0.4732 | 1 | `phase5` |
| 6 | Summarization | BERTScore F1 | 0.8527 | 13 | `phase6` |
| 7 | Prior case retrieval | micro-F1@7 | 0.4832 | 13 | `phase7` |

Each phase folder has a README with the task, models, files, how to run, and the Codabench submissions it reproduces.

## Phase 1 models

| | Arch 1 | Arch 2 | Arch 3 | Arch 4 |
|---|---|---|---|---|
| Type | GPT-2 style decoder (my A3 code) | LLaMA style decoder | encoder-decoder | latent-attention decoder with n-gram memory |
| Parameters | 97.6M | 61.6M | 86.1M | 139.1M |

`phase1/ablations` has small runs that test the design choices of Arch 2 and Arch 4, one part at a time.

## Final systems in later phases

| Phase | Final system | Distillation |
|---|---|---|
| 2 | ensemble with the bidirectional Arch 4, BIOES Viterbi, propagation and document rules | Arch 2 student of Arch 4 |
| 3 | BiLSTM-CRF over sentence vectors, transition Viterbi, Dissent rule | Arch 2 student of two Arch 4 runs |
| 4 | BiLSTM over the last 8 chunks, three Arch 4 runs averaged | Arch 2 and Arch 4 students |
| 5 | label attention, per-label thresholds, 12 runs plus kNN label votes | Arch 2 student of Arch 4 |
| 6 | extractive: TF-IDF centrality mixed with seven salience models | six cross-architecture students |
| 7 | TF-IDF phrases, citation contexts, CSLS, co-citation and neighbour votes | Arch 2 student of TF-IDF scores |

The fine-tuning method used for each model is listed in each phase README.
The bidirectional Arch 4 comes from `phase2/adapt.py`: I removed the causal mask and trained it for 800 updates with masked next-token prediction.

## Setup

- Python 3, PyTorch, NumPy, scikit-learn, SciPy, peft, ijson, pyahocorasick, bert_score, matplotlib (see `requirements.txt`).
- I trained on Kaggle T4 GPUs. The lexical parts of Phases 5 to 7 run on CPU.
- Phases 2 to 7 load the Phase 1 model code from `phase1`. The tokenizer is `phase1/legal_bpe_16384.json`.
- Data and model weights are not included. Each script sets its paths as constants at the top.
