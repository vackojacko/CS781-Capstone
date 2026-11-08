# CS781 Capstone: In-House Legal Intelligence System

Abeer Gupta (230031), Codabench username a_230031.

This repository has the code of my best models for the seven phases of the CS781 capstone.
Every model is trained from scratch with my own BPE tokenizer on the official course data only.
Phase 1 trains four language models, and Phases 2 to 7 fine-tune all four of them for one legal task each.

| Phase | Task | Metric | Best score | Rank | Folder |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining | BERTScore F1 | 0.8688 | 2 | `phase1` |
| 2 | Legal NER | strict macro-F1 | 0.6872 | 3 | `phase2` |
| 3 | Rhetorical role prediction | macro-F1 | 0.6345 | 6 | `phase3` |
| 4 | Court judgment prediction | macro-F1 | 0.8154 | 11 | `phase4` |
| 5 | Legal statute identification | macro-F1 | 0.4732 | 1 | `phase5` |
| 6 | Summarization | BERTScore F1 | 0.8527 | 13 | `phase6` |
| 7 | Prior case retrieval | micro-F1@7 | 0.4832 | 13 | `phase7` |

Each phase folder has its own README with the task, data and preprocessing, the models and fine-tuning
methods, a description of every file, how to run it, and the Codabench submissions the code reproduces.

## Phase 1 models

| | Arch 1 | Arch 2 | Arch 3 | Arch 4 |
|---|---|---|---|---|
| Type | GPT-2 style decoder (my A3 code) | LLaMA style decoder | encoder-decoder | latent-attention decoder with n-gram memory |
| Parameters | 97.6M | 61.6M | 86.1M | 139.1M |
| Code | `arch1_model.py` | `arch2_model.py` | `arch3_model.py` | `arch4_model.py` |

`phase1/ablations` has the small-scale runs I used to test the design choices of Arch 2 and Arch 4,
one component removed or replaced at a time.

## Methods used in later phases

| Phase | Final system | Fine-tuning techniques (model) | Distillation |
|---|---|---|---|
| 2 | BIOES tagging, Viterbi, propagation and document rules over an ensemble with the bidirectional Arch 4 | full fine-tuning (Arch 2, 3, 4), LoRA (Arch 4), LoRMA (Arch 1) | Arch 2 student of Arch 4 |
| 3 | BiLSTM-CRF over sentence vectors, transition Viterbi, Dissent rule | LoRA (Arch 4), LoRMA (Arch 1, Arch 4), IA3 (Arch 3) | Arch 2 student of two Arch 4 runs |
| 4 | BiLSTM over the last 8 chunks, three Arch 4 runs averaged | LoRMA+ (Arch 4), gradual unfreezing (Arch 2), adapters (Arch 3), full fine-tuning (Arch 1) | Arch 2 and Arch 4 students |
| 5 | label attention, per-label thresholds, 12 runs plus kNN label votes | LoRMA+ and QLoRA (Arch 4), partial and full fine-tuning (Arch 2), adapters (Arch 3), full fine-tuning (Arch 1) | Arch 2 student of Arch 4 |
| 6 | extractive: TF-IDF centrality blended with seven salience models | LoRMA+ (Arch 2), LoRA (Arch 4), Pfeiffer adapters (Arch 3), Houlsby adapters (Arch 1) | six cross-architecture students |
| 7 | TF-IDF phrases, citation contexts, CSLS, co-citation and neighbour votes | partial fine-tuning (Arch 3), IA3 and LoRMA (Arch 2), QLoRA and LoRMA (Arch 4), LoRMA (Arch 1) | Arch 2 student of TF-IDF scores |

The bidirectional Arch 4 comes from `phase2/adapt.py`: the causal mask is removed and the model is trained
for 800 updates with masked next-token prediction. Phases 2 to 7 load the Phase 1 model code from `phase1`.

## Setup

- Python 3, PyTorch, NumPy, scikit-learn, SciPy, peft, ijson, pyahocorasick, bert_score, matplotlib (see `requirements.txt`).
- All training ran on Kaggle T4 GPUs. The lexical parts of Phases 5 to 7 run on CPU.
- The course data is not included. Each script has its data, tokenizer and model paths as constants at the top.
- The tokenizer file is `phase1/legal_bpe_16384.json`.
- Model weights are not included because of their size.
