# CS781 Capstone: In-House Legal Intelligence System

Abeer Gupta (230031), Codabench username a_230031.

This folder has the code for my best models in all seven capstone phases. I trained every model from scratch with my own BPE tokenizer, using only the official course data. In Phase 1 I trained four language models, and in Phases 2 to 7 I fine-tuned all four for one legal task each.

| Phase | Task | Metric | Leaderboard score | Rank | Folder |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining | BERTScore F1 | 0.8402 | 17 | `phase1` |
| 2 | Legal NER | strict macro-F1 | 0.6503 | 7 | `phase2` |
| 3 | Rhetorical role prediction | macro-F1 | 0.6345 | 6 | `phase3` |
| 4 | Court judgment prediction | macro-F1 | 0.8154 | 11 | `phase4` |
| 5 | Legal statute identification | macro-F1 | 0.4450 | 8 | `phase5` |
| 6 | Summarization | BERTScore F1 | 0.8527 | 14 | `phase6` |
| 7 | Prior case retrieval | micro-F1@7 | 0.4802 | 14 | `phase7` |

Each phase folder has its own README with the task, my final system, the models I tried, the files, how to run them, and the Codabench submissions they reproduce. Each phase folder also has an `experiments/` folder with the code for my other Codabench submissions, listed under "Other submissions" at the end of its README.

## Phase 1 models

| Model | Type | Parameters |
|---|---|---|
| Arch 1 | GPT-2 style decoder (my A3 code) | 97.6M |
| Arch 2 | LLaMA style decoder | 61.6M |
| Arch 3 | encoder-decoder | 86.1M |
| Arch 4 | latent-attention decoder with n-gram memory | 139.1M |

`phase1/ablations` has small runs that test the design choices of Arch 2 and Arch 4, one at a time.

## Bidirectional Arch 4

Arch 4 is a decoder, so each token can only see the tokens before it. For the later tasks I wanted each token to see the whole text. In `phase2/adapt.py` I first continued its next-token training for 1500 updates, then removed the causal mask and trained it for 800 more updates with masked next-token prediction. It is my best model or the teacher of my best model in Phases 2 to 6.

## Setup

- Python 3 with the packages in `requirements.txt`: torch, numpy, scipy, scikit-learn, peft, ijson, pyahocorasick, bert_score, matplotlib.
- I trained on Kaggle T4 GPUs. The TF-IDF parts of Phases 6 and 7 run on CPU.
- Phases 2 to 7 load the Phase 1 model code from `phase1`. The tokenizer is `phase1/legal_bpe_16384.json`.
- Data and model weights are not included. Each script sets its paths as constants at the top.
