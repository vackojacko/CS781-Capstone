# Phase 3: Rhetorical Role Prediction

## Task

The input is a legal judgment split into sentences. The output gives each sentence one of 13 rhetorical roles: Fact, Issue, ArgumentPetitioner, ArgumentRespondent, Statute, Dissent, PrecedentReliedUpon, PrecedentNotReliedUpon, PrecedentOverruled, RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt and None. The metric is macro-F1 over all 13 labels, with the sentences of all test documents pooled together.

## My final system

- One model covers both domains (Competition Law and Income Tax), with the domain given as an input.
- Each Phase 1 model encodes the sentences, and a BiLSTM-CRF on top predicts the roles. The LSTM code is from my Assignment 2.
- The final answer averages the sentence probabilities of three models: 0.6 distilled Arch 2, 0.2 Arch 4 with LoRA (rr1) and 0.2 Arch 4 with longer adaptation and LoRA (x2).
- A Viterbi decoder then uses label transition counts from the training documents.
- A simple rule marks a Dissent block: it starts at a sentence like "Name, Member (dissenting)" and runs to the end of the document.
- All 13 labels can be predicted. Best Codabench score: 0.6345.

## Models I tried

| Phase 1 model | Technique | Codabench |
|---|---|---|
| Arch 1 | LoRMA+ | 0.1103 |
| Arch 3 (encoder only) | IA3 | 0.4479 |
| Arch 4, bidirectional | LoRMA+ (rr2b) | 0.4739 |
| Arch 4, bidirectional, longer adaptation | LoRA (x2) | 0.5129 |
| Arch 4, bidirectional | LoRA (rr1) | 0.5139 |
| Arch 2 | LoRA, distilled from rr1 and x2 | 0.5354 |

The bidirectional Arch 4 comes from Phase 2. For x2 it was adapted for longer, with the Phase 3 training text added. For the distilled Arch 2, the teacher is the average of rr1 and x2, and the student learns half from the gold labels and half from the teacher.

## Files

| File | What it does |
|---|---|
| `backbone.py` | piece tokenizer, loads a Phase 1 model as a sentence encoder, path constants |
| `data.py` | loads documents, tokenizes, packs sentences into windows, counts label transitions |
| `model.py` | LSTM, CRF and the role model |
| `peft_methods.py` | LoRMA+ layer and the code that adds LoRMA+, LoRA or IA3 |
| `train.py` | loss, training loop, the settings of the single runs |
| `distill.py` | teacher probabilities and the distilled Arch 2 |
| `decode.py` | CRF prediction, Viterbi with counted transitions, macro-F1 |
| `rules.py` | Dissent rule |
| `predict.py` | the final ensemble and the best submission |

## How to run

Dependencies: Python 3, `torch`, `numpy`, `scikit-learn` and `peft`. Training needs a GPU (I used a Kaggle T4). On Kaggle, uninstall `torchao` first, or `peft` crashes.

Path constants at the top of the files:

- `DATA_DIR = "data"` in `data.py`: `CL_train.json`, `CL_dev.json`, `CL_test.json`, `IT_train.json`, `IT_dev.json`, `IT_test.json`.
- `BPE_PATH = "tokenizer/legal_bpe_16384.json"` in `backbone.py`. The same folder needs `bpe_tokenizer.py` (a copy of `../phase1/tokenizer.py`).
- `KIT_DIR = "phase2_kit"` in `backbone.py`: folders `arch1/`, `arch2/` and `arch3/`, each with `model_arch.py` (a copy of `../phase1/archN_model.py`) and `model.pt`. Arch 3 also needs `metrics.json`.
- `ARCH4_MNTP_DIR = "arch4_mntp"` and `ARCH4_MNTP_P3_DIR = "arch4_mntp_p3"` in `backbone.py`: each with `model_arch.py`, `model.pt`, `metrics.json` and `arch4_engram_compress.pt`.
- `OUT_DIR = "runs"` in `train.py`, and `RUNS_DIR = "runs"` and `SUBMISSION_PATH = "submission_phase3.zip"` in `predict.py`. `RUNS_DIR` must match `OUT_DIR`.

Commands for the best submission, in this order:

```
python train.py rr1
python train.py x2
python distill.py teacher
python distill.py student
python predict.py
```

The other runs can be trained on their own: `python train.py rr2b`, `python train.py rr3` and `python train.py rr2`. Each `train.py` run writes its probabilities, model and submission zip to `runs/`. `predict.py` only reads saved probabilities, so it runs on a CPU in seconds.

## Submissions this code reproduces

| Submission | Codabench |
|---|---|
| Arch 1, LoRMA+ (rr2) | 0.1103 |
| Arch 3 encoder, IA3 (rr3) | 0.4479 |
| Arch 4, LoRMA+ (rr2b) | 0.4739 |
| Arch 4 longer adaptation, LoRA (x2) | 0.5129 |
| Arch 4, LoRA (rr1) | 0.5139 |
| Distilled Arch 2 | 0.5354 |
| Ensemble, Viterbi and Dissent rule (best) | 0.6345 |

## Notes

- The step that made `arch4_mntp_p3` is not in this folder. The code only loads its result.
- On my saved probabilities, `predict.py` gives 0.6345 again. Retraining is not exactly the same each time, and dev scores moved by about 0.01 between reruns.
- The single runs were submitted before I added the Dissent rule, so they never predicted Dissent or PrecedentOverruled. To get them exactly, set `NEVER = [5, 8]` in `decode.py` and remove the Dissent rule call in `decode.py` before training.
- `distill.py` writes no zip. The distilled Arch 2 submission is its CRF prediction on the test set.
