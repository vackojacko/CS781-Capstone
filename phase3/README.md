# Phase 3: Rhetorical role prediction

## Task

Every sentence of a legal judgment gets one of 13 rhetorical roles: Fact, Issue, ArgumentPetitioner,
ArgumentRespondent, Statute, Dissent, PrecedentReliedUpon, PrecedentNotReliedUpon, PrecedentOverruled,
RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt, None.

The output is `{doc_id: [one label per sentence]}`. The metric is macro-F1 over all 13 labels, with the
sentences of all test documents pooled, as in the official `evaluate_phase3.py`. `decode.macro_f1` computes
the same score.

Best Codabench score: 0.6345.

## Data

- Two domains, Competition Law and Income Tax (IT), each with 40 train, 5 dev and 5 test documents.
- Dissent appears in only two Competition Law training documents, always as a closing block that starts
  with a "Member (dissenting)" header.
- One model serves both domains, with the domain as an input feature (`data.load_docs`).
- Tokenizer: my own Assignment 1 BPE (16,384 tokens), used as in Phase 2 (`backbone.PieceTokenizer`).
- Each sentence is cut to 128 tokens (`data.MAX_SENT_TOKENS`). Whole sentences are packed into windows of
  1024 tokens (704 for Arch 3, with one special token first) by `data.make_windows`.

## Models

The Phase 1 models are in `../phase1`. Arch 1 (97.6M) runs in fp32 because it did not learn in fp16.
Arch 2 (61.6M) is the distillation student. For Arch 3 (86.1M) I use only the encoder. Arch 4 (139.1M) is
the bidirectional version from Phase 2, in two forms:

- `arch4_mntp`: the Phase 2 version.
- `arch4_mntp_p3`: adapted for longer, with the Phase 3 training documents added to the adaptation text.
  That step is not in this folder. The code only loads the result.

The task model (`model.py`) is the MTL BiLSTM-CRF of Malik et al. 2022, with my backbones instead of BERT.
One BiLSTM (256 per direction) predicts the role, a second one predicts label changes, and a CRF picks the
label sequence. The LSTM code is from my Assignment 2.

## Fine-tuning runs

| Run | Phase 1 model | Technique | Trainable backbone params | Body LR | Epochs |
|---|---|---|---|---|---|
| rr1 | Arch 4 (`arch4_mntp`) | LoRA r = 16 | 3.62M | 5e-4 | 6 |
| x2 | Arch 4 (`arch4_mntp_p3`) | LoRA r = 16, backbone frozen for the first 2 epochs | 3.62M | 5e-4 | 15 |
| rr2b | Arch 4 (`arch4_mntp`) | LoRMA+ r = 16, backbone frozen for the first 2 epochs | 4.15M | 5e-4 | 6 |
| rr3 | Arch 3 (encoder) | IA3 | 0.03M | 3e-3 | 6 |
| rr2 | Arch 1 | LoRMA+ r = 16, fp32, backbone frozen for the first 2 epochs | 2.65M | 5e-4 | 6 |
| arch2_kd | Arch 2 | LoRA r = 16, distilled from rr1 + x2 | 2.14M | 5e-4 | 6 |

- LoRA (`peft`): r = 16, alpha = 16, dropout 0.05, on every attention and feed-forward linear layer.
- LoRMA+ (my Phase 2 layer, `peft_methods.LoRMALinear`): r = 16, alpha = 16. It did not learn when trained
  with the new head from the start, so its backbone is frozen for the first 2 epochs.

Training (`train.py`):

- Loss: 0.4 CRF negative log-likelihood + 0.3 class-weighted cross-entropy + 0.6 label-shift loss.
- Class weights: square root of inverse label frequency, divided by their mean, capped at 10.
- Pieces of at most 64 sentences and 4,096 tokens. AdamW, weight decay 0.01, head LR 1e-3, 5% warmup,
  clipping 1.0, seed 0. I keep the epoch with the best dev macro-F1.

Distillation (`distill.py`): the teacher is the average of the CRF marginals of rr1 and x2 on the training
documents. The student loss is half the normal loss plus half a KL loss with temperature 2.

## Best submission (`predict.py`, `decode.py`, `rules.py`)

1. Average the sentence probabilities: 0.6 distilled Arch 2 (overlapping windows) + 0.2 rr1 + 0.2 x2.
2. Viterbi decoding with one shared label-transition matrix counted on the training documents (add-one
   smoothing), weighted by `BETA = 0.3`.
3. Dissent rule: a sentence that starts with "Name[, Member] (dissenting|minority)" starts a Dissent block
   up to the end of the document. On train and dev it matches the gold Dissent exactly.

All 13 labels can be predicted (`decode.NEVER = []`).

## Files

| File | What it does |
|---|---|
| `backbone.py` | piece tokenizer, loads a Phase 1 model as a sentence encoder, path constants |
| `data.py` | loads documents, tokenizes, packs windows, counts the transition matrix |
| `model.py` | Assignment 2 LSTM, CRF, the role model |
| `peft_methods.py` | LoRMA+ layer and `add_peft` (LoRMA, LoRA, IA3) |
| `train.py` | loss, training loop, the five single runs (`RUNS`), saving probabilities |
| `distill.py` | teacher probabilities and the distilled Arch 2 student |
| `decode.py` | CRF prediction, Viterbi with counted transitions, macro-F1 |
| `rules.py` | Dissent rule |
| `predict.py` | the best ensemble and the submission zip |

## How to run

Needs Python 3, `torch`, `numpy`, `scikit-learn`, `peft`, and a GPU for training (I used a Kaggle T4). On
Kaggle, uninstall `torchao` before importing `peft`, or `peft` crashes.

| Constant | File | Default | Content |
|---|---|---|---|
| `DATA_DIR` | `data.py` | `data` | `CL_train.json`, `CL_dev.json`, `CL_test.json`, `IT_train.json`, `IT_dev.json`, `IT_test.json` |
| `BPE_PATH` | `backbone.py` | `tokenizer/legal_bpe_16384.json` | BPE file. The same folder needs `bpe_tokenizer.py` (a copy of `../phase1/tokenizer.py`) |
| `KIT_DIR` | `backbone.py` | `phase2_kit` | `arch1/`, `arch2/`, `arch3/`, each with `model_arch.py` (copy of `../phase1/archN_model.py`) and `model.pt`. Arch 3 also needs `metrics.json` |
| `ARCH4_MNTP_DIR` | `backbone.py` | `arch4_mntp` | `model_arch.py`, `model.pt`, `metrics.json`, `arch4_engram_compress.pt` |
| `ARCH4_MNTP_P3_DIR` | `backbone.py` | `arch4_mntp_p3` | same files |
| `OUT_DIR` | `train.py` | `runs` | outputs of all runs |
| `RUNS_DIR` | `predict.py` | `runs` | must be the same as `OUT_DIR` |
| `SUBMISSION_PATH` | `predict.py` | `submission_phase3.zip` | final submission |

Commands, in this order:

```
python train.py rr1          # Arch 4 + LoRA
python train.py x2           # Arch 4 (longer adaptation) + LoRA, 15 epochs
python distill.py teacher    # average of rr1 and x2 on the training documents
python distill.py student    # distilled Arch 2
python predict.py            # best submission
```

The other runs are independent: `python train.py rr2b`, `python train.py rr3`, `python train.py rr2`.

Each `train.py NAME` writes its probabilities (`NAME_*_probs*.npz`), `submission_NAME.zip` and
`NAME_model.pt` to `runs/`. `predict.py` only needs `rr1_test_probs.npz`, `x2_test_probs.npz`,
`arch2_kd_test_probs_centered.npz` and the training labels, so it runs on a CPU in seconds.

On my saved probabilities, `predict.py` gives the best submission (0.6345) again. Retraining is not
bit-identical. A rerun of the distilled Arch 2 got dev 0.503 instead of 0.511, so runs differ by about 0.01.

## Codabench submissions

| Submission | Dev macro-F1 | Codabench | Command |
|---|---|---|---|
| Arch 4 + LoRA (rr1) | 0.498 | 0.5139 | `python train.py rr1` |
| Arch 4 longer adaptation + LoRA (x2) | 0.462 | 0.5129 | `python train.py x2` |
| Arch 4 + LoRMA+ (rr2b) | 0.480 | 0.4739 | `python train.py rr2b` |
| Arch 3 + IA3 (rr3) | 0.390 | 0.4479 | `python train.py rr3` |
| Arch 1 + LoRMA+ (rr2) | 0.091 | 0.1103 | `python train.py rr2` |
| Distilled Arch 2 (arch2_kd) | 0.511 | 0.5354 | `python distill.py teacher`, then `python distill.py student` |
| Best: 0.6 distilled Arch 2 + 0.2 rr1 + 0.2 x2, shared transition Viterbi (beta 0.3), Dissent rule | 0.539 | 0.6345 | `python predict.py` |

The single-run submissions were made before I found the Dissent rule, so Dissent and PrecedentOverruled
were never predicted. To get them exactly, set `NEVER = [5, 8]` in `decode.py` and remove the
`dissent_rule` call in `decode.predict` before training. `distill.py` writes no zip. The student's
submission is its CRF Viterbi prediction on test (`decode.predict`).
