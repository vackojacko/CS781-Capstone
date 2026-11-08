# Phase 3: Rhetorical role prediction

## Task and metric

Every sentence of a legal judgment gets one of 13 rhetorical roles:
Fact, Issue, ArgumentPetitioner, ArgumentRespondent, Statute, Dissent, PrecedentReliedUpon,
PrecedentNotReliedUpon, PrecedentOverruled, RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt, None.

The output is `{doc_id: [one label per sentence]}`, in the order of the input sentences.

The metric is macro-F1 over all 13 labels, with the sentences of all test documents pooled
(`f1_score(..., labels=<all 13>, average="macro", zero_division=0)`, as in the official `evaluate_phase3.py`).
A label that is missing from the test gold scores 0 for everyone. `decode.macro_f1` computes the same score.

## Data and preprocessing

- Two domains: Competition Law (CL) and Income Tax (IT), 40 / 5 / 5 documents per domain (train / dev / test).
- Train and dev files are lists of `{"id", "text", "labels"}`, where `text` is the list of sentences.
  Test files are `{id: [sentences]}` with no labels.
- CL is blocky (94% of sentences share the next sentence's label, 75% in IT).
  CL is dominated by Fact (28%) and RatioOfTheDecision (27%). IT has three times more Statute, Issue and None.
- Dissent occurs in two CL training documents, always as a closing block that starts with a
  "Member (dissenting)" header.
- Both domains are loaded together and every document keeps its domain (`data.load_docs`).
  One model serves both domains, with the domain as an input feature.
- Tokenizer: my Assignment 1 byte-level BPE with end-of-word markers, 16,384 tokens, trained on the legal corpus.
  As in Phase 2, every word is first split into letter, digit and punctuation pieces, and only the last
  piece of a word gets the end-of-word marker (`backbone.PieceTokenizer`). Text is not lowercased.
- The provided sentence split is kept. Each sentence is cut to 128 tokens
  (`data.MAX_SENT_TOKENS`); an empty sentence becomes the separator token (id 3).
- Sentences are packed whole into windows of 1024 tokens (703 for the Arch 3 encoder, whose window of 704
  starts with one special token) and go through the backbone together (`data.make_windows`).

## Models

The Phase 1 model code lives in `../phase1` (`arch1_model.py` to `arch4_model.py`).
All four were trained from scratch on the Phase 1 corpus with the tokenizer above.

| | Arch 1 | Arch 2 | Arch 3 | Arch 4 |
|---|---|---|---|---|
| Type | GPT-2 style decoder | LLaMA style decoder | encoder-decoder | latent-attention decoder |
| Layers x width | 12 x 768 | 12 x 640 | (10 enc + 4 dec) x 640 | 16 x 640 |
| Attention | MHA, 12 heads | GQA 10/2 | MHA + cross-attention | MLA, latent 320 |
| Positions | sinusoidal | RoPE | RoPE | decoupled RoPE |
| Norm | LayerNorm | RMSNorm | RMSNorm | RMSNorm, QK-norm |
| FFN | GELU 3072 | SwiGLU 1706 | SwiGLU 1712 | SwiGLU 1712 |
| Parameters | 97.6M | 61.6M | 86.1M | 139.1M |

What is specific about each one:

- Arch 1 is my Assignment 3 decoder scaled to 97.6M, with fixed sinusoidal positions.
  In this phase it runs in fp32, because it did not learn in fp16.
- Arch 2 is a compact decoder: the 10/2 grouped-query ratio removes 80% of the key and value parameters.
  It is the distillation student of this phase.
- Arch 3 has 704 encoder and 320 decoder positions and was pretrained with a mix of continuation and
  span corruption. Here only its encoder is used (the decoder blocks are dropped), so it is bidirectional,
  and every window starts with its continuation token (id 16384).
- Arch 4 uses multi-head latent attention, QK normalisation, a sigmoid output gate and a hashed
  bigram/trigram memory (42.9M parameters) at layers 2 and 8. In this phase it is used in its bidirectional
  form: in Phase 2 its causal mask was removed and it was trained for 800 updates with masked next-token
  prediction (20% of tokens masked, each predicted from the position before it). Two versions are used:
  - `arch4_mntp`: the Phase 2 adaptation.
  - `arch4_mntp_p3`: the same model adapted longer, with the Phase 3 training documents added to the
    adaptation text. This adaptation is not part of this folder; the code only loads the result.

### Rhetorical role model (`model.py`)

The MTL BiLSTM-CRF of Malik et al. 2022, with my backbones in place of BERT:

1. A sentence vector is the mean of the backbone states of its own tokens.
2. Role branch: a BiLSTM (256 per direction) over the sentence vectors of the document, plus relative
   position, position squared and a CL / IT one-hot feature.
3. Shift branch: a BiLSTM (256 per direction) over pairs [v_i ; v_i+1], with a linear output that predicts
   whether the label changes after sentence i.
4. A linear emission layer on [role ; shift] gives 13 scores per sentence, and a CRF decodes the sequence.
5. Dropout 0.2 on the sentence vectors and on the features; dropout 0.1 inside the backbone.

The LSTM cell and the recurrent encoder are my Assignment 2 code.

## Methodology

### Fine-tuning technique per Phase 1 model

| Run | Phase 1 model | Technique | Trainable backbone params | Body LR | Epochs |
|---|---|---|---|---|---|
| rr1 | Arch 4 (`arch4_mntp`) | LoRA r = 16 | 3.62M | 5e-4 | 6 |
| x2 | Arch 4 (`arch4_mntp_p3`) | LoRA r = 16, backbone frozen for the first 2 epochs | 3.62M | 5e-4 | 15 |
| rr2b | Arch 4 (`arch4_mntp`) | LoRMA+ r = 16, backbone frozen for the first 2 epochs | 4.15M | 5e-4 | 6 |
| rr3 | Arch 3 (encoder) | IA3 | 0.03M | 3e-3 | 6 |
| rr2 | Arch 1 | LoRMA+ r = 16, fp32, backbone frozen for the first 2 epochs | 2.65M | 5e-4 | 6 |
| arch2_kd | Arch 2 | LoRA r = 16, distilled from rr1 + x2 | 2.14M | 5e-4 | 6 |

- LoRA (`peft` library): r = 16, alpha = 16, dropout 0.05, on every linear layer of the attention and
  feed-forward blocks.
- LoRMA+ (my Phase 2 layer, `peft_methods.LoRMALinear`): W' = (I + (alpha / r) B A) W0 on every linear
  layer of every block, r = 16, alpha = 16. A starts at zero, so the layer starts as the frozen W0.
  LoRMA did not learn when trained with the head from the start, so its backbone is frozen for the
  first 2 epochs.
- IA3 (`peft` library): learned scaling of the keys, the values and the input of the feed-forward
  down projection.

### Training (`train.py`)

- Loss per training piece: 0.4 x CRF negative log-likelihood (divided by the number of sentences)
  + 0.3 x class-weighted cross-entropy + 0.6 x binary cross-entropy of the label-shift branch.
- Class weights: square root of inverse label frequency on the training documents, divided by their mean,
  capped at 10. Every label gets a weight above 0.
- Training pieces: at most 64 consecutive sentences and 4,096 tokens, one piece per update.
- AdamW, weight decay 0.01, head LR 1e-3, 5% linear warmup then linear decay to 0, gradient clipping 1.0.
- fp16 autocast with a gradient scaler on GPU (Arch 1 in fp32). Seed 0.
- Training uses only the 80 training documents. After every epoch the model is scored on the 10 dev documents,
  and the epoch with the best dev macro-F1 is kept.

### Distillation (`distill.py`)

- Teacher: the average of the sentence probabilities (CRF marginals) of rr1 and x2 on the training documents.
- Student: Arch 2 with LoRA, same model and settings as above, 6 epochs.
- Loss: 0.5 x the training loss above + 0.5 x KL(teacher || student), with the student scores softened
  at T = 2 and the KL multiplied by T^2.

### Overlapping windows

Training uses windows with no overlap. At prediction, `save_run` also stores probabilities with overlapping
windows ("centered"): each window keeps the vectors of its middle 512 tokens of sentences and uses the rest
as context on both sides. Only the distilled student uses these probabilities in the final ensemble.

### Ensemble, decoding and rule (`predict.py`, `decode.py`, `rules.py`)

1. Per document, the row-normalised sentence probabilities are averaged with weights
   0.6 (distilled Arch 2, overlapping windows) + 0.2 (rr1) + 0.2 (x2).
2. Viterbi decoding with a label-transition matrix counted on the training documents
   (label bigram counts with add-one smoothing, rows sum to 1). The score of a path is the sum of the log
   probabilities plus 0.3 x the log transition probabilities (`BETA = 0.3`). One shared matrix is used for
   both domains.
3. Dissent rule: a sentence that starts with "Name[, Member] (dissenting|minority)" begins a Dissent block
   that runs to the end of the document. On train and dev it reproduces the gold Dissent exactly and fires
   nowhere else.

All 13 labels can be predicted (`decode.NEVER = []`).

## Files

| File | Content |
|---|---|
| `backbone.py` | piece tokenizer; loads a Phase 1 model without its LM head as a sentence encoder; makes Arch 4 bidirectional; path constants for the tokenizer and the backbones |
| `data.py` | loads CL + IT documents, tokenizes sentences, packs windows, counts the label-transition matrix |
| `model.py` | Assignment 2 LSTM cell and recurrent encoder, CRF (loss, Viterbi, marginals), the role model, `build_model` |
| `peft_methods.py` | LoRMA+ layer and `add_peft` (LoRMA, LoRA, IA3) |
| `train.py` | loss, class weights, training loop with dev selection, the five single runs (`RUNS`), saving probabilities |
| `distill.py` | builds the teacher probabilities and trains the distilled Arch 2 student |
| `decode.py` | prediction with CRF marginals, Viterbi with counted transitions, macro-F1 |
| `rules.py` | Dissent rule |
| `predict.py` | saving and loading probabilities, the best ensemble, writing the submission zip |

## How to run

### Dependencies

Python 3, `torch`, `numpy`, `scikit-learn`, `peft`. On Kaggle, uninstall `torchao` before importing `peft`
(the old version in the Kaggle image makes `peft` crash). A GPU (I used a Kaggle T4) is needed for training.

### Inputs and path constants

| Constant | File | Default | Content |
|---|---|---|---|
| `DATA_DIR` | `data.py` | `data` | `CL_train.json`, `CL_dev.json`, `CL_test.json`, `IT_train.json`, `IT_dev.json`, `IT_test.json` |
| `BPE_PATH` | `backbone.py` | `tokenizer/legal_bpe_16384.json` | BPE file; the same folder must contain `bpe_tokenizer.py` (a copy of `../phase1/tokenizer.py`) |
| `KIT_DIR` | `backbone.py` | `phase2_kit` | folders `arch1/`, `arch2/`, `arch3/`, each with `model_arch.py` (a copy of `../phase1/archN_model.py`) and `model.pt`; Arch 3 also needs `metrics.json` |
| `ARCH4_MNTP_DIR` | `backbone.py` | `arch4_mntp` | bidirectional Arch 4 from Phase 2: `model_arch.py`, `model.pt`, `metrics.json`, `arch4_engram_compress.pt` |
| `ARCH4_MNTP_P3_DIR` | `backbone.py` | `arch4_mntp_p3` | the longer adapted Arch 4, same files |
| `OUT_DIR` | `train.py` | `runs` | outputs of all runs |
| `RUNS_DIR` | `predict.py` | `runs` | must be the same folder as `OUT_DIR` |
| `SUBMISSION_PATH` | `predict.py` | `submission_phase3.zip` | final submission |

### Commands (in this order)

```
python train.py rr1          # Arch 4 + LoRA
python train.py x2           # Arch 4 (longer adaptation) + LoRA, 15 epochs
python distill.py teacher    # average of rr1 and x2 on the training documents
python distill.py student    # distilled Arch 2
python predict.py            # best submission
```

The other single runs are independent of these: `python train.py rr2b`, `python train.py rr3`, `python train.py rr2`.

Outputs in `runs/`:

- `train.py NAME`: `NAME_{train,dev,test}_probs.npz` (no overlap), `NAME_{dev,test}_probs_centered.npz` and
  `NAME_{dev,test}_probs_left.npz` (overlapping windows), `submission_NAME.zip` (CRF Viterbi, Dissent rule),
  `NAME_model.pt` (head and adapter weights).
- `distill.py teacher`: `teacher_rr1_x2_train_probs.npz`.
- `distill.py student`: `arch2_kd_*_probs*.npz` as above and `arch2_kd_model.pt`.
- `predict.py`: `submission_phase3.zip` with `prediction.json` inside. It needs only the saved probabilities
  `rr1_test_probs.npz`, `x2_test_probs.npz`, `arch2_kd_test_probs_centered.npz` and the training labels,
  so it runs on a CPU in seconds.

`predict.py` on the saved probabilities of the original runs reproduces the best submission (0.6345).
Retraining on a GPU is not bit-identical: a rerun of the distilled Arch 2 recipe gave dev 0.503 against 0.511
in the original run, so run-to-run variation is about 0.01.

## Codabench submissions covered by this code

| Submission | Dev macro-F1 | Codabench | Command / config |
|---|---|---|---|
| Arch 4 + LoRA (rr1) | 0.498 | 0.5139 | `python train.py rr1` |
| Arch 4 longer adaptation + LoRA (x2) | 0.462 | 0.5129 | `python train.py x2` |
| Arch 4 + LoRMA+ (rr2b) | 0.480 | 0.4739 | `python train.py rr2b` |
| Arch 3 + IA3 (rr3) | 0.390 | 0.4479 | `python train.py rr3` |
| Arch 1 + LoRMA+ (rr2) | 0.091 | 0.1103 | `python train.py rr2` |
| Distilled Arch 2 (arch2_kd) | 0.511 | 0.5354 | `python distill.py teacher`, then `python distill.py student`; CRF Viterbi predictions of the student |
| **Best: 0.6 distilled Arch 2 (overlapping windows) + 0.2 rr1 + 0.2 x2, transition Viterbi (beta 0.3), Dissent rule** | **0.539** | **0.6345** | `python predict.py` |

The single-run submissions were made before the Dissent rule was found, with Dissent and PrecedentOverruled
never predicted. To reproduce them exactly, set `NEVER = [5, 8]` in `decode.py` and drop the `dissent_rule`
call in `decode.predict` before training. `distill.py` saves the student's probabilities and model but writes
no zip; its submission is the student's CRF Viterbi prediction on the test documents (`decode.predict`).
