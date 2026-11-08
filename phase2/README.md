# Phase 2: Legal Named Entity Recognition (L-NER)

## Task and metric

The input is a long Indian court judgment. The output is every entity in it, as a character span (start, end) with one of 12 labels:
APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO.

The metric is strict macro-F1. A predicted span counts only if its start, end and label all match a gold span. F1 is computed per label and then averaged over the 12 labels, so rare labels count as much as common ones.

## Data and preprocessing

- Training data: 70 documents in two folds (`fold1`, `fold2`, 35 each). Test data: 35 documents (`fold3`, text only).
- 8,393 gold entities, no nested or overlapping spans. Median document length is 26.6k characters.
- Label counts range from 1,575 (DATE) to 184 (R.COUNSEL). WIT occurs in only 12 of the 70 documents.
- Annotation is fully consistent inside a document (no string carries two labels) but not across documents. This is why within-document propagation is the main post-processing step.

Preprocessing (`preprocess.py`):

- Tokenizer: my own Phase 1 BPE (16,384 tokens, from Assignment 1). The plain tokenizer attaches punctuation to words ("IPC.", "1994,"), so only 68.4% of entity boundaries fall on token edges.
- Piece tokenizer: every word is first split into letter runs, digit runs and single punctuation marks, and each piece is encoded with the same BPE. Only the last piece of a word gets the end-of-word marker. This raises boundaries on token edges to 99.3%. Every token keeps its character start and end, so predicted tags map back to exact offsets. 8,338 of the 8,393 gold spans survive the round trip from spans to tags and back.
- Tags: BIOES, 12 labels x {Begin, Inside, End, Single} plus Outside, so 49 tags.
- Windows: documents are cut into overlapping windows with stride half the window. The window is 1024 tokens for the decoders and 704 for the Arch 3 encoder (which also gets its `<cont>` prefix token). The first 128 tokens of the document (the header with parties, counsel and judges) are put before every window, followed by a separator. One ensemble member uses a 256-token header.
- Imbalance: class weights (inverse frequency)^0.5, capped at 10, and 2 copies of every window that contains APP, RESP, A.COUNSEL, R.COUNSEL or WIT.
- Phase 1 judgments used for continued pretraining are skipped if they match a Phase 2 train or test text.

## Models

All models start from my Phase 1 checkpoints. The Phase 1 model code lives in `../phase1` (`arch1_model.py` to `arch4_model.py`). The LM head is removed and a linear token head (49 tags, dropout 0.1) is put on top.

| Model | Size | What is specific about it |
|---|---|---|
| Arch 1 | 97.6M | GPT-2 style decoder from my A3 code: 12 layers x 768, 12-head attention, sinusoidal positions, LayerNorm, GELU FFN 3072 |
| Arch 2 | 61.6M | LLaMA style decoder: 12 layers x 640, grouped-query attention 10/2, RoPE, RMSNorm, SwiGLU 1706. Phase 2 uses the checkpoint after continued pretraining (300M + 329M tokens) |
| Arch 3 | 86.1M | Encoder-decoder: 10 encoder + 4 decoder layers x 640, RoPE, RMSNorm, SwiGLU 1712, 704 / 320 split. Phase 2 uses only the 10-layer encoder (59.8M) |
| Arch 4 | 139.1M | Deep decoder: 16 layers x 640, multi-head latent attention (latent 320, decoupled RoPE), QK-norm, sigmoid output gate, SwiGLU 1712, 42.9M hashed n-gram memory at layers 2 and 8 |

Decoders run causal by default. In bidirectional mode the causal mask is removed (`Backbone(..., bidirectional=True)`).

## Methodology

### Fine-tuning techniques per Phase 1 model

All parameter-efficient layers are written by hand in `lorma.py`. Rank 16 and alpha 16 are used on every linear layer of every block; the backbone is frozen and the token head is always trained.

| Model | Technique | Trainable | Settings |
|---|---|---|---|
| Arch 1 | LoRMA+ (compulsory): `W' = (I + (alpha / r) B A) W0`, computed as `h + s B(A h)` with `h = W0 x`; A starts at zero | 2.7M | body LR 5e-4, batch 4, up to 10 epochs |
| Arch 2 | Full fine-tuning | 61.6M | LR 1e-4, batch 8, up to 8 epochs |
| Arch 2 | Knowledge distillation student of the fine-tuned Arch 4 (139.1M) | 61.6M | LR 1e-4, batch 8, 4 epochs, no CV (the teacher saw both folds) |
| Arch 3 | Full fine-tuning of the encoder | 59.8M | LR 1e-4, batch 8, up to 8 epochs |
| Arch 4 | Full fine-tuning (2 seeds, and once on the continued-pretraining backbone) | 139.1M | LR 1e-4, batch 4, up to 8 epochs |
| Arch 4 | LoRA, `h = W0 x + (alpha / r) B A x`, B starts at zero, on the bidirectional backbone (2 seeds) | 3.7M | body LR 5e-4, batch 4, up to 10 epochs (seed 0), 6 epochs (seed 1) |

Common settings (`train.py`): AdamW with weight decay 0.01, head LR 1e-3, 6% linear warmup then linear decay, gradient clipping 1.0, backbone dropout 0.1, label smoothing 0.1, fp16 autocast with a gradient scaler on GPU.

Training protocol: with cross-validation, the model is trained on fold1 and scored on fold2, and the reverse. Each fold keeps its best epoch. The propagation setting is then chosen on the pooled CV predictions, and a final model is trained on both folds for the mean best epoch (rounded). Its test log-probabilities are saved for the ensemble.

### Backbone adaptation of Arch 4 (`adapt.py`)

- Continued pretraining (`tapt`): causal LM training of the Phase 1 Arch 4 on 25M piece-tokenized tokens of Phase 1 judgments (60% Supreme Court, 40% High Courts), 1,500 updates of 16 sequences x 1024 tokens, LR 1e-4. LM loss on Phase 2 text fell from 3.152 to 2.320. Its NER CV alone was 0.5137 (original Arch 4: 0.5132), but it helps as an ensemble member.
- Bidirectional adaptation (`mntp`, LLM2Vec style): starts from the continued-pretraining model, removes the causal mask and trains for 800 updates (16 x 1024 tokens each, about 13M tokens) with masked next-token prediction. 20% of tokens are replaced by a mask token and each one is predicted from the output one position before it. This took 36 minutes on a T4. Two seeds were adapted.
- LoRA NER on the bidirectional backbone raised CV from 0.522 (LoRA on the causal Arch 4) to 0.580.

### Distillation (`distill.py`)

The student is Arch 2 (61.6M). The teacher is the fully fine-tuned Arch 4 run `arch4_full_h128` (139.1M), frozen. The loss is `0.5 * CE(gold) + 0.5 * T^2 * KL(teacher || student)` with temperature T = 2. The student scored 0.5425 on Codabench, against 0.5407 for the same Arch 2 trained on gold labels only, and 91% of the teacher's 0.5957 with 44% of its parameters.

### Decoding and rules (`decode.py`, `rules.py`)

1. Window merge: each token keeps the window where it has the most context (both sides for bidirectional models, the left side for causal ones).
2. Viterbi decoding restricted to valid BIOES sequences, then edge whitespace is trimmed.
3. Within-document propagation: one label per string, chosen by summed confidence, and every free word-bounded occurrence of a predicted string is tagged. The final setting is minimum length 6 and minimum confidence 0.5, case-insensitive for APP.
4. Boundary repair: snap span edges to whole words (all labels except PREC), trim trailing punctuation and bracketed footnote numbers, drop a final full stop from PREC unless it closes an abbreviation.
5. Document rules, in this order: header party linking for APP and PW-marked witness names; precedent repair (footnotes, brackets, case name plus citation, trailing "case"); pruning of body APP mentions that match no header party and of WIT mentions never next to a PW marker; label-specific boundary repair (DATE, CASENO, STAT, COURT, CASENO prefix); counsel side from the explicit pattern after the name ("X, learned counsel for the appellants"); the respondent list parsed from the header, linked into the body, and exact body matches relabelled to RESP.

### Ensembling (`predict.py`)

- Causal ensemble (weights 2 : 1 : 1 : 1): Arch 4 full fine-tuning seed 0, Arch 4 seed 1, Arch 2 with a 256-token header (seed 1), and Arch 4 on the continued-pretraining backbone.
- Bidirectional model: the LoRA model on the bidirectional Arch 4, seed 0.
- The causal ensemble and the bidirectional model are mixed 5 : 3. All averages are over token probabilities.
- The mixed log-probabilities then go through Viterbi decoding, propagation, boundary repair and the full rule chain above. No per-label thresholds are used.

## Files

| File | Contents |
|---|---|
| `preprocess.py` | Label and tag lists, the BPE merges loader, the piece tokenizer with character offsets, spans to BIOES tags and back, data loading, windows, class weights, the Phase 2 duplicate check |
| `model.py` | Loads a Phase 1 model without its LM head (optionally bidirectional, Arch 3 encoder only), the token classifier, sliding-window inference with the header prefix |
| `lorma.py` | Hand-written LoRA and LoRMA+ layers and `apply_method`, which freezes the backbone and wraps every linear layer |
| `decode.py` | Constrained BIOES Viterbi, span extraction, within-document propagation, boundary repair, the strict macro-F1 scorer |
| `rules.py` | Document-level rules: party and witness linking, precedent repair, role pruning, label-specific repair, counsel side, header respondent parser |
| `train.py` | NER fine-tuning with cross-validation and a final model, the propagation search, and the configurations of all runs (`RUNS`) |
| `distill.py` | Arch 2 student distilled from the fine-tuned Arch 4 |
| `adapt.py` | Continued pretraining (`tapt`) and bidirectional masked next-token adaptation (`mntp`) of Arch 4 |
| `predict.py` | The final submission: ensemble, propagation, repair, rules, zip writer |

## How to run

### Dependencies

Python 3, PyTorch, NumPy and ijson (`pip install torch numpy ijson`). Training was done on a Kaggle T4 GPU. `adapt.py` needs a CUDA GPU. `predict.py` only reads saved log-probabilities and runs on a CPU.

### Inputs and path constants

Run every command from this folder, or change these constants:

- `preprocess.DATA_DIR = "data"`: `fold1-text.json`, `fold1-gold.json`, `fold2-text.json`, `fold2-gold.json`, `fold3-text.json`.
- `preprocess.TOKENIZER_PATH = "legal_bpe_16384.json"`: the tokenizer file written by `../phase1/train_tokenizer.py`.
- `model.MODEL_DIRS`: `models/arch1` to `models/arch4`. Each folder holds `model_arch.py` (a copy of the matching `../phase1/archN_model.py`) and the Phase 1 `model.pt`. `models/arch3` and `models/arch4` also need the Phase 1 `metrics.json`, and `models/arch4` needs `arch4_engram_compress.pt`. `models/arch2` holds the Arch 2 checkpoint after continued pretraining.
- `adapt.CORPUS_FILES`: the Phase 1 files `data/phase1/train_supreme_court.json` and `data/phase1/train_high_courts.json`.
- `train.OUT_DIR = "runs"`, `train.TAPT_DIR = "runs/arch4_tapt"`, `train.MNTP_DIRS`: where runs and adapted backbones are written.
- `predict.RUNS_DIR = "runs"`, `predict.OUT_PATH = "submission_best.zip"`.

### Order of commands

```
python train.py arch1_lorma_h128
python train.py arch2_full_h128
python train.py arch3_full_h128
python train.py arch4_full_h128
python train.py arch4_full_h128_s1
python train.py arch2_full_h256_s1
python distill.py                      # needs runs/arch4_full_h128/model_final.pt
python adapt.py tapt                   # writes runs/arch4_tapt
python train.py arch4_tapt_full_h128
python adapt.py mntp 0                 # writes runs/arch4_mntp_s0
python adapt.py mntp 1                 # writes runs/arch4_mntp_s1
python train.py arch4_mntp_lora_h128_bidir
python train.py arch4_mntp_lora_h128_s1_bidir
python predict.py                      # writes submission_best.zip
```

Each `train.py` run writes `runs/<run name>/` with `cv_logprobs_fold1.npz` and `cv_logprobs_fold2.npz` (with cross-validation), `test_logprobs.npz`, `model_final.pt` (fp16), `final_results.json` (final epochs, propagation setting, history) and `submission_<run name>.zip`. It prints the loss and fold F1 per epoch and the pooled CV score with and without propagation. `distill.py` writes `runs/arch2_full_h128_kd/`. `adapt.py` writes `model.pt` and `adapt_log.json` (tokens, updates, loss before and after) into the adapted backbone folder. Every zip holds one `prediction.json` that maps each test document id to a list of `[start, end, label]`.

## Codabench submissions covered by this code

Single models: the zip written by the command. Ensemble: `python predict.py`.

| Submission | Codabench | Command |
|---|---|---|
| Arch 1, LoRMA+ | 0.3351 | `python train.py arch1_lorma_h128` |
| Arch 3 encoder, full fine-tuning | 0.3640 | `python train.py arch3_full_h128` |
| Arch 2, full fine-tuning | 0.5407 | `python train.py arch2_full_h128` |
| Arch 2, distilled from Arch 4 | 0.5425 | `python distill.py` |
| Arch 4, full fine-tuning | 0.5957 | `python train.py arch4_full_h128` |
| Bidirectional Arch 4 + LoRA, seed 0 | 0.6261 | `python train.py arch4_mntp_lora_h128_bidir` |
| Bidirectional Arch 4 + LoRA, seed 1 | 0.6503 | `python train.py arch4_mntp_lora_h128_s1_bidir` |
| **Causal ensemble + bidirectional seed 0 + rules (best)** | **0.6872** | `python predict.py` |

My best submission is the ensemble (0.6872). Its members are Arch 4 full fine-tuning seed 0 (weight 2), Arch 4 full fine-tuning seed 1 (weight 1), Arch 2 with a 256-token header (weight 1) and Arch 4 on the continued-pretraining backbone (weight 1). This causal ensemble is mixed 5 : 3 with the bidirectional Arch 4 LoRA model (seed 0), and then the full rule chain is applied. The four required submissions of the phase are Arch 1 LoRMA+ (0.3351), Arch 2 distilled (0.5425), Arch 3 full fine-tuning (0.3640) and bidirectional Arch 4 LoRA (0.6503).
