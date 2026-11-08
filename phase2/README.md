# Phase 2: Legal Named Entity Recognition (L-NER)

## Task

The input is a long Indian court judgment. The output is every entity in it, as a character span (start, end) with one of 12 labels:
APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO.
The metric is strict macro-F1: a span counts only if start, end and label all match, and every label counts equally.

## Data

- Training: 70 documents in two folds (`fold1`, `fold2`). Test: 35 documents (`fold3`, text only).
- 8,393 gold entities, none nested. WIT appears in only 12 of the 70 documents.
- Labels are consistent inside a document, so I spread predicted labels inside each document.

## Preprocessing (`preprocess.py`)

- The plain Phase 1 BPE glues punctuation to words ("IPC.", "1994,"), so only 68.4% of entity edges fall on a token edge.
- My piece tokenizer first splits words into letter runs, digit runs and punctuation, then encodes each piece with the same BPE. This raises it to 99.3%.
- Tags: BIOES, 12 labels x 4 plus Outside, so 49 tags.
- Windows overlap by half: 1024 tokens for the decoders, 704 for the Arch 3 encoder. The first 128 tokens of the document (parties, counsel, judges) are put before every window. One ensemble member uses a 256-token header.
- Rare labels: class weights of (inverse frequency)^0.5, capped at 10, and windows with APP, RESP, A.COUNSEL, R.COUNSEL or WIT are used twice.

## Models

All models start from my Phase 1 checkpoints (code in `../phase1`). I replace the LM head with a linear token head (49 tags, dropout 0.1).

| Model | Size | Notes |
|---|---|---|
| Arch 1 | 97.6M | GPT-2 style decoder from my A3 code |
| Arch 2 | 61.6M | LLaMA style decoder, checkpoint after continued pretraining (300M + 329M tokens) |
| Arch 3 | 86.1M | encoder-decoder. Phase 2 uses only the 10-layer encoder (59.8M) |
| Arch 4 | 139.1M | latent-attention decoder with 42.9M n-gram memory |

## Fine-tuning

I wrote the LoRA and LoRMA+ layers by hand in `lorma.py`, with rank 16 and alpha 16 on every linear layer. The backbone is frozen and the head is trained.
LoRMA+ multiplies each weight by (I + (alpha / r) B A). LoRA adds (alpha / r) B A to it.

| Model | Technique | Trainable | Settings |
|---|---|---|---|
| Arch 1 | LoRMA+ (compulsory) | 2.7M | body LR 5e-4, batch 4, up to 10 epochs |
| Arch 2 | Full fine-tuning | 61.6M | LR 1e-4, batch 8, up to 8 epochs |
| Arch 2 | Distillation student of Arch 4 | 61.6M | LR 1e-4, batch 8, 4 epochs, no CV |
| Arch 3 | Full fine-tuning of the encoder | 59.8M | LR 1e-4, batch 8, up to 8 epochs |
| Arch 4 | Full fine-tuning (2 seeds, plus one on the continued-pretraining backbone) | 139.1M | LR 1e-4, batch 4, up to 8 epochs |
| Arch 4 | LoRA on the bidirectional backbone (2 seeds) | 3.7M | body LR 5e-4, batch 4, up to 10 epochs (seed 0), 6 epochs (seed 1) |

Common settings (`train.py`): AdamW with weight decay 0.01, head LR 1e-3, 6% warmup then linear decay, gradient clip 1.0, dropout 0.1, label smoothing 0.1, fp16.

Each run trains on one fold and tests on the other, both ways, to pick the best epoch and the propagation setting. Then a final model is trained on both folds and its test log-probabilities are saved.

## Adapting Arch 4 (`adapt.py`)

- `tapt`: continued causal LM training on 25M tokens of Phase 1 judgments (1,500 updates, LR 1e-4). LM loss on Phase 2 text fell from 3.152 to 2.320.
- `mntp`: removes the causal mask and trains for 800 updates with masked next-token prediction (20% of tokens masked). I made two seeds.
- LoRA on this bidirectional backbone raised CV from 0.522 to 0.580.

## Distillation (`distill.py`)

The student is Arch 2. The teacher is the frozen fine-tuned Arch 4 run `arch4_full_h128`.
The loss is half cross entropy on gold tags and half KL to the teacher, with temperature 2.
The student scored 0.5425, against 0.5407 for Arch 2 on gold labels only.

## Decoding and rules (`decode.py`, `rules.py`)

1. Each token keeps the window where it has the most context.
2. Viterbi decoding that only allows valid BIOES sequences.
3. Propagation: each string gets one label in a document, and every other copy of it is tagged too (minimum length 6, minimum confidence 0.5).
4. Boundary repair: snap spans to whole words, trim punctuation.
5. Document rules: link header parties and witnesses, repair precedents, remove unlinked APP and WIT mentions, fix DATE, CASENO, STAT and COURT edges, find the counsel side, and relabel header respondents in the body as RESP.

## Ensemble (`predict.py`)

- Causal ensemble, weights 2 : 1 : 1 : 1: Arch 4 full fine-tuning seed 0, Arch 4 seed 1, Arch 2 with a 256-token header, and Arch 4 on the continued-pretraining backbone.
- This is mixed 5 : 3 with the bidirectional Arch 4 LoRA model (seed 0), by averaging token probabilities.
- Then decoding and rules run as above.

## Files

| File | Contents |
|---|---|
| `preprocess.py` | labels and tags, piece tokenizer with offsets, spans to tags and back, windows, class weights |
| `model.py` | loads a Phase 1 model without its LM head, the token classifier, sliding-window inference |
| `lorma.py` | LoRA and LoRMA+ layers and `apply_method` |
| `decode.py` | BIOES Viterbi, spans, propagation, boundary repair, the macro-F1 scorer |
| `rules.py` | document rules |
| `train.py` | fine-tuning with cross-validation, the propagation search, and all run configs (`RUNS`) |
| `distill.py` | Arch 2 student distilled from Arch 4 |
| `adapt.py` | `tapt` and `mntp` adaptation of Arch 4 |
| `predict.py` | the final ensemble submission |

## How to run

Needs Python 3, PyTorch, NumPy and ijson (`pip install torch numpy ijson`). I trained on a Kaggle T4. `adapt.py` needs a CUDA GPU. `predict.py` runs on a CPU.

Path constants (run from this folder, or change them):

- `preprocess.DATA_DIR = "data"`: `fold1-text.json`, `fold1-gold.json`, `fold2-text.json`, `fold2-gold.json`, `fold3-text.json`.
- `preprocess.TOKENIZER_PATH = "legal_bpe_16384.json"`, written by `../phase1/train_tokenizer.py`.
- `model.MODEL_DIRS`: `models/arch1` to `models/arch4`. Each holds `model_arch.py` (a copy of `../phase1/archN_model.py`) and the Phase 1 `model.pt`. `models/arch3` and `models/arch4` also need the Phase 1 `metrics.json`, and `models/arch4` needs `arch4_engram_compress.pt`. `models/arch2` holds the continued-pretraining checkpoint.
- `adapt.CORPUS_FILES`: `data/phase1/train_supreme_court.json` and `data/phase1/train_high_courts.json`.
- `train.OUT_DIR = "runs"`, `train.TAPT_DIR = "runs/arch4_tapt"`, `train.MNTP_DIRS`.
- `predict.RUNS_DIR = "runs"`, `predict.OUT_PATH = "submission_best.zip"`.

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

- Each `train.py` run writes `runs/<run name>/` with the CV and test log-probabilities (`.npz`), `model_final.pt`, `final_results.json` and `submission_<run name>.zip`.
- `distill.py` writes `runs/arch2_full_h128_kd/`.
- Every zip holds one `prediction.json` that maps each test document id to a list of `[start, end, label]`.

## Codabench submissions

| Submission | Codabench | Command |
|---|---|---|
| Arch 1, LoRMA+ | 0.3351 | `python train.py arch1_lorma_h128` |
| Arch 3 encoder, full fine-tuning | 0.3640 | `python train.py arch3_full_h128` |
| Arch 2, full fine-tuning | 0.5407 | `python train.py arch2_full_h128` |
| Arch 2, distilled from Arch 4 | 0.5425 | `python distill.py` |
| Arch 4, full fine-tuning | 0.5957 | `python train.py arch4_full_h128` |
| Bidirectional Arch 4 + LoRA, seed 0 | 0.6261 | `python train.py arch4_mntp_lora_h128_bidir` |
| Bidirectional Arch 4 + LoRA, seed 1 | 0.6503 | `python train.py arch4_mntp_lora_h128_s1_bidir` |
| Ensemble + rules (best) | 0.6872 | `python predict.py` |

The four required submissions are Arch 1 LoRMA+ (0.3351), Arch 2 distilled (0.5425), Arch 3 full fine-tuning (0.3640) and bidirectional Arch 4 LoRA (0.6503).
