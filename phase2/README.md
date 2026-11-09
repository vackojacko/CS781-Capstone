# Phase 2: Legal Named Entity Recognition (L-NER)

## Task

The input is a long Indian court judgment. The output is every entity in it, given as a character span (start, end) with one of 12 labels: APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO. The metric is strict macro-F1: a span counts only if its start, end and label are all right, and every label counts equally.

## My final system

- I split each word into letter, digit and punctuation pieces before my Phase 1 BPE, so that entity edges line up with token edges.
- Each Phase 1 model gets a small tagging head that gives every token a BIOES tag. Long documents are read in overlapping windows, and the start of the document (parties, counsel, judges) is added to every window.
- The final answer averages the token probabilities of five models: four causal ones mixed 2:1:1:1, then mixed 5:3 with the bidirectional Arch 4 LoRA model (seed 0).
- A Viterbi decoder keeps only valid tag sequences. If a string is tagged once in a document, its other copies get the same label.
- Simple document rules then fix span edges and party, counsel and witness labels.
- Best Codabench score: 0.6872.

## Models I tried

| Phase 1 model | Technique | Codabench |
|---|---|---|
| Arch 1 | LoRMA+ | 0.3351 |
| Arch 3 (encoder only) | Full fine-tuning | 0.3640 |
| Arch 2 | Full fine-tuning | 0.5407 |
| Arch 4 | Full fine-tuning | 0.5957 |
| Arch 4, made bidirectional | LoRA, seed 0 | 0.6261 |
| Arch 4, made bidirectional | LoRA, seed 1 | 0.6503 |
| Arch 2 | Distillation from the fine-tuned Arch 4 | 0.5425 |

I wrote the LoRA and LoRMA+ layers myself (rank 16). To make Arch 4 bidirectional, I first trained it a bit more on Phase 1 judgments, then removed the causal mask and trained it to predict masked tokens. The distilled Arch 2 learns half from the gold tags and half from the Arch 4 predictions.

## Files

| File | What it does |
|---|---|
| `preprocess.py` | labels, piece tokenizer, spans to tags and back, windows, class weights |
| `model.py` | loads a Phase 1 model and adds the tagging head |
| `lorma.py` | LoRA and LoRMA+ layers |
| `decode.py` | Viterbi decoding, label spreading, edge fixes, the macro-F1 score |
| `rules.py` | document rules |
| `train.py` | fine-tuning, cross-validation and the settings of every run |
| `distill.py` | Arch 2 distilled from Arch 4 |
| `adapt.py` | extra training of Arch 4 and the bidirectional version |
| `predict.py` | the final ensemble and the best submission |

## How to run

Dependencies: Python 3, PyTorch, NumPy and ijson (`pip install torch numpy ijson`). I trained on a Kaggle T4 GPU. `predict.py` runs on a CPU.

Path constants at the top of the files (the defaults work if you run from this folder):

- `DATA_DIR = "data"` in `preprocess.py`: `fold1-text.json`, `fold1-gold.json`, `fold2-text.json`, `fold2-gold.json`, `fold3-text.json`.
- `TOKENIZER_PATH = "legal_bpe_16384.json"` in `preprocess.py`, made by `../phase1/train_tokenizer.py`.
- `MODEL_DIRS` in `model.py`: `models/arch1` to `models/arch4`. Each folder has `model_arch.py` (a copy of `../phase1/archN_model.py`) and the Phase 1 `model.pt`. `models/arch3` and `models/arch4` also need `metrics.json`, and `models/arch4` needs `arch4_engram_compress.pt`. `models/arch2` holds the Arch 2 checkpoint after its extra pretraining.
- `CORPUS_FILES` in `adapt.py`: `data/phase1/train_supreme_court.json` and `data/phase1/train_high_courts.json`.
- `OUT_DIR = "runs"`, `TAPT_DIR` and `MNTP_DIRS` in `train.py`.
- `RUNS_DIR = "runs"` and `OUT_PATH = "submission_best.zip"` in `predict.py`.

Commands, in this order:

```
python train.py arch1_lorma_h128
python train.py arch2_full_h128
python train.py arch3_full_h128
python train.py arch4_full_h128
python train.py arch4_full_h128_s1
python train.py arch2_full_h256_s1
python distill.py
python adapt.py tapt
python train.py arch4_tapt_full_h128
python adapt.py mntp 0
python adapt.py mntp 1
python train.py arch4_mntp_lora_h128_bidir
python train.py arch4_mntp_lora_h128_s1_bidir
python predict.py
```

Each `train.py` run writes its outputs and a submission zip to `runs/<run name>/`. `distill.py` needs `runs/arch4_full_h128/model_final.pt` and writes `runs/arch2_full_h128_kd/`. Every zip holds one `prediction.json` that maps each test document id to a list of `[start, end, label]`.

## Submissions this code reproduces

| Submission | Codabench |
|---|---|
| Arch 1, LoRMA+ | 0.3351 |
| Arch 3 encoder, full fine-tuning | 0.3640 |
| Arch 2, full fine-tuning | 0.5407 |
| Arch 2, distilled from Arch 4 | 0.5425 |
| Arch 4, full fine-tuning | 0.5957 |
| Bidirectional Arch 4, LoRA, seed 0 | 0.6261 |
| Bidirectional Arch 4, LoRA, seed 1 | 0.6503 |
| Ensemble and rules (best) | 0.6872 |

## Notes

The four required submissions are Arch 1 LoRMA+ (0.3351), Arch 2 distilled (0.5425), Arch 3 full fine-tuning (0.3640) and bidirectional Arch 4 LoRA (0.6503).
