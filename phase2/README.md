# Phase 2: Legal Named Entity Recognition (L NER)

## Task

The input is a long Indian court judgment. The output is every entity in it, given as a character span (start, end) with one of 12 labels: APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO. The metric is strict macro F1: a span counts only if its start, end and label are all right, and every label counts equally.

## My final system

My final system is one model: my Phase 1 Arch 4 made bidirectional, with LoRA and a small tagging head.

1. I split each word into letter, digit and punctuation pieces before my Phase 1 BPE, so that entity edges line up with token edges.
2. To make Arch 4 bidirectional, I first continued its pretraining for 1500 updates of next token prediction. Then I removed the causal mask and trained it for 800 updates to predict masked tokens from the output one position before them. Both steps are in `adapt.py`.
3. I then trained LoRA layers (rank 16) and a tagging head that gives every token a BIOES tag. Long documents are read in overlapping windows, and the start of the document (parties, counsel, judges) is added to every window.
4. A Viterbi decoder keeps only valid tag sequences.
5. If a string of at least 6 characters is tagged with an average confidence of at least 0.5, its other copies in the same document get the same label, and every copy of that string takes the label with the highest total confidence.
6. A small boundary repair fixes span edges that cut a word, and trailing brackets and punctuation.

Leaderboard score: 0.6503 (seed 1).

## Models I tried

| Phase 1 model | Technique | Codabench |
|---|---|---|
| Arch 1 | LoRMA+ | 0.3351 |
| Arch 3 (encoder only) | Full fine tuning | 0.3640 |
| Arch 2 | Full fine tuning | 0.5407 |
| Arch 2 | Distillation from the fine tuned Arch 4 | 0.5425 |
| Arch 4 | Full fine tuning | 0.5957 |
| Arch 4, made bidirectional | LoRA, seed 0 | 0.6261 |
| Arch 4, made bidirectional | LoRA, seed 1 | 0.6503 |

I wrote the LoRA and LoRMA+ layers myself (rank 16). The distilled Arch 2 learns half from the gold tags and half from the Arch 4 predictions.

## Training setup

- Final model: bidirectional Arch 4 with LoRA (rank 16, alpha 16, dropout 0.05) on every linear layer, and a linear tagging head over 49 BIOES tags.
- Backbone learning rate 5e-4, batch 4, 6 epochs, dropout 0.1, label smoothing 0.1.
- Windows of 1024 tokens: the first 128 tokens of the document, a separator, then the text. Rare label windows are used twice and rare tags get larger loss weights.
- I chose settings with two-fold cross validation on the 70 training documents, then trained the final model on both folds.
- All runs were on one Kaggle T4 GPU with fp16 mixed precision. Task runs use AdamW with weight decay 0.01, gradient clipping 1.0, a linear warmup then linear decay, and learning rate 1e-3 for the task head.

## Files

| File | What it does |
|---|---|
| `preprocess.py` | labels, piece tokenizer, spans to tags and back, windows, class weights |
| `model.py` | loads a Phase 1 model and adds the tagging head |
| `lorma.py` | LoRA and LoRMA+ layers |
| `decode.py` | Viterbi decoding, label spreading in a document, boundary repair, the macro F1 score |
| `train.py` | fine tuning, cross validation, the test predictions and the settings of every run |
| `distill.py` | Arch 2 distilled from Arch 4 |
| `adapt.py` | extra training of Arch 4 and the bidirectional version |
| `experiments/` | code for my other submissions, listed under Other submissions below |

## How to run

Dependencies: Python 3, PyTorch, NumPy and ijson (`pip install torch numpy ijson`). I trained on a Kaggle T4 GPU.

Path constants at the top of the files (the defaults work if you run from this folder):

| Constant | File | What it needs |
|---|---|---|
| `DATA_DIR = "data"` | `preprocess.py` | `fold1-text.json`, `fold1-gold.json`, `fold2-text.json`, `fold2-gold.json`, `fold3-text.json` |
| `TOKENIZER_PATH = "legal_bpe_16384.json"` | `preprocess.py` | my Phase 1 tokenizer, made by `../phase1/train_tokenizer.py` |
| `MODEL_DIRS` | `model.py` | `models/arch1` to `models/arch4`, each with `model_arch.py` (a copy of `../phase1/archN_model.py`) and the Phase 1 `model.pt`; `models/arch3` and `models/arch4` also need `metrics.json`, and `models/arch4` needs `arch4_engram_compress.pt`; `models/arch2` holds the Arch 2 checkpoint after its extra pretraining |
| `CORPUS_FILES` | `adapt.py` | `data/phase1/train_supreme_court.json` and `data/phase1/train_high_courts.json` |
| `OUT_DIR = "runs"` and `MNTP_DIRS` | `train.py` | where runs are written and where the bidirectional models are read from |

Commands, in this order:

```
python train.py arch1_lorma_h128
python train.py arch2_full_h128
python train.py arch3_full_h128
python train.py arch4_full_h128
python distill.py
python adapt.py tapt
python adapt.py mntp 0
python adapt.py mntp 1
python train.py arch4_mntp_lora_h128_bidir
python train.py arch4_mntp_lora_h128_s1_bidir
```

Each `train.py` run writes its test log probabilities, the final model and a submission zip to `runs/<run name>/`. `distill.py` needs `runs/arch4_full_h128/model_final.pt` and writes `runs/arch2_full_h128_kd/`. `adapt.py tapt` writes `runs/arch4_tapt`, and `adapt.py mntp 0` and `adapt.py mntp 1` start from it and write `runs/arch4_mntp_s0` and `runs/arch4_mntp_s1`. Every zip holds one `prediction.json` that maps each test document id to a list of `[start, end, label]`.

## Submissions this code reproduces

| Submission | Run | Codabench |
|---|---|---|
| Arch 1, LoRMA+ | `arch1_lorma_h128` | 0.3351 |
| Arch 3 encoder, full fine tuning | `arch3_full_h128` | 0.3640 |
| Arch 2, full fine tuning | `arch2_full_h128` | 0.5407 |
| Arch 2, distilled from Arch 4 | `arch2_full_h128_kd` | 0.5425 |
| Arch 4, full fine tuning | `arch4_full_h128` | 0.5957 |
| Bidirectional Arch 4, LoRA, seed 0 | `arch4_mntp_lora_h128_bidir` | 0.6261 |
| Bidirectional Arch 4, LoRA, seed 1 (leaderboard) | `arch4_mntp_lora_h128_s1_bidir` | 0.6503 |

The four required submissions are Arch 1 LoRMA+ (0.3351), Arch 2 distilled (0.5425), Arch 3 full fine tuning (0.3640) and bidirectional Arch 4 LoRA (0.6503).

## Other submissions

These are my other Codabench submissions: single model runs and ensembles of my runs. Run the commands from this folder.

| Submission | Codabench | How to reproduce |
|---|---|---|
| Arch 2, full fine tuning, attention unmasked (no adaptation) | 0.5253 | `python -m experiments.extra_runs arch2_full_h128_bidir` |
| Arch 4, LoRA, causal | 0.6121 | `python -m experiments.extra_runs arch4_lora_h128` |
| Arch 1, LoRMA+ (required submission, same file as `arch1_lorma_h128`) | 0.3351 | `python train.py arch1_lorma_h128` |
| Arch 2, distilled from Arch 4 (required submission, same file as `arch2_full_h128_kd`) | 0.5425 | `python distill.py` |
| Arch 3 encoder, full fine tuning (required submission, same file as `arch3_full_h128`) | 0.3640 | `python train.py arch3_full_h128` |
| Bidirectional Arch 4, LoRA, seed 1 (required submission, same file as `arch4_mntp_lora_h128_s1_bidir`) | 0.6503 | `python train.py arch4_mntp_lora_h128_s1_bidir` |
| Ensemble: Arch 4 (weight 2) and Arch 2 with a 256 token header, seed 1 (weight 1); spreading in the document | 0.6142 | `python -m experiments.ensembles ens_a4_a2_a2h256` |
| Same ensemble, plus boundary repair | 0.6211 | `python -m experiments.ensembles ens_a4x2_a2h256_repair` |
| Same ensemble, plus per label biases | 0.6281 | `python -m experiments.ensembles ens_bias` |
| Same, plus case insensitive spreading of APP names | 0.6293 | `python -m experiments.ensembles ens_bias_ciapp` |
| Ensemble of Arch 4 (weight 2), Arch 4 seed 1 and Arch 2 h256, with its own per label biases | 0.6282 | `python -m experiments.ensembles ens3_bias` |

`experiments/extra_runs.py` only adds settings; the training, cross validation and submission writing all come from `train.py`, and each run writes to `runs/<run name>/`. `arch4_lora_h128` starts from the original Phase 1 Arch 4 in `models/arch4`.

`experiments/ensembles.py` reads `runs/<run name>/test_logprobs.npz` of each member, averages the probabilities with the weights in its `RECIPES` table, and then runs Viterbi, spreading in the document and boundary repair; it writes `submissions/submission_<name>.zip`. The two members I never submitted alone are trained with `python -m experiments.members arch4_full_h128_s1` and `python -m experiments.members arch2_full_h256_s1`. The per label biases are small numbers added to a label's tag scores before Viterbi, chosen on cross validation.
