# Phase 4: Court Judgment Prediction (CJPE)

## Task and metric

Each input is the full text of an Indian Supreme Court judgment with the final decision removed (ILDC). The model predicts 1 if the appeal was accepted and 0 if it was rejected. The metric is macro-F1 over the 1,517 test cases. A submission is `submission.zip` with one file, `prediction.json`, holding `[{"id": ..., "pred": 0 or 1}, ...]`.

Best Codabench score: **0.8154** (average of three Arch 4 LoRMA runs, hierarchical input).

## Data and preprocessing

| | train | dev | test |
|---|---|---|---|
| rows | 37,387 | 1,988 | 1,517 |
| unique cases used | 32,305 | 994 | 1,517 |
| label 0 / 1 | 59 / 41 | 50 / 50 | balanced |

- Train is ILDC multi plus ILDC single. 5,082 case ids appear twice with identical text, so repeated ids are dropped (32,305 cases left). Dev holds every case twice, so it is also reduced to its 994 unique cases. Macro-F1 is the same on both.
- Test labels are present in `test.json` but are never read: the loader sets every test label to -1.
- Train text is cased and unwrapped. Dev and test are lowercased and carry anonymiser artefacts ("company" for "Co", "number" for "No"). Train is rewritten into the same format (`data.align_train`: three regexes, then lowercase). Whitespace is collapsed in every split.
- Text is tokenized with my Phase 1 BPE (16,384 tokens). Each word is split into letter, digit and punctuation pieces first, and only the last piece of a word gets the end-of-word marker (`tokenizer.PieceTokenizer`, same as Phases 2 and 3).
- Train cases keep their last 5,600 words and then their last 6,400 tokens (8 chunks of 1024 with stride 768). Dev and test cases are kept whole (up to 50,000 tokens).
- Chunks are counted back from the end of the case: length 1024 (Arch 3: 703 tokens plus its continuation marker, 704 in total), stride 768, so neighbouring chunks overlap by 256 tokens. A case shorter than one chunk is used whole. Batches only hold chunks of the same length, since Arch 4 has no padding mask.
- The end of a judgment carries the outcome. TF-IDF with logistic regression on dev (tuned threshold): first 512 words 0.604, last 512 words 0.666, last 2,048 words 0.705, full text 0.704.
- 99.5% of test cases also appear in the Phase 1 pretraining file, with their decision. That text is not used.

## Models

The Phase 1 model code lives in `../phase1` (`arch1_model.py` to `arch4_model.py`). Phase 4 removes the language model head and uses each model as a backbone.

| | Arch 1 | Arch 2 | Arch 3 | Arch 4 |
|---|---|---|---|---|
| Type | GPT-2 style decoder | LLaMA style decoder | encoder-decoder | latent-attention decoder |
| Layers x width | 12 x 768 | 12 x 640 | (10 enc + 4 dec) x 640 | 16 x 640 |
| Attention | MHA, 12 heads | GQA 10/2 | MHA + cross-attention | MLA, latent 320 |
| Positions | sinusoidal | RoPE | RoPE | decoupled RoPE |
| Norm | LayerNorm | RMSNorm | RMSNorm | RMSNorm, QK-norm |
| FFN | GELU 3072 | SwiGLU 1706 | SwiGLU 1712 | SwiGLU 1712 |
| Parameters (Phase 1) | 97.6M | 61.6M | 86.1M | 139.1M |
| Used in Phase 4 as | causal decoder | causal decoder | encoder only (60.6M with adapters) | bidirectional encoder |

- **Arch 1**: my Assignment 3 decoder scaled to 97.6M, with fixed sinusoidal positions. It keeps its dropout of 0.1.
- **Arch 2**: compact LLaMA style decoder. Grouped-query attention with 10 query heads and 2 key/value heads.
- **Arch 3**: the decoder is dropped and only the 10-layer encoder is used. Every chunk starts with token 16384, the continuation marker from its pretraining, so the text part is 703 tokens.
- **Arch 4**: multi-head latent attention, a sigmoid output gate and a hashed bigram/trigram memory (42.9M parameters) at layers 2 and 8. Phase 4 uses the bidirectional version from Phase 2: the causal mask removed and 800 updates of masked next-token training.
- Backbone dropout is 0 for Arch 2, 3 and 4 (their Phase 1 configs) and 0.1 for Arch 1.

**Chunk classifier** (`model.SentimentClassifier`): backbone, pooled vector, `Linear(d, 2)`. Pooling is the mean over tokens for the bidirectional models (Arch 3 encoder, Arch 4) and the last token for the causal ones (Arch 1, Arch 2).

**Aggregator** (`model.Aggregator`): LayerNorm, a one-layer BiLSTM (my Assignment 2 LSTM cell, hidden size 128), dropout 0.2 and `Linear(256, 2)` on the final states of both directions. It reads the vectors of the last 8 chunks of a case.

## Method

### Stage 1: fine-tune a chunk classifier

- Each epoch, every training case gives one chunk drawn at random from its last 2 chunks (last 3 for E6), labelled with the case label.
- Cross-entropy loss, no class weights.
- AdamW, weight decay 0.01, learning rate 1e-3 for the head and `lr_body` for the backbone or the added modules, 6% linear warmup then linear decay, gradient clipping 1.0.
- Batch 4 chunks x 8 accumulation steps = 32 chunks per update (Arch 1: 2 x 16).
- fp16 autocast with a gradient scaler, except Arch 1, which runs in fp32 (it did not learn in fp16 in Phase 3).
- Dev is scored every 400 updates and at the end of every epoch, on the mean logit margin of the last 2 chunks (3 for E6), with a tuned threshold. The trainable weights of the best point are kept.

### Fine-tuning technique per Phase 1 model

| Run | Model | Technique (`peft_methods.py`) | Settings | Trainable |
|---|---|---|---|---|
| E1 | Arch 4 bidirectional | LoRMA+, `W' = (alpha/r B A + I) W0`, on every `nn.Linear` of every block | r 16, alpha 32, B = 0 at start, lr 3e-4, 2 epochs | 4,146,434 of 143.2M |
| E2 | Arch 2 | gradual unfreezing | top 4 blocks + final norm + head, top 8 at 25% of updates, all at 50%; lr 5e-5, 2 epochs | 17.0M, 34.1M, then 61.6M |
| E3 | Arch 3 encoder | bottleneck adapter after every encoder block | size 64, GELU, up-projection zero at start; lr 5e-4, 3 epochs | 827,522 of 60.6M |
| E4 | Arch 1 | full fine-tuning, fp32 | lr 3e-5, batch 2 x 16, 1 epoch, 200-minute cap | 97.6M |
| E6 | Arch 4 bidirectional | LoRMA+ as E1 | trained on the last 3 chunks, seed 1 | 4,146,434 |
| E10 | Arch 4 bidirectional | LoRMA+ | r 32, alpha 64, 3 epochs, seed 2 | 8,291,586 |
| E11 | Arch 4 bidirectional | LoRMA+ as E1 | seed 3 | 4,146,434 |
| E5 | Arch 2 student | E2 recipe with distillation | see below | 17.0M to 61.6M |
| E12 | Arch 4 student | E1 recipe with distillation | see below | 4,146,434 |

LoRMA+ and adapters give exactly the frozen model's output at the start. The checks from Assignment 4 (dense and reordered LoRMA forms agree in float64, ranks, distillation loss properties) are `peft_methods.check_lorma()` and `distill.check_distillation_loss()`.

### Distillation

- Teachers: E1, E6 and E2. After stage 1, each teacher writes the logits of the last two chunks of 16,000 training cases (the same random subset for all three) to `runs/<run>/teacher_train.npz`.
- The teacher signal is the average of the three teachers' logits for each chunk.
- Loss: `0.5 CE(y, z_s) + 0.5 tau^2 KL(softmax(z_t / tau) || softmax(z_s / tau))`, tau 2.
- The students train only on the 16,000 cached cases, so they see half the cases and about half the updates (1,098 against 2,110) of their cross-entropy baselines.
- E5 (Arch 2 student, 61.6M) scored 0.7638 on Codabench, 95% of the teacher ensemble's Codabench score (E1 + E2 + E6, submission #40: 0.8048, by chance the same as E6 alone) with 18% of its parameters (61.6M against 339.8M) and one forward pass instead of three.

### Stage 2: hierarchical aggregator

- The fine-tuned model is frozen. It encodes every chunk of every dev and test case (up to 48 from the end) and the last 8 chunks of 6,000 random training cases (3,000 for Arch 1).
- The BiLSTM aggregator trains on those training vectors for 15 epochs (AdamW, lr 1e-3, weight decay 0.01, batch 64). The epoch with the best dev macro-F1 is kept.

### Input variants (long-document study)

Each run scores six inputs from the same fine-tuned model (`predict.py` and `train.run`):

| Variant | Score of a case | E1 dev / Codabench |
|---|---|---|
| `first` | logit margin of the first chunk | 0.620 / 0.604 |
| `selected` | mean margin of the 2 chunks with the most reasoning cue words (`train.CUES`) | 0.741 / 0.740 |
| `last` | margin of the last chunk | 0.782 / 0.741 |
| `tail2` | mean margin of the last 2 chunks | 0.779 / 0.775 |
| `whole` | mean margin over all chunks | 0.788 / 0.767 |
| `hier` | BiLSTM over the last 8 chunk vectors | 0.804 / 0.790 |

### Threshold

Train is 59/41 while dev and test are balanced, so at threshold 0 the models predict too few accepted appeals (22% to 42% on dev with the `hier` input, none for Arch 1). Every submission uses a threshold tuned on dev: the cut among 181 dev score quantiles (5% to 95%) with the best dev macro-F1. The prior-shift rule (add log(0.59/0.41) to the margin) is also logged, and it was lower (E1 `hier`: 0.7814 against 0.8038).

### Ensemble

`ensemble.py` standardises each member's `hier` scores with that run's dev mean and standard deviation (the same values for dev and test), averages them with equal weights and tunes the threshold on dev. Members: E1, E10 and E11 (all Arch 4 LoRMA, trained on the last 2 chunks). Dev 0.8147 (threshold 0.079), 49.0% of test cases predicted 1, Codabench 0.8154.

## Files

| File | What it does |
|---|---|
| `data.py` | Loads the splits, drops repeated ids, rewrites train into the dev format, tokenizes all splits with 4 processes and writes `tokens/{train,dev,test}.npz`. Also has the chunking helpers (`tail_starts`, `chunk_ids`, `make_batches`). Run it once. |
| `tokenizer.py` | Loads the Phase 1 BPE merges and applies them to the letter, digit and punctuation pieces of each word. |
| `backbone.py` | Loads a Phase 1 checkpoint without its language model head. Arch 1 and 2 configs are fixed in the file, Arch 3 and 4 configs come from `metrics.json` in the model folder. Removes the causal mask for the bidirectional Arch 4 and keeps only the encoder of Arch 3. |
| `model.py` | Pooling functions, the chunk classifier, my Assignment 2 LSTM cell and recurrent encoder, and the BiLSTM aggregator. |
| `peft_methods.py` | Bottleneck adapters, LoRMA+ (`LoRMALinear`) with its float64 checks, layer unfreezing (`unfreeze_top`) and `attach_peft`, which freezes the backbone and adds the trainable parts of a method. |
| `distill.py` | Distillation loss with its checks, and saving and loading of the teacher logit caches. `TEACHER_FILES` lists the three teacher caches. |
| `predict.py` | Scores for each input variant, the dev threshold search, prior-shift and threshold-0 scores, and the submission zip writer. |
| `train.py` | One full run: the configs of all runs (`CONFIGS`), stage 1, encoding, teacher cache, stage 2, dev scores of every variant, `scores.npz`, submission zips and `metrics.json`. |
| `ensemble.py` | The best submission from the `scores.npz` files of E1, E10 and E11. |

## How to run

Dependencies: Python 3, PyTorch, NumPy, scikit-learn. Training needs a GPU (runs were done on one Kaggle T4 each). Run every command from this folder.

Path constants (relative to the working folder):

- `data.py`: `DATA_DIR = "data/cjpe"` (holds `train.json`, `dev.json`, `test.json`), `BPE_PATH = "data/legal_bpe_16384.json"` (Phase 1 tokenizer), `TOKEN_DIR = "tokens"`.
- `train.py`: `OUT_DIR = "runs"`, `MODEL_DIRS` (one folder per architecture; Arch 4 is the bidirectional checkpoint from Phase 2). Each model folder needs `model.pt`, the matching Phase 1 model file from `../phase1` saved as `model_arch.py`, and for Arch 3 and Arch 4 the Phase 1 `metrics.json` (its `model_config` is read). Arch 4 may also hold `arch4_engram_compress.pt`.
- `distill.py`: `TEACHER_FILES`, the caches written by E1, E6 and E2.
- `ensemble.py`: `RUNS_DIR`, `OUT_PATH`, `MEMBERS`, `VARIANT`.

Order:

```
python data.py                                   # tokens/train.npz, dev.npz, test.npz
python train.py e1_arch4_lorma                   # teacher; writes runs/e1_arch4_lorma/teacher_train.npz
python train.py e6_arch4_lorma_tail3_s1          # teacher
python train.py e2_arch2_unfreeze                # teacher
python train.py e3_arch3_adapter
python train.py e4_arch1_full
python train.py e10_arch4_lorma_r32_e3_s2
python train.py e11_arch4_lorma_s3
python train.py e5_arch2_kd                      # needs the three teacher caches
python train.py e12_arch4_lorma_kd               # needs the three teacher caches
python ensemble.py                               # submission_ens_a4_tail2_hier.zip
```

Each `train.py <run>` writes to `runs/<run>/`:

- `log.txt`: training log with dev scores every 400 updates, aggregator epochs and the dev score of every variant.
- `metrics.json`: config, trainable and total parameters, training history, dev results, chosen variant, minutes.
- `scores.npz`: dev and test scores of all six variants, case ids and dev labels.
- `aggregator.pt`, and `trainable_state.pt` when fewer than 30M parameters are trained.
- `submission_<run>_tail2.zip`, `submission_<run>_hier.zip`, and the zip of the best variant on dev if that is `whole`.
- `teacher_train.npz` for E1, E6 and E2.

Measured T4 times per run: E1 242 min, E6 241, E10 299, E11 212, E2 117, E3 79, E4 199, E5 65, E12 135.

The other input variants (`first`, `last`, `selected`, `whole`) are written from `scores.npz` with the same dev threshold rule, for example:

```
python -c "import sys, numpy as np; from predict import report, write_submission; run, v = sys.argv[1], sys.argv[2]; z = np.load('runs/' + run + '/scores.npz'); r = report(z['dev_' + v], z['dev_y']); write_submission(run + '__' + v + '.zip', list(z['test_ids']), z['test_' + v] > r['thr'])" e1_arch4_lorma first
```

## Codabench submissions covered by this code

Dev is macro-F1 on the 994 dev cases with the dev-tuned threshold. "#" is the submission number in the phase record. "variant command" means the `python -c` line above with the given run and variant, after that run has finished.

| # | Submission | Dev | Codabench | Produced by |
|---|---|---|---|---|
| 01 | Arch 1 E4, first | 0.5133 | 0.4752 | variant command `e4_arch1_full first` |
| 02 | Arch 1 E4, whole | 0.5315 | 0.4894 | variant command `e4_arch1_full whole` |
| 03 | Arch 1 E4, hier | 0.5483 | 0.5108 | `python train.py e4_arch1_full` |
| 04 | Arch 3 E3, first | 0.5813 | 0.5618 | variant command `e3_arch3_adapter first` |
| 05 | Arch 2 E2, first | 0.6143 | 0.5866 | variant command `e2_arch2_unfreeze first` |
| 06 | Arch 3 E3, tail2 | 0.6196 | 0.6021 | `python train.py e3_arch3_adapter` |
| 07 | Arch 4 E1, first | 0.6203 | 0.6044 | variant command `e1_arch4_lorma first` |
| 08 | Arch 3 E3, hier | 0.6582 | 0.6154 | `python train.py e3_arch3_adapter` |
| 09 | Arch 2 E2, selected | 0.7122 | 0.7168 | variant command `e2_arch2_unfreeze selected` |
| 10 | Arch 4 E1, selected | 0.7409 | 0.7396 | variant command `e1_arch4_lorma selected` |
| 11 | Arch 4 E6, selected | 0.7485 | 0.7498 | variant command `e6_arch4_lorma_tail3_s1 selected` |
| 13 | Arch 2 E2, last | 0.7555 | 0.7436 | variant command `e2_arch2_unfreeze last` |
| 14 | Arch 2 E2, whole | 0.7560 | 0.7473 | variant command `e2_arch2_unfreeze whole` |
| 16 | Arch 2 E2, hier | 0.7686 | 0.7646 | `python train.py e2_arch2_unfreeze` |
| 17 | Arch 4 E6, tail2 | 0.7744 | 0.7755 | `python train.py e6_arch4_lorma_tail3_s1` |
| 18 | Arch 4 E1, tail2 | 0.7786 | 0.7751 | `python train.py e1_arch4_lorma` |
| 19 | Arch 4 E1, last | 0.7816 | 0.7408 | variant command `e1_arch4_lorma last` |
| 20 | Arch 4 E6, whole | 0.7875 | 0.7698 | variant command `e6_arch4_lorma_tail3_s1 whole` |
| 21 | Arch 4 E1, whole | 0.7877 | 0.7665 | variant command `e1_arch4_lorma whole` |
| 27 | Arch 4 E1, hier | 0.8038 | 0.7902 | `python train.py e1_arch4_lorma` |
| 37 | Arch 4 E6, hier (best single model) | 0.8075 | 0.8048 | `python train.py e6_arch4_lorma_tail3_s1` |
| 41 | Arch 2 student E5, whole | 0.7433 | 0.7502 | variant command `e5_arch2_kd whole` |
| 42 | Arch 2 student E5, tail2 | 0.7461 | 0.7607 | `python train.py e5_arch2_kd` |
| 43 | Arch 2 student E5, hier | 0.7515 | 0.7638 | `python train.py e5_arch2_kd` |
| 44 | Arch 4 student E12, tail2 | 0.7645 | 0.7487 | `python train.py e12_arch4_lorma_kd` |
| 45 | Arch 4 student E12, hier | 0.7837 | 0.7772 | `python train.py e12_arch4_lorma_kd` |
| 46 | Arch 4 student E12, whole | 0.7865 | 0.7551 | `python train.py e12_arch4_lorma_kd` (best variant on dev) |
| 48 | Arch 4 E11, hier | 0.7958 | 0.7896 | `python train.py e11_arch4_lorma_s3` |
| 50 | Arch 4 E10, hier | 0.8099 | 0.7947 | `python train.py e10_arch4_lorma_r32_e3_s2` |
| **53** | **Ensemble E1 + E10 + E11, hier (best)** | **0.8147** | **0.8154** | **`python ensemble.py`** |

Best experiment per architecture: Arch 4 E6 (0.8048), Arch 2 E2 (0.7646), Arch 3 E3 (0.6154), Arch 1 E4 (0.5108).

The phase had 80 submissions in total. The rest are ensembles over other member lists and runs not configured here (E7: E2 trained on the last 3 chunks; E13 and E14: E6 with seeds 4 and 5; E15 to E20: E1 with seeds 6 to 11). None scored above #53.
