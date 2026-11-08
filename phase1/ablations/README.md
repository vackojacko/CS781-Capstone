# Phase 1 component ablations

## What the runs test

These runs test the design choices of Arch 4 and Arch 2. Each run removes or replaces one component of a small version of the model, and everything else stays the same. The submitted Phase 1 models were not changed or retrained.

## Data and training setup

Data (the same for every run):

- A subset of the official Phase 1 corpus, tokenised with my own 16,384 token BPE from Assignment 1.
- Training: 64M tokens, 16M from each source (supreme_court, high_courts, acts, tribunals).
- Validation: 1M tokens (250k per source) from held out documents. 1,945 blocks of 513 tokens are scored in every evaluation.
- Arch 2 pipeline: split by document (5% of documents go to validation, seed 42), MD5 dedup on the first 500 characters, documents under 50 words dropped.

Training budget (the same for every run):

- Context 512. Every run sees 48.0M tokens: 2,929 updates of 32 x 512 tokens, one pass, no repetition.
- The data order is a fixed permutation (data seed 1234) shared by all runs. The model seed is 42 unless stated otherwise.
- AdamW (0.9, 0.95), peak LR 1e-3, min LR 1e-4, 150 warmup updates, weight decay 0.1 on 2D weights, gradient clip 1.0, fp16 mixed precision.

Group A (small Arch 4): d 384, 8 layers, 6 heads, MLA latent 192 with decoupled RoPE 32, SwiGLU 1024, QK norm, sigmoid output gate, Engram (bigram and trigram, 4 hash heads, 20,480 rows, 32 dims) at layers 1 and 4. 32.8M parameters (21.7M backbone and 11.0M Engram). WSD schedule: warmup, stable, then cosine decay over the last 20% of updates. Engram tables get 5x LR and no weight decay.

Group B (small Arch 2): d 384, 8 layers, GQA with 6 query heads and 2 KV heads, SwiGLU 1024, RoPE, RMSNorm. 18.9M parameters. Schedule: warmup, then cosine.

Validation loss (token level cross entropy on the 1M held out tokens) is measured 10 times during training and at the end.

## Files

| File | Content |
|---|---|
| `prep_data.py` | builds the token subset and the Engram compression map once |
| `arch4_model.py` | small Arch 4 with the switches `engram_layers`, `qk_norm`, `gated` and `latent` (False gives standard MHA with full head RoPE) |
| `arch2_model.py` | small Arch 2 with the switches `n_kv_heads`, `pos_emb_type`, `scale_emb`, `ffn_type` and `norm_type` |
| `train_ablation.py` | trains and evaluates one run |
| `configs.json` | the 18 run configs |
| `collect_results.py` | builds `results.json` and `results_table.md` from the run outputs |
| `results.json` | final numbers and the full validation history of every run |
| `results_table.md` | results table with throughput and the loss at every checkpoint |

## Results

Lower is better. Delta is the variant loss minus the baseline trained with the same seed, so the init seed and the data order match.

Group A, Arch 4 components. Baseline over 3 seeds: mean 2.7276, seed std 0.0173.

| Run | Seed | Change | Params | Val loss | Val PPL | Delta |
|---|---:|---|---:|---:|---:|---:|
| A0_full | 42 | full small Arch 4 | 32.8M | 2.7435 | 15.541 | |
| A0_seed2 | 7 | same, seed 7 | 32.8M | 2.7093 | 15.018 | |
| A0_seed3 | 99 | same, seed 99 | 32.8M | 2.7301 | 15.335 | |
| A1_no_engram | 42 | Engram removed | 21.7M | 2.7686 | 15.937 | +0.0251 |
| A1_no_engram_seed2 | 7 | Engram removed | 21.7M | 2.7922 | 16.317 | +0.0829 |
| A2_no_gate | 42 | sigmoid output gate removed | 31.6M | 2.7466 | 15.590 | +0.0031 |
| A3_no_qknorm | 42 | QK norm removed | 32.8M | 2.8039 | 16.510 | +0.0604 |
| A3_no_qknorm_seed2 | 7 | QK norm removed | 32.8M | 2.7743 | 16.027 | +0.0650 |
| A4_mha | 42 | MLA replaced by standard MHA with RoPE | 32.7M | 2.6704 | 14.446 | -0.0731 |
| A4_mha_seed2 | 7 | MLA replaced by standard MHA with RoPE | 32.7M | 2.6576 | 14.262 | -0.0517 |

Group B, Arch 2 components. Baseline over 3 seeds: mean 2.9106, seed std 0.0395. All variants use seed 42 and are compared with B0_full.

| Run | Seed | Change | Params | Val loss | Val PPL | Delta |
|---|---:|---|---:|---:|---:|---:|
| B0_full | 42 | full small Arch 2 | 18.9M | 2.8669 | 17.583 | |
| B0_seed2 | 7 | same, seed 7 | 18.9M | 2.9438 | 18.989 | |
| B0_seed3 | 99 | same, seed 99 | 18.9M | 2.9211 | 18.561 | |
| B1_mha | 42 | full MHA (6 KV heads) instead of GQA | 20.5M | 2.8537 | 17.352 | -0.0132 |
| B2_sinusoidal | 42 | sinusoidal positions instead of RoPE | 18.9M | 5.3069 | 201.726 | +2.4400 |
| B2b_sinusoidal_scaled | 42 | sinusoidal positions, embedding scaled by sqrt(d) | 18.9M | 2.9754 | 19.597 | +0.1085 |
| B3_gelu | 42 | GELU MLP (d_ff 1536, with biases) instead of SwiGLU | 18.9M | 2.8876 | 17.950 | +0.0207 |
| B4_layernorm | 42 | LayerNorm instead of RMSNorm | 18.9M | 2.8586 | 17.437 | -0.0083 |

## What the runs show

- Engram helps. Removing it raised the loss on both seeds (+0.025 and +0.083). The gap is largest early in training and gets smaller later.
- QK norm helps. Removing it raised the loss on both seeds (+0.060 and +0.065).
- RoPE beats sinusoidal positions. The plain sinusoidal version (added to an embedding initialised with std 0.02) nearly failed to train, because the position signal is much larger than the token signal. With the embedding scaled by sqrt(d) it trains normally but is still 0.109 worse than RoPE.
- The sigmoid gate, GQA versus MHA, SwiGLU versus GELU and RMSNorm versus LayerNorm all differ by less than the seed noise at this scale.
- MLA is 0.05 to 0.07 worse than standard MHA at the same width on two seeds. MLA is meant to shrink the KV cache, not to lower the loss, so this is its cost.

## Limits

- The models are small (19M to 33M parameters) and trained on 48M tokens with context 512, much smaller than the submitted models. The effects may change at full scale.
- Several runs (A2, B1, B2, B2b, B3, B4) use only one seed. With a group B seed std of about 0.04, small deltas from these runs are within noise.
- Removing Engram or the gate also removes parameters, so those two comparisons are not parameter matched. MHA versus MLA and GELU versus SwiGLU are matched within 0.3%.
- The metric is validation loss on held out documents, not the Phase 1 BERTScore.

## Reproducing the runs

1. Put the four training files in `data/` and the tokenizer file `legal_bpe_16384.json` in the working folder (the paths are constants at the top of `prep_data.py`). Run `python prep_data.py`. It writes `abl_data/train.npy`, `abl_data/val.npy`, `abl_data/engram_compress.npy` and `abl_data/meta.json`.
2. Train each run on a GPU with `python train_ablation.py <tag>`, for example `python train_ablation.py A0_full`. The tags are in `configs.json`. Each run writes `runs/<tag>/log.txt` and `runs/<tag>/result.json`. My runs took about 20 to 30 minutes each on one T4.
3. Run `python collect_results.py` to rebuild `results.json` and `results_table.md`.
