# Phase 1: Ablations

These runs test the parts of my Arch 4 and Arch 2 models. Each run takes a small version of the model and removes or swaps one part, keeping everything else the same. The submitted models were not changed.

## Setup

- Data: 64M training tokens (16M per source) and 1M validation tokens from held-out documents, with my 16,384-token BPE.
- Every run sees 48.0M tokens with context 512, and uses model seed 42 unless the table says otherwise.
- Group A is a small Arch 4 with 32.8M parameters. Group B is a small Arch 2 with 18.9M parameters.
- Metric: validation loss on the 1M held-out tokens.

## Results

Lower is better. Delta is the loss minus the baseline with the same seed.

Group A. Baseline over 3 seeds: mean 2.7276, std 0.0173.

| Run | Seed | Change | Params | Val loss | Val PPL | Delta |
|---|---:|---|---:|---:|---:|---:|
| A0_full | 42 | full small Arch 4 | 32.8M | 2.7435 | 15.541 | |
| A0_seed2 | 7 | same, seed 7 | 32.8M | 2.7093 | 15.018 | |
| A0_seed3 | 99 | same, seed 99 | 32.8M | 2.7301 | 15.335 | |
| A1_no_engram | 42 | Engram removed | 21.7M | 2.7686 | 15.937 | +0.0251 |
| A1_no_engram_seed2 | 7 | Engram removed | 21.7M | 2.7922 | 16.317 | +0.0829 |
| A2_no_gate | 42 | output gate removed | 31.6M | 2.7466 | 15.590 | +0.0031 |
| A3_no_qknorm | 42 | QK norm removed | 32.8M | 2.8039 | 16.510 | +0.0604 |
| A3_no_qknorm_seed2 | 7 | QK norm removed | 32.8M | 2.7743 | 16.027 | +0.0650 |
| A4_mha | 42 | MLA replaced by MHA with RoPE | 32.7M | 2.6704 | 14.446 | -0.0731 |
| A4_mha_seed2 | 7 | MLA replaced by MHA with RoPE | 32.7M | 2.6576 | 14.262 | -0.0517 |

Group B. Baseline over 3 seeds: mean 2.9106, std 0.0395. Variants use seed 42 and are compared with B0_full.

| Run | Seed | Change | Params | Val loss | Val PPL | Delta |
|---|---:|---|---:|---:|---:|---:|
| B0_full | 42 | full small Arch 2 | 18.9M | 2.8669 | 17.583 | |
| B0_seed2 | 7 | same, seed 7 | 18.9M | 2.9438 | 18.989 | |
| B0_seed3 | 99 | same, seed 99 | 18.9M | 2.9211 | 18.561 | |
| B1_mha | 42 | MHA (6 KV heads) instead of GQA | 20.5M | 2.8537 | 17.352 | -0.0132 |
| B2_sinusoidal | 42 | sinusoidal instead of RoPE | 18.9M | 5.3069 | 201.726 | +2.4400 |
| B2b_sinusoidal_scaled | 42 | sinusoidal, embedding scaled by sqrt(d) | 18.9M | 2.9754 | 19.597 | +0.1085 |
| B3_gelu | 42 | GELU MLP (d_ff 1536) instead of SwiGLU | 18.9M | 2.8876 | 17.950 | +0.0207 |
| B4_layernorm | 42 | LayerNorm instead of RMSNorm | 18.9M | 2.8586 | 17.437 | -0.0083 |

## What I learned

- Engram memory and QK norm help. Removing either one made the loss worse on both seeds.
- RoPE beats sinusoidal positions. Plain sinusoidal nearly failed to train, because the position signal is much bigger than the small token embedding.
- The output gate, GQA, SwiGLU and RMSNorm change the loss by less than the seed noise.
- MLA is 0.05 to 0.07 worse than MHA. MLA is meant to shrink the KV cache, so this is its cost.

Limits: the models are small, several runs have one seed, and the metric is validation loss, not BERTScore.

## How to run

1. Put the training files in `data/` and `legal_bpe_16384.json` in the working folder (constants at the top of `prep_data.py`). Run `python prep_data.py`. It writes the data to `abl_data/`.
2. On a GPU, run `python train_ablation.py <tag>`, for example `python train_ablation.py A0_full`. The 18 tags are in `configs.json`. Each run writes to `runs/<tag>/` and takes about 20 to 30 minutes on a T4.
3. Run `python collect_results.py` to rebuild `results.json` and `results_table.md`. The second file also has the loss at every checkpoint.
