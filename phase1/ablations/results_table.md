
Group A: baseline over 3 seeds, mean val loss 2.7276, seed std 0.0173, range 0.0342

| Run | Seed | Change | Params | Val loss | Val PPL | Delta vs same-seed baseline | tok/s | Train min |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A0_full | 42 | baseline: MLA + decoupled RoPE, QK-norm, sigmoid gate, Engram at layers 1 and 4 | 32.8M | 2.7435 | 15.541 | - | 30587 | 26.1 |
| A0_seed2 | 7 | baseline, second seed | 32.8M | 2.7093 | 15.018 | - | 29839 | 26.8 |
| A0_seed3 | 99 | baseline, third seed | 32.8M | 2.7301 | 15.335 | - | 29178 | 27.4 |
| A1_no_engram | 42 | Engram memory removed | 21.7M | 2.7686 | 15.937 | +0.0251 | 33517 | 23.9 |
| A1_no_engram_seed2 | 7 | A1 with seed 7 (paired with A0_seed2) | 21.7M | 2.7922 | 16.317 | +0.0829 | 31341 | 25.5 |
| A2_no_gate | 42 | sigmoid output gate removed | 31.6M | 2.7466 | 15.590 | +0.0031 | 29631 | 27.0 |
| A3_no_qknorm | 42 | QK-norm removed | 32.8M | 2.8039 | 16.510 | +0.0604 | 36676 | 21.8 |
| A3_no_qknorm_seed2 | 7 | A3 with seed 7 (paired with A0_seed2) | 32.8M | 2.7743 | 16.027 | +0.0650 | 37530 | 21.3 |
| A4_mha | 42 | MLA replaced by standard MHA with full-head RoPE (QK-norm, gate, Engram kept) | 32.7M | 2.6704 | 14.446 | -0.0731 | 34188 | 23.4 |
| A4_mha_seed2 | 7 | A4 with seed 7 (paired with A0_seed2) | 32.7M | 2.6576 | 14.262 | -0.0517 | 35023 | 22.8 |

Group B: baseline over 3 seeds, mean val loss 2.9106, seed std 0.0395, range 0.0769

| Run | Seed | Change | Params | Val loss | Val PPL | Delta vs same-seed baseline | tok/s | Train min |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| B0_full | 42 | baseline: GQA 6/2, RoPE, RMSNorm, SwiGLU | 18.9M | 2.8669 | 17.583 | - | 37231 | 21.5 |
| B0_seed2 | 7 | baseline, second seed | 18.9M | 2.9438 | 18.989 | - | 36316 | 22.0 |
| B0_seed3 | 99 | baseline, third seed | 18.9M | 2.9211 | 18.561 | - | 36600 | 21.9 |
| B1_mha | 42 | full MHA (6 KV heads) instead of GQA | 20.5M | 2.8537 | 17.352 | -0.0132 | 34538 | 23.2 |
| B2_sinusoidal | 42 | sinusoidal absolute positions instead of RoPE | 18.9M | 5.3069 | 201.726 | +2.4400 | 39184 | 20.4 |
| B2b_sinusoidal_scaled | 42 | sinusoidal positions, token embedding multiplied by sqrt(d_model) before adding them (Vaswani et al.) | 18.9M | 2.9754 | 19.597 | +0.1085 | 39514 | 20.2 |
| B3_gelu | 42 | GELU MLP (d_ff 1536, with biases, Arch 1 FeedForward) instead of SwiGLU | 18.9M | 2.8876 | 17.950 | +0.0207 | 37324 | 21.4 |
| B4_layernorm | 42 | LayerNorm instead of RMSNorm | 18.9M | 2.8586 | 17.437 | -0.0083 | 40069 | 20.0 |

Validation loss at fixed checkpoints (tokens seen in millions)
- A0_full: 4.8M:4.281 9.6M:3.702 14.4M:3.455 19.1M:3.314 23.9M:3.220 28.7M:3.137 33.5M:3.063 38.3M:3.002 43.1M:2.864 47.8M:2.745 48.0M:2.744
- A0_seed2: 4.8M:4.280 9.6M:3.704 14.4M:3.453 19.1M:3.310 23.9M:3.209 28.7M:3.120 33.5M:3.021 38.3M:2.956 43.1M:2.825 47.8M:2.710 48.0M:2.709
- A0_seed3: 4.8M:4.279 9.6M:3.701 14.4M:3.454 19.1M:3.316 23.9M:3.220 28.7M:3.142 33.5M:3.066 38.3M:2.993 43.1M:2.851 47.8M:2.731 48.0M:2.730
- A1_no_engram: 4.8M:4.450 9.6M:3.846 14.4M:3.571 19.1M:3.404 23.9M:3.271 28.7M:3.163 33.5M:3.073 38.3M:3.022 43.1M:2.896 47.8M:2.769 48.0M:2.769
- A1_no_engram_seed2: 4.8M:4.459 9.6M:3.865 14.4M:3.597 19.1M:3.443 23.9M:3.330 28.7M:3.223 33.5M:3.119 38.3M:3.055 43.1M:2.920 47.8M:2.793 48.0M:2.792
- A2_no_gate: 4.8M:4.297 9.6M:3.724 14.4M:3.485 19.1M:3.348 23.9M:3.251 28.7M:3.150 33.5M:3.047 38.3M:2.990 43.1M:2.863 47.8M:2.748 48.0M:2.747
- A3_no_qknorm: 4.8M:4.311 9.6M:3.729 14.4M:3.482 19.1M:3.346 23.9M:3.253 28.7M:3.174 33.5M:3.107 38.3M:3.057 43.1M:2.928 47.8M:2.805 48.0M:2.804
- A3_no_qknorm_seed2: 4.8M:4.303 9.6M:3.733 14.4M:3.480 19.1M:3.343 23.9M:3.245 28.7M:3.164 33.5M:3.093 38.3M:3.033 43.1M:2.901 47.8M:2.775 48.0M:2.774
- A4_mha: 4.8M:4.234 9.6M:3.655 14.4M:3.410 19.1M:3.269 23.9M:3.156 28.7M:3.050 33.5M:2.951 38.3M:2.899 43.1M:2.780 47.8M:2.671 48.0M:2.670
- A4_mha_seed2: 4.8M:4.240 9.6M:3.665 14.4M:3.414 19.1M:3.256 23.9M:3.123 28.7M:3.012 33.5M:2.933 38.3M:2.887 43.1M:2.769 47.8M:2.659 48.0M:2.658
- B0_full: 4.8M:4.515 9.6M:3.862 14.4M:3.577 19.1M:3.388 23.9M:3.237 28.7M:3.109 33.5M:3.009 38.3M:2.943 43.1M:2.894 47.8M:2.868 48.0M:2.867
- B0_seed2: 4.8M:4.514 9.6M:3.871 14.4M:3.595 19.1M:3.426 23.9M:3.296 28.7M:3.184 33.5M:3.090 38.3M:3.021 43.1M:2.973 47.8M:2.945 48.0M:2.944
- B0_seed3: 4.8M:4.506 9.6M:3.867 14.4M:3.585 19.1M:3.409 23.9M:3.277 28.7M:3.164 33.5M:3.068 38.3M:2.999 43.1M:2.949 47.8M:2.922 48.0M:2.921
- B1_mha: 4.8M:4.513 9.6M:3.850 14.4M:3.564 19.1M:3.377 23.9M:3.230 28.7M:3.102 33.5M:3.000 38.3M:2.932 43.1M:2.882 47.8M:2.854 48.0M:2.854
- B2_sinusoidal: 4.8M:7.023 9.6M:6.248 14.4M:6.131 19.1M:6.077 23.9M:5.731 28.7M:5.592 33.5M:5.494 38.3M:5.415 43.1M:5.349 47.8M:5.312 48.0M:5.307
- B2b_sinusoidal_scaled: 4.8M:5.027 9.6M:4.142 14.4M:3.756 19.1M:3.539 23.9M:3.371 28.7M:3.238 33.5M:3.130 38.3M:3.056 43.1M:3.005 47.8M:2.976 48.0M:2.975
- B3_gelu: 4.8M:4.621 9.6M:3.957 14.4M:3.652 19.1M:3.465 23.9M:3.297 28.7M:3.145 33.5M:3.037 38.3M:2.968 43.1M:2.917 47.8M:2.888 48.0M:2.888
- B4_layernorm: 4.8M:4.507 9.6M:3.860 14.4M:3.567 19.1M:3.380 23.9M:3.223 28.7M:3.095 33.5M:2.996 38.3M:2.932 43.1M:2.885 47.8M:2.859 48.0M:2.859
