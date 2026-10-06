# ⚖️ Phase 1: Legal Language Modelling

> 512-word legal prompt in, 128-word continuation out. Metric: BERTScore F1.

## 📓 Notebooks

| Notebook | Purpose |
|---|---|
| `Train_BPE_Tokenizer` | Train our BPE tokenizer (16,384 tokens) on the legal corpus |
| `Capstone_Phase1_Arch1` | GPT-2 style decoder |
| `Capstone_Phase1_Arch2` | LLaMA style decoder |
| `Capstone_Phase1_Arch2_continue` | Continue Arch 2 on 329M unseen tokens |
| `Capstone_Phase1_Arch3` | Encoder-decoder, mixed objective |
| `Capstone_Phase1_Arch4` | Latent attention + Engram memory decoder |
| `Capstone_Phase1_Decoding_Selection` | Decoding sweep (40 settings) and MBR |
| `Capstone_Phase1_Arch2_Pool18` / `Pool48` | Large MBR candidate pools, many submissions |
| `Capstone_Phase1_Overlap_Check` | Measure text shared between test prompts and the training corpus |
| `Capstone_Phase1_Arch2_RAG` | Teach Arch 2 to use retrieved passages, then generate |

## 🏆 Scores (CodaBench)

| Submission | BERTScore F1 |
|---|:--:|
| Arch 1 | 0.8299 |
| Arch 2 | 0.8334 |
| Arch 2 continued | 0.8351 |
| Arch 3 | 0.8275 |
| Arch 4 | 0.8365 |
| Arch 2 + MBR x8 | 0.8388 |
| Arch 2 + MBR x16 | 0.8402 |
| **Arch 2 + retrieval fine-tune + MBR x16** | **0.8681** |

## 🔎 How the best score works

1. **Search**: the last 12 words of a test prompt are looked up in the training corpus (40% of prompts match)
2. **Context**: the passage that follows the match is placed inside the model input
3. **Fine-tune**: Arch 2 learns to use such passages (and to ignore wrong ones, 25% distractors)
4. **Generate**: the model writes 16 continuations, MBR keeps the consensus one
5. **Fallback**: prompts without a match keep the MBR x16 output

Held-out effect on matched prompts: 0.8817 to **0.9253** BERTScore F1.

## 📁 Layout

- `notebooks/`: the notebooks above
- `tokenizer/`: `bpe_tokenizer.py`, `train_tokenizer.py`, `legal_bpe_16384.json`
- `results/<model>/`: `metrics.json` and `loss_curves.png` for each training run
