# Phase 3: Rhetorical Role Prediction

## Task

The input is a legal judgment split into sentences. The output gives each sentence one of 13 rhetorical roles: Fact, Issue, ArgumentPetitioner, ArgumentRespondent, Statute, Dissent, PrecedentReliedUpon, PrecedentNotReliedUpon, PrecedentOverruled, RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt and None. The metric is macro-F1 over all 13 labels, with the sentences of all test documents pooled together.

## My final system

- One model covers both domains (Competition Law and Income Tax), with the domain given as an input.
- Each Phase 1 model encodes the sentences, and a BiLSTM-CRF on top predicts the roles. The LSTM code is from my Assignment 2.
- The final answer averages the sentence probabilities of three models: 0.6 distilled Arch 2, 0.2 Arch 4 with LoRA (rr1) and 0.2 Arch 4 with longer adaptation and LoRA (x2).
- A Viterbi decoder then uses label transition counts from the training documents.
- A simple rule marks a Dissent block: it starts at a sentence like "Name, Member (dissenting)" and runs to the end of the document.
- All 13 labels can be predicted. Leaderboard score: 0.6345.

## Models I tried

| Phase 1 model | Technique | Codabench |
|---|---|---|
| Arch 1 | LoRMA+ | 0.1103 |
| Arch 3 (encoder only) | IA3 | 0.4479 |
| Arch 4, bidirectional | LoRMA+ (rr2b) | 0.4739 |
| Arch 4, bidirectional, longer adaptation | LoRA (x2) | 0.5129 |
| Arch 4, bidirectional | LoRA (rr1) | 0.5139 |
| Arch 2 | LoRA, distilled from rr1 and x2 | 0.5354 |

The bidirectional Arch 4 comes from Phase 2. For x2 it was adapted for longer, with the Phase 3 training text added. For the distilled Arch 2, the teacher is the average of rr1 and x2, and the student learns half from the gold labels and half from the teacher.

## Training setup

- Best single model: Arch 2 with LoRA, distilled from two Arch 4 runs (loss 0.5 cross entropy plus 0.5 KL at temperature 2).
- Learning rate 5e-4, up to 64 sentences per training piece, 6 epochs, backbone dropout 0.1.
- Each sentence is cut to 128 tokens, sentences are packed into windows, and each sentence vector is the mean of its token states.
- Decoding: Viterbi with label transitions counted on the training documents.
- All runs were on one Kaggle T4 GPU with fp16 mixed precision. Task runs use AdamW with weight decay 0.01, gradient clipping 1.0, a linear warmup then linear decay, and learning rate 1e-3 for the task head.

## Files

| File | What it does |
|---|---|
| `backbone.py` | piece tokenizer, loads a Phase 1 model as a sentence encoder, path constants |
| `data.py` | loads documents, tokenizes, packs sentences into windows, counts label transitions |
| `model.py` | LSTM, CRF and the role model |
| `peft_methods.py` | LoRMA+ layer and the code that adds LoRMA+, LoRA or IA3 |
| `train.py` | loss, training loop, the settings of the single runs |
| `distill.py` | teacher probabilities and the distilled Arch 2 |
| `decode.py` | CRF prediction, Viterbi with counted transitions, macro-F1 |
| `rules.py` | Dissent rule |
| `predict.py` | the final ensemble and the best submission |
| `experiments/` | code for my other submissions, listed under Other submissions below |

## How to run

Dependencies: Python 3, `torch`, `numpy`, `scikit-learn` and `peft`. Training needs a GPU (I used a Kaggle T4). On Kaggle, uninstall `torchao` first, or `peft` crashes.

Path constants at the top of the files:

- `DATA_DIR = "data"` in `data.py`: `CL_train.json`, `CL_dev.json`, `CL_test.json`, `IT_train.json`, `IT_dev.json`, `IT_test.json`.
- `BPE_PATH = "tokenizer/legal_bpe_16384.json"` in `backbone.py`. The same folder needs `bpe_tokenizer.py` (a copy of `../phase1/tokenizer.py`).
- `KIT_DIR = "phase2_kit"` in `backbone.py`: folders `arch1/`, `arch2/` and `arch3/`, each with `model_arch.py` (a copy of `../phase1/archN_model.py`) and `model.pt`. Arch 3 also needs `metrics.json`.
- `ARCH4_MNTP_DIR = "arch4_mntp"` and `ARCH4_MNTP_P3_DIR = "arch4_mntp_p3"` in `backbone.py`: each with `model_arch.py`, `model.pt`, `metrics.json` and `arch4_engram_compress.pt`.
- `OUT_DIR = "runs"` in `train.py`, and `RUNS_DIR = "runs"` and `SUBMISSION_PATH = "submission_phase3.zip"` in `predict.py`. `RUNS_DIR` must match `OUT_DIR`.

Commands for the best submission, in this order:

```
python train.py rr1
python train.py x2
python distill.py teacher
python distill.py student
python predict.py
```

The other runs can be trained on their own: `python train.py rr2b`, `python train.py rr3` and `python train.py rr2`. Each `train.py` run writes its probabilities, model and submission zip to `runs/`. `predict.py` only reads saved probabilities, so it runs on a CPU in seconds.

## Submissions this code reproduces

The single model rows were scored before I added the Dissent rule, so rerunning them now gives slightly different files. The ensemble row is my leaderboard submission and is written by `predict.py`.

| Submission | Codabench |
|---|---|
| Arch 1, LoRMA+ (rr2) | 0.1103 |
| Arch 3 encoder, IA3 (rr3) | 0.4479 |
| Arch 4, LoRMA+ (rr2b) | 0.4739 |
| Arch 4 longer adaptation, LoRA (x2) | 0.5129 |
| Arch 4, LoRA (rr1) | 0.5139 |
| Distilled Arch 2 | 0.5354 |
| Ensemble, Viterbi and Dissent rule (leaderboard) | 0.6345 |

## Other submissions

These are my other Phase 3 submissions. The scripts in `experiments/` run from this folder after the five commands above, and the last column says which extra run a row needs first. `python -m experiments.ensembles NAME` writes `runs/experiments/NAME.zip`, and `python -m experiments.ensembles` with no name writes all of them.

| Submission | Codabench | How to reproduce |
|---|---|---|
| Arch 4 + LoRA (rr1), backbone frozen for the first 2 epochs | 0.4883 | `python -m experiments.extra_runs rr1_frozen` |
| Average of rr1 and x2 | 0.5245 | `python -m experiments.ensembles submission_ens_rr1_x2` |
| Average of rr1, x2 and the frozen-start rr1 | 0.5057 | after `rr1_frozen`: `python -m experiments.ensembles submission_ens_rr1_x2_rerun` |
| Average of rr1 and distilled Arch 2 | 0.5258 | `python -m experiments.ensembles submission_ens_rr1_kd` |
| Average of rr1, x2 and distilled Arch 2 | 0.5310 | `python -m experiments.ensembles submission_ens_rr1_x2_kd` |
| rr1 + x2, ArgumentPetitioner +0.3 in log probability | 0.5111 | `python -m experiments.ensembles bias_argpet_up` |
| rr1 + x2, RulingByLowerCourt +0.5 | 0.5237 | `python -m experiments.ensembles bias_lowercourt_up` |
| rr1 + x2, None +1.5 | 0.5274 | `python -m experiments.ensembles bias_none_up` |
| rr1 + x2, RatioOfTheDecision -0.5 | 0.5189 | `python -m experiments.ensembles bias_ratio_down` |
| Distilled Arch 2 with overlapping windows, CRF Viterbi | 0.5445 | `python -m experiments.extra_runs kd_overlap` |
| Distilled Arch 2 with overlapping windows (kdov), argmax | 0.5444 | `python -m experiments.ensembles 01_kdov_argmax` |
| 0.7 kdov + 0.3 rr1 | 0.5425 | `python -m experiments.ensembles 02_kdov7_rr1_3` |
| 0.6 kdov + 0.2 rr1 + 0.2 x2 | 0.5381 | `python -m experiments.ensembles 07_kdov6_rr1_2_x2_2` |
| 0.7 kdov + 0.3 rr1, geometric mean | 0.5432 | `python -m experiments.ensembles 13_geo_kdov7_rr1_3` |
| 0.6 kdov + 0.2 rr1 + 0.2 x2, Viterbi with counted transitions (0.3) | 0.5524 | `python -m experiments.ensembles 16_kdov6_rr1_2_x2_2_transvit03` |
| kdov, prior correction + transition Viterbi (0.5) | 0.5507 | `python -m experiments.ensembles 18_kdov_em_transvit05` |
| 0.5 kdov + 0.5 rr1, prior correction | 0.5317 | `python -m experiments.ensembles 19_kdov5_rr1_5_em` |
| 0.6 kdov + rr1, x2, rr2b (0.133 each), transitions 0.4 | 0.5460 | `python -m experiments.ensembles 1_kdov6_rr1_x2_lorma_transvit04` |
| 0.7 kdov + 0.15 rr1 + 0.15 x2, transitions 0.5 | 0.5479 | `python -m experiments.ensembles 2_kdov7_rr1_15_x2_15_transvit05` |
| 0.5 kdov + rr1, x2, rr2b (0.167 each), transitions 0.2 | 0.5469 | `python -m experiments.ensembles 3_kdov5_rr1_x2_lorma_transvit02` |
| 0.6 kdov + 0.2 rr1 + 0.2 x2, transitions 0.4 | 0.5506 | `python -m experiments.ensembles 4_kdov6_rr1_2_x2_2_transvit04` |
| 0.6 kdov + 0.2 rr1 + 0.2 x2, transitions 0.2 | 0.5480 | `python -m experiments.ensembles 5_kdov6_rr1_2_x2_2_transvit02` |
| 0.5 kdov + 0.25 rr1 + 0.25 x2, transitions 0.3 | 0.5522 | `python -m experiments.ensembles 6_kdov5_rr1_25_x2_25_transvit03` |
| b4: Arch 2, round-2 teacher, contrastive + prototype losses, train + dev | 0.5010 | `python -m experiments.extra_runs b4`, then `python -m experiments.ensembles 1_b4_transvit03` |
| 0.6 b4 + 0.2 rr1 + 0.2 x2 | 0.5001 | after `b4`: `python -m experiments.ensembles 3_b4_6_rr1_2_x2_2_transvit03` |
| 0.3 b4 + 0.3 kdov + 0.2 rr1 + 0.2 x2 | 0.5165 | after `b4`: `python -m experiments.ensembles 4_b4_3_kdov_3_rr1_2_x2_2_transvit03` |
| 0.5 b4 + 0.5 kdov | 0.5276 | after `b4`: `python -m experiments.ensembles 5_b4_5_kdov_5_transvit03` |
| 0.6 kdov + 0.2 rr1 + 0.2 x2, smoothed inside shift-head blocks (0.9) | 0.5437 | after `kd_overlap`: `python -m experiments.ensembles 1_ens_shiftsmooth09` |
| Same file submitted again | 0.5437 | same as the row above |
| Same, smoothing 0.7 | 0.5525 | after `kd_overlap`: `python -m experiments.ensembles 2_ens_shiftsmooth07` |
| Same, hard blocks at shift probability 0.2 | 0.5203 | after `kd_overlap`: `python -m experiments.ensembles 3_ens_shiftblocks02` |
| b3: Arch 2, round-2 teacher, contrastive + prototype losses, kNN | 0.5244 | `python -m experiments.extra_runs b3` (writes `runs/submission_b3.zip`) |
| 0.4 b3 + 0.3 kdov + 0.15 rr1 + 0.15 x2 | 0.5278 | after `b3`: `python -m experiments.ensembles 1_b340_kdov30_rr115_x215_transvit03` |
| 0.3 b3 + 0.3 kdov + 0.2 rr1 + 0.2 x2 | 0.5282 | after `b3`: `python -m experiments.ensembles 2_b330_kdov30_rr120_x220_transvit03` |
| 0.6 kdov + 0.2 rr1 + 0.2 x2, Dissent allowed, no rule | 0.5972 | `python -m experiments.ensembles best_with_dissent_allowed` |
| Half the best trio, half the six big-run models, Dissent rule | 0.6321 | `python -m experiments.extra_runs big`, then `python -m experiments.ensembles 1_half_oldbest_half_new6_transvit_dissent` |
| 11 runs with equal weights, Dissent rule | 0.6286 | after `big`, `b3`: `python -m experiments.ensembles 2_all11_equal_transvit_dissent` |
| Six big-run models + kdov + rr1 + x2, equal weights, Dissent rule | 0.6295 | after `big`: `python -m experiments.ensembles 3_new6_kdov_rr1_x2_transvit_dissent` |
| Separate blends for CL (picked on CL dev) and IT (0.3 kdov + 0.7 big-run models) | 0.6302 | after `big`: `python -m experiments.ensembles 5_perdomain_blend` |
| Distilled Arch 2 refit on train + dev, alone | 0.6068 | `python -m experiments.extra_runs refit`, then `python -m experiments.ensembles 4_r1_alone` |
| Best trio, transition matrices counted per domain | 0.6300 | `python -m experiments.ensembles old_trio_perdomain_cl03` |
| Best trio + 0.25 fixed-class-weight Arch 2 | 0.6340 | `python -m experiments.extra_runs fixed_weights`, then `python -m experiments.ensembles 1_trio_plus_fixedweights` |
| Fixed-class-weight Arch 2 in place of kdov | 0.6094 | after `fixed_weights`: `python -m experiments.ensembles 2_fixedweights_replaces_kd` |
| kdov and fixed-class-weight Arch 2 equal, + rr1 + x2 | 0.6254 | after `fixed_weights`: `python -m experiments.ensembles 3_kd_fixedweights_rr1_x2_equal_arch2` |
| Best trio + fixed-class-weight Arch 2 + six big-run models | 0.6295 | after `fixed_weights`, `big`: `python -m experiments.ensembles 4_trio_fixedweights_new6` |
