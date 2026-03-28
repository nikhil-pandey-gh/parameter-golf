# Candidate: MTP-lite auxiliary heads on the 2026-03-23 record stack

## Hypothesis

A lightweight multi-token-prediction (MTP) auxiliary objective can improve sample efficiency for this repository's 10-minute training budget without increasing final artifact size, because the extra prediction heads are used only during training and are excluded from export.

## Why it is promising for this repository

This repository already appears sample-limited: most of the biggest gains came from better use of the same 600-second training budget rather than from dramatically larger models. The strongest local stack already has mature quantization, EMA/SWA, XSA, Partial RoPE, and legal TTT, so a training-only objective is one of the few remaining levers that can plausibly improve pre-quant and post-quant quality without fighting the 16 MB cap.

A particularly attractive detail here is that the current record code already contains an MTP path and explicitly excludes `mtp_heads` from export. That means the idea can be tested with essentially zero artifact-size cost, provided the heads are actually optimized.

## Prior records and candidates that influenced this choice

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Best current local result (`val_bpb: 1.1194` post-TTT mean) and the direct code base for this candidate.
  - Contributed the 11-layer banked model, LeakyReLU(0.5)^2 MLP, legal score-first TTT, GPTQ-lite int6 export, EMA, Partial RoPE, XSA tail, BigramHash, and SmearGate.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Strongest non-TTT training/quantization baseline; confirms GPTQ-lite + EMA + long warmdown are part of the stable base.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - Important because it already carried MTP scaffolding, but logs show `mtp_num_heads:0`, so the idea was present in code but not actually exercised.

There were no prior `candidates/` directories to review in this repository snapshot.

## External research that informed this candidate

- Fabian Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (arXiv:2404.19737, 2024).
  - The core motivation for this candidate. The paper reports better sample efficiency from auxiliary future-token heads on top of a shared trunk, with the training-only heads removable at inference/export time.
- Zhenzhong Lan et al., *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations* (arXiv:1909.11942, 2019).
  - Considered as an alternative direction because cross-layer sharing is compatible with this repo's banked design, but local history suggests more aggressive depth reuse is riskier than turning on a lightweight auxiliary objective.
- Mostafa Dehghani et al., *Universal Transformers* (arXiv:1807.03819, 2018).
  - Also considered as a depth-reuse alternative. I did not choose it for this candidate because local notes already flag recurrence/depth-reuse as a likely dead end under this challenge's wallclock constraints.

## What changed versus the chosen base implementation

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

- turns on a **default MTP-lite setting** (`MTP_NUM_HEADS=1`, `MTP_LOSS_WEIGHT=0.15`),
- fixes the **optimizer wiring** so MTP heads are actually trained instead of remaining inert,
- keeps export behavior unchanged so `mtp_heads` are still omitted from the final artifact,
- leaves the rest of the record stack intact to isolate the effect of the auxiliary objective.

## How to run or evaluate it

Training/evaluation entrypoint from this directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides if the auxiliary loss is too strong or too slow:

```bash
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py   # disable MTP
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Completed lightweight validation:

- `python -m compileall candidates/202603281213_mtp-lite-aux/train_gpt.py`
  - **Passed**.
- Attempted CPU smoke test by importing the candidate module and instantiating a tiny `GPT` on CPU.
  - **Could not run in this workflow environment** because the runner's default `python` is missing required packages from `requirements.txt`, including `torch`, `numpy`, and `sentencepiece`.
- Candidate code review before issue/PR creation.
  - **Completed cleanly**; no significant issues were reported.

## Main expected risks or tradeoffs

- The auxiliary heads add training compute, so wallclock-capped training may get fewer optimizer steps.
- If the auxiliary objective is overweighted, it could help pretraining loss but hurt the final exported int6 model or downstream TTT behavior.
- The current repo's MTP path existed but was unused; this candidate fixes the obvious optimizer issue, but full leaderboard-quality validation still needs a real GPU run.
