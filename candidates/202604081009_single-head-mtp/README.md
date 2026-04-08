# Single-Head MTP on the 11L EMA + GPTQ-lite base

## Hypothesis

Add a **single training-only multi-token prediction (MTP) head** to the strongest clean pre-TTT stack so the trunk gets denser future-token supervision during the fixed 600-second training budget, then **strip that head before export** so the final artifact size stays effectively unchanged.

## Why this is promising here

The repository's recent gains have mostly come from evaluation/context handling, export-side quantization, and small architecture refinements on top of the same 11-layer backbone. That makes **sample efficiency inside the 10-minute training window** one of the more underexplored remaining levers.

This repo is unusually well set up for an MTP experiment:

1. The `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack is already a strong non-TTT base.
2. Its code already supports `mtp_heads` and already **drops them from the exported state dict** before quantization.
3. That means MTP can be tested as a **training-only auxiliary loss** with almost no artifact-budget downside.

## Prior records that influenced this candidate

- **`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`**  
  Chosen base. It already combines the best stable non-TTT ingredients: 11L/512d, MLP3x, XSA on the last 4 layers, partial RoPE, LN scaling, VE128, EMA + tight SWA, warmdown 3500, and GPTQ-lite export.
- **`records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`**  
  Reinforced that small zero- or near-zero-artifact changes can still move the frontier when layered on the 11L stack.
- **`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`**  
  Showed that top-end gains are now coming from targeted refinements rather than wholesale rewrites.
- **`records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`**  
  Explicitly argued against naive recurrence / layer reuse as the next easy lever, which pushed this candidate toward a training-objective change instead.

There were **no prior `candidates/` directories** in the repo when this candidate was created.

## External research that informed it

- **Fabian Gloeckle et al., _Better & Faster Large Language Models via Multi-token Prediction_**  
  arXiv:2404.19737 — https://arxiv.org/abs/2404.19737  
  Key takeaway used here: auxiliary future-token heads can improve sample efficiency and downstream capability without requiring a different model trunk.

I also considered more invasive compression ideas from **AWQ** (arXiv:2306.00978), **SmoothQuant** (arXiv:2211.10438), and **SpinQuant** (arXiv:2405.16406), but this repo already has a strong GPTQ-lite export path. MTP looked like the best next step that was both **distinct from existing records** and **cheap to integrate into the current codebase**.

## What changed vs. the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. `MTP_NUM_HEADS` default changed from `0` to `1`.
2. `MTP_LOSS_WEIGHT` default changed from `0.2` to `0.15` to keep the auxiliary objective conservative.
3. Added comments clarifying that MTP heads are **training-only** and intentionally excluded from export.

Everything else stays aligned with the strong non-TTT base:

- 11 transformer layers, 512 width, 8 heads / 4 KV heads
- 3x MLP, XSA on the last 4 layers
- partial RoPE (16/64), LN scaling
- SmearGate + BigramHash + VE128
- EMA + tight SWA
- GPTQ-lite mixed int6 export
- sliding-window evaluation

## How to run

From the repository root:

```bash
torchrun --standalone --nproc_per_node=8 \
  candidates/202604081009_single-head-mtp/train_gpt.py
```

Useful overrides if you want to ablate the idea:

```bash
# Disable the candidate's main change and recover the base behavior.
MTP_NUM_HEADS=0 MTP_LOSS_WEIGHT=0.0 \
torchrun --standalone --nproc_per_node=8 \
  candidates/202604081009_single-head-mtp/train_gpt.py
```

## Main risks / tradeoffs

- **Training-speed tax:** even one extra future-token head adds another output projection and loss, so any sample-efficiency win must beat the reduced step count.
- **Objective mismatch:** the auxiliary task may improve representation learning, but it can also over-regularize the trunk if the weight is too high.
- **No artifact win by itself:** unlike a quantization change, MTP only helps if it improves the trained trunk enough to survive export.

## Validation

- `python -m compileall candidates/202604081009_single-head-mtp/train_gpt.py`
- `python -m py_compile candidates/202604081009_single-head-mtp/train_gpt.py`

Both commands completed successfully.

A true CPU smoke run was **not feasible** without changing the candidate, because this trainer hard-requires CUDA and the FlashAttention 3 runtime path used by the chosen base.
