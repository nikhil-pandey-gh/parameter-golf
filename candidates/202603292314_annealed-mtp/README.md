# Annealed training-only MTP on the PR #549 stack

## Hypothesis

A small **multi-token prediction (MTP)** auxiliary loss should improve sample efficiency on this repo's strongest 11-layer stack without increasing submission bytes, because the extra prediction heads are only used during training and are stripped before export.

This candidate makes that auxiliary loss practical in four ways:

- it enables a single extra MTP head by default,
- it actually optimizes the MTP head weights (the copied base stack instantiated them but did not put them in any optimizer),
- it linearly fades the auxiliary loss out during warmdown,
- and it hard-disables MTP in the final 5 seconds of wallclock-capped training so the last real optimizer updates are pure next-token.

## Why this is promising for this repository

The `records/` history shows that the biggest recent gains already came from export/eval engineering and targeted architecture tweaks:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`

That makes another byte-spending architecture rewrite less attractive than a **training-only** improvement. MTP is a good fit because:

- it adds no permanent artifact bytes,
- it reuses the existing trunk,
- and the copied winner already contains most of the plumbing, so the implementation stays minimal.

I specifically avoided full layer recurrence / broad parameter sharing because prior repo evidence says naive looping hurt badly under the 10-minute budget:

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`

## Prior records that influenced this candidate

### Primary base

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`

This is the copied base implementation. It already has the current best architecture stack plus dormant MTP codepaths.

### Supporting pre-TTT stack

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`

This reinforced the idea that the strongest recent progress came from squeezing more training quality out of the same byte budget.

### Negative guidance

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`

These made me reject recurrence / shared-depth as the next candidate and bias toward a training-only auxiliary objective instead.

## External research that informed it

- **Fabian Gloeckle et al., 2024, "Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`)
  - Trains a shared trunk with independent future-token heads.
  - Reports improved sample efficiency and stronger induction-style behavior.
  - Most relevant repo-fit insight: the auxiliary heads can be training-only, which matches the challenge's strict artifact budget.

- **Zhenzhong Lan et al., 2019, "ALBERT"** (`arXiv:1909.11942`)
  - Useful as a comparison point for parameter sharing as a bytes-saving strategy.
  - I considered this route, but the repo's own recurrence results made it a worse next bet than MTP.

- **Mostafa Dehghani et al., 2018, "Universal Transformer"** (`arXiv:1807.03819`)
  - Reinforces the appeal of recurrent/depth-reuse ideas in principle.
  - Again, repo evidence suggests the 10-minute training budget punishes naive reuse here, so I kept the candidate focused on auxiliary supervision instead.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`:

1. `MTP_NUM_HEADS` now defaults to `1` instead of `0`.
2. `MTP_LOSS_WEIGHT` now defaults to `0.15`.
3. Added `MTP_WARMDOWN_CUTOFF` (default `0.2`), which linearly anneals the auxiliary loss over the final low-LR portion of training.
4. Added `MTP_FINAL_ZERO_MS` (default `5000`), which forces pure next-token updates in the final wallclock slice.
5. MTP heads are now added to the AdamW-managed small-parameter group, so they actually train.
6. MTP heads are initialized from the main token-output projection weight, avoiding an initially dead auxiliary path.

The export path still excludes `mtp_heads`, so the submission artifact should remain governed by the base stack rather than by the auxiliary heads.

## How to run or evaluate it

From this candidate directory:

```bash
RUN_ID=annealed_mtp_seed1337 \
SEED=1337 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
# Turn the auxiliary objective off completely
MTP_NUM_HEADS=0

# Keep one auxiliary head but weaken it
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10

# Keep MTP active longer into warmdown
MTP_WARMDOWN_CUTOFF=0.10

# Shorten or extend the final pure-next-token phase
MTP_FINAL_ZERO_MS=2000
```

This candidate is intended to test **training-time quality before spending extra eval budget on TTT**. If it helps, it should remain compatible with the copied TTT path as a later follow-up experiment.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202603292314_annealed-mtp/train_gpt.py
python -m py_compile candidates/202603292314_annealed-mtp/train_gpt.py
```

Outcome:

- `compileall`: passed
- `py_compile`: passed

Minimal CPU smoke test:

- Not feasible in this workflow without changing the candidate. The local workflow Python environment does not have either `torch` or `flash_attn_interface` importable, and the real script is intentionally CUDA/Hopper-oriented. A CPU-only startup check here would require stubbing the actual runtime dependencies and would not be a faithful signal for the intended training path.

## Main expected risks and tradeoffs

- **Training-time overhead:** even one extra MTP head adds another vocab projection and cross-entropy pass, so step count may drop if the auxiliary objective is too expensive.
- **Objective mismatch:** MTP is not used at evaluation, so keeping it active too late could hurt final next-token quality. That is why this candidate fades it out during warmdown.
- **Head count sensitivity:** larger `MTP_NUM_HEADS` may help sample efficiency, but they are much riskier on a strict 600-second budget.
- **Strong baseline effect:** the copied base is already highly optimized, so gains may be small and could disappear if the extra compute costs too many steps.
