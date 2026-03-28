# Compile-Safe Bank Late-QAT

## Hypothesis

The strongest reusable training stack in this repo already moved to parameter-banked weights, GPTQ-lite post-training quantization, EMA/SWA, partial RoPE, VE, and LeakyReLU². But its `late_qat_threshold` path only toggles `CastedLinear._qat_enabled`, so the dominant bank tensors never participate in fake quantization, and earlier record notes show that `torch.compile` can also erase late-QAT branches when they are toggled after tracing.

This candidate tests a narrower, more implementation-safe idea: **make late QAT actually hit the heavy bank tensors, and recompile exactly when it turns on so the QAT branch cannot be constant-folded away**.

## Why this is promising for this repository

- Repo evidence says plain architecture recurrence is risky under a 10-minute cap, so this candidate avoids that dead end.
- Recent winners already show that tiny improvements in the **quantization gap** matter at this score range.
- The best banked model line (`2026-03-23`) improved training throughput and pre-TTT quality, but still left the biggest tensors outside late-QAT.
- This change is small, local, and directly targets the artifact bottleneck that matters for the 16 MB submission limit.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Best overall score in the repo.
  - Contributed the banked model, Parallel Muon, and LeakyReLU² base.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best strong non-TTT line.
  - Motivated keeping GPTQ-lite-style compression awareness and late warmdown behavior.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Explicitly documents that a prior late-QAT path was compiled away.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - Good reminder that QAT overhead can outweigh benefit if applied too broadly or too early.

## External research that informed it

- Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (`arXiv:1712.05877`)
  - Supports the core idea of using straight-through fake quantization during training to reduce the post-quantization accuracy drop.
- Esser et al., *Learned Step Size Quantization* (`arXiv:1902.08153`)
  - Reinforces that quantizer-aware training can recover low-bit performance when the quantization path is part of optimization rather than only a post-hoc transform.
- Choi et al., *PACT* (`arXiv:1805.06085`)
  - Useful supporting evidence that low-bit training works best when the clipping/quantization behavior is explicitly represented during training.

This candidate does **not** implement full LSQ or PACT. Instead, it adopts the lighter-weight repo-friendly version: STE fake quantization on the bank tensors during the late warmdown phase.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added `ste_fake_quantize_int6()` for row-wise fake quantization of 2D/3D bank tensors.
2. Added `bank_qat_enabled` to `GPT` so the parameter banks can opt into late-QAT.
3. Routed `forward()` and `forward_logits()` through `_get_active_banks()`, which fake-quantizes the banks only when training with bank QAT enabled.
4. Replaced the old “flip a Python flag after compile” behavior with **recompile-on-enable** during late warmdown:
   - when `scale < LATE_QAT_THRESHOLD`, the script enables bank QAT,
   - enables the existing small `CastedLinear` QAT path too,
   - recompiles the model so `torch.compile` traces the QAT branch instead of erasing it.
5. Left the rest of the strong 2026-03-23 stack intact: LeakyReLU², banked weights, partial RoPE, XSA, VE, EMA/SWA, GPTQ-lite quantization, and optional legal TTT support.

## How to run or evaluate it

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional follow-up: compare against the full 2026-03-23 evaluation recipe by reusing its `TTT_*` flags after confirming the pre-TTT training path is stable.

## Validation

Ran:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603281920_compile-safe-bank-qat/train_gpt.py
```

Outcome:

- Passed for the root baseline files, `data/`, and this candidate `train_gpt.py`.

Attempted:

```bash
python - <<'PY'
# import candidate module and run a tiny CPU smoke
PY
```

Outcome:

- Not feasible in this workflow environment because the available Python interpreter does not have `torch` installed.
- The actual training entrypoint also hard-requires CUDA plus `flash_attn_interface`, so a faithful CPU startup check is not available here without adding infrastructure the repo does not currently use.

## Main expected risks or tradeoffs

- Recompiling at the late-QAT transition adds a one-time compile pause.
- Fake quantization of the bank tensors adds compute in the last training phase and may reduce end-of-run step count.
- The current fake-quant path uses simple row-max scaling, not a learned step-size method like LSQ.
- This is still unvalidated on real 8xH100 hardware, so the true pre/post-quant gap reduction is the main open question.
- Interaction with optional legal TTT is intentionally left as a follow-up rather than bundled into the core hypothesis.
