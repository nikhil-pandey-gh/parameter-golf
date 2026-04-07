# 202604071626_awq-lite-channel-scale

## Hypothesis

The strongest remaining bottleneck in this repository is not raw pre-quant model quality, but the gap between a strong fp/bf16 checkpoint and the final low-bit artifact. This candidate keeps a proven 11-layer EMA/XSA/Partial-RoPE stack, adds the recent **LeakyReLU(0.5)^2** MLP win from the current best record, and introduces an **AWQ-lite activation-aware export transform** that rescales each linear layer's input channels before int6 quantization without changing the exact full-precision function.

## Why this is promising here

Repository history keeps converging on the same pattern:

- better training stacks continue to help, but **quantization/export quality is still a major limiter**;
- **EMA / SWA / GPTQ-lite / fp16-sensitive tensors** repeatedly buy small but real gains;
- naive recurrence / looping looks weak in a strict 10-minute budget, so the next candidate should prefer **post-training compression improvements** over a more invasive architectural rewrite.

That makes this repo a good fit for an AWQ/SmoothQuant-style idea: keep the training recipe stable, then make the exported low-bit weights easier to quantize.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the direct base because it is a strong, self-contained 11-layer stack with EMA, GPTQ-lite int6 export, warmdown 3500, XSA, partial RoPE, LN scale, BigramHash, and value embeddings.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the **LeakyReLU(0.5)^2** MLP change, which was already strong on the current SOTA stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - reinforced that **partial RoPE + LN scale** remain good zero-parameter structure.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - confirmed the value of **EMA + deepest-layer XSA**.
- No prior `candidates/` directory existed in this repository when this candidate was created.

## External research that informed it

- **SmoothQuant** (Xiao et al., 2023, arXiv:2211.10438): equivalent channel rescaling to move quantization difficulty away from sensitive directions.
- **AWQ** (Lin et al., 2024, arXiv:2306.00978): activation-aware channel scaling for weight-only low-bit quantization.
- **RPTQ** (Yuan et al., 2023, arXiv:2304.01089): channel-range imbalance matters for low-bit quantization, not just a few isolated outliers.

I also considered grouped cross-layer sharing ideas from **ALBERT**-style literature, but repository evidence argues against betting this candidate on more layer reuse: the existing recurrence/looping experiments under fixed wall-clock were negative, while quantization/export improvements kept paying off.

## What changed versus the chosen base

Relative to the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` script, this candidate adds:

1. **LeakyReLU(0.5)^2 MLP**
   - replaces `relu(x)^2` with `leaky_relu(x, 0.5)^2`.

2. **AWQ-lite activation-aware channel scaling**
   - collects per-layer input RMS statistics on a few train batches after EMA is applied;
   - computes per-input-channel scales from activation magnitude and weight RMS;
   - folds those scales into each linear weight matrix and stores a tiny inverse `input_scale` buffer in the module;
   - keeps the exact fp/bf16 function unchanged while making the exported weights friendlier to int6 quantization.

3. **FlashAttention fallback**
   - if `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA so the script can still be smoke-tested or iterated on outside the exact leaderboard environment.

4. **CPU smoke-test mode**
   - `SMOKE_TEST=1 python train_gpt.py` runs a tiny random-input forward/export/roundtrip check without requiring CUDA or dataset files.

## How to run

From this candidate directory:

```bash
cd candidates/202604071626_awq-lite-channel-scale

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_WD=0.04 ADAM_WD=0.04 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
MLP_NEGATIVE_SLOPE=0.5 \
AWQ_ENABLED=1 AWQ_ALPHA=0.5 AWQ_MAX_SCALE=4.0 \
AWQ_CALIBRATION_BATCHES=4 AWQ_CALIBRATION_TOKENS=131072 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The script still prefers FlashAttention 3 when it is available.
- The export path remains mixed int6/int8 + zstd like the chosen base, but now with activation-aware preconditioning before quantization.

## Lightweight evaluation / smoke commands

Syntax check:

```bash
python -m compileall train_gpt.py
```

CPU-only smoke:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation run in this workflow

- `python -m compileall candidates/202604071626_awq-lite-channel-scale/train_gpt.py` -> passed
- `SMOKE_TEST=1 python candidates/202604071626_awq-lite-channel-scale/train_gpt.py` -> passed, printing `smoke_test_ok loss:4.8460 mean_abs_diff:0.0000 logits_shape:(2, 32, 128)`

The smoke path intentionally checks the new export logic as well as the forward pass.

## Main expected risks / tradeoffs

- **Calibration sensitivity:** the AWQ-lite scales are heuristic and may need `AWQ_ALPHA`, `AWQ_MAX_SCALE`, or batch-count retuning.
- **Artifact overhead:** the per-layer `input_scale` buffers are small, but not free.
- **Interaction risk:** LeakyReLU^2 and activation-aware export may not stack as cleanly with the existing GPTQ-lite clip search as expected.
- **Eval/runtime overhead:** the extra per-layer input scaling is tiny, but non-zero.

## Added files

- `train_gpt.py`
- `README.md`
