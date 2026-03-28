# Clip-Matched Late QAT

## Hypothesis

The repo's strongest pre-TTT stack already shows that low-bit export quality is the main remaining bottleneck, but the prior "Late QAT" path appears to have been neutralized by a `torch.compile` constant-folding issue. This candidate keeps the strong 2026-03-22 architecture intact and instead focuses on making late QAT both more faithful to the deployed int6 quantizer and more robust under compilation.

The core bet is that a **progressive, compile-safe fake-quant blend** plus **GPTQ-lite-matched per-row clip search** can reduce the train/export mismatch without paying for a broad new architecture.

## Why this is promising for this repository

Repository evidence points in the same direction:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly notes that the previous Late QAT path likely never activated because `torch.compile` constant-folded the class attribute toggle.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` improved the frontier by making the **export quantizer** smarter via per-row clip-percentile search.
- Multiple earlier records improved primarily by reducing the post-quantization penalty rather than by inventing a brand-new training stack.

That makes this repository a particularly good fit for a candidate that narrows the gap between **training-time fake quantization** and **the final int6 export path**.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen base implementation.
  - Supplies the 11-layer EMA + GPTQ-lite stack and the existing per-row percentile search used at export time.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Important negative result: the README documents why the earlier Late QAT toggle likely had no effect under `torch.compile`.

- `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`
  - Shows there is real upside when the model actually trains against low-bit noise.

## External research that informed it

- **PACT** — learnable/controlled clipping helps low-bit quantization stay accurate.  
  https://arxiv.org/abs/1805.06085

- **LSQ (Learned Step Size Quantization)** — training quality improves when the quantizer itself is treated as a first-class object instead of a fixed afterthought.  
  https://arxiv.org/abs/1902.08153

- **QDrop** — smoother low-bit adaptation benefits from avoiding brittle, all-at-once quantization behavior.  
  https://arxiv.org/abs/2203.05740

- **OmniQuant** — weight clipping and better alignment between the quantizer and deployed weights matters, especially at aggressive bit widths.  
  https://arxiv.org/abs/2308.13137

This candidate does **not** implement those papers wholesale. Instead, it adapts the parts that match this repo's constraints: progressive fake-quant activation, better clipping alignment, and minimal code growth.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in `train_gpt.py`:

1. Replaced the compile-fragile class-attribute Late QAT toggle with **per-module QAT buffers**.
2. Switched Late QAT from a hard boolean flip to a **progressive `qat_mix` ramp** as the wallclock-aware LR scale approaches zero.
3. Reused the same **GPTQ-lite per-row percentile clip search** for the QAT scale refresh path, so fake quant is closer to the exported int6 quantizer.
4. Matched the fake-quant range to the deployed export grid (`[-31, 31]`) instead of the old asymmetric `[-32, 31]` path.
5. Added a **FlashAttention fallback** to PyTorch SDPA for import-time / CPU component smoke tests when `flash_attn_interface` is unavailable.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603280423_clip-matched-late-qat

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
LATE_QAT_THRESHOLD=0.15 QAT_REFRESH_EVERY=64 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

- `QAT_ENABLED=1` to start with full fake quant from step 0.
- `QAT_REFRESH_EVERY=32` or `128` to trade off fidelity vs. refresh overhead.
- `LATE_QAT_THRESHOLD=0.10` or `0.20` to move the ramp window later or earlier.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202603280423_clip-matched-late-qat/train_gpt.py
```

Outcome:

- Passed.

I also attempted a minimal CPU-side import/component smoke, but this workflow environment does not currently have the declared runtime dependencies installed:

```bash
python - <<'PY'
import importlib.util
print('torch_spec', importlib.util.find_spec('torch'))
print('sentencepiece_spec', importlib.util.find_spec('sentencepiece'))
PY
```

Outcome:

- `torch_spec None`
- `sentencepiece_spec None`

So a true import-time smoke test was **not feasible in this container** even though `requirements.txt` declares both packages.

## Main expected risks and tradeoffs

- The improvement is still speculative until someone runs the full 8xH100 path.
- Refreshing per-row QAT scales adds a small amount of extra work late in training.
- Because the QAT blend is now progressive, the effect could be smaller than a fully tuned LSQ-style implementation.
- The CPU/SDPA fallback exists for validation convenience, not as the intended benchmark path; leaderboard runs should still use the optimized CUDA environment.
