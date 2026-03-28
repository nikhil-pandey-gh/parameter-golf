# Compile-Safe Progressive Late QAT

## Hypothesis

One of the strongest clean non-TTT training stacks in this repository already combines 11 layers, XSA, partial RoPE, LN scaling, EMA, and GPTQ-lite, but its intended late QAT path never actually fired under `torch.compile` because the toggle was implemented as a class-level Python boolean. A compile-safe, progressively increasing fake-quantization strength during warmdown should reduce the final int6 roundtrip gap without paying the optimization penalty of hard STE quantization from step 0.

## Why this is promising for this repository

Repository evidence points to quantization as the central bottleneck once the architecture is strong. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is a particularly clean base because GPTQ-lite plus EMA already squeezed out another ~0.0013 BPB without relying on TTT, while `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly documents that its late QAT branch was dead-code-eliminated by `torch.compile`. That means the repository already has strong evidence that this direction is promising but not yet actually tested in a compile-safe way.

Recent quantization literature also reinforces this direction. LSQ shows that quantizer-in-the-loop training can recover low-bit accuracy (`arXiv:1902.08153`), while HESTIA argues that hard STE quantization from the start of training harms optimization and that progressively hardened quantization is a better path for low-bit LLMs (`arXiv:2601.20745`). GPTQ (`arXiv:2210.17323`) and newer format-aware rounding work such as FAAR (`arXiv:2603.22370`) further support the idea that better alignment between training/export quantization and final rounding should pay off.

## Prior records or candidates that influenced this

There were no prior folders under `candidates/` when this candidate was created.

The main local influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` (for the broader context that recent wins are coming from small, targeted improvements layered on top of the mature 11-layer stack)

## External research that informed this

- **LSQ** (`arXiv:1902.08153`): quantization-aware training can substantially recover low-bit accuracy when the quantizer is part of optimization.
- **GPTQ** (`arXiv:2210.17323`): stronger post-training quantization and better clipping/rounding materially reduce accuracy loss in GPT-style models.
- **HESTIA** (`arXiv:2601.20745`): low-bit LLM QAT benefits from delaying or softening quantization early in training, then progressively hardening it later.
- **FAAR** (`arXiv:2603.22370`): quantization quality improves when the optimization is aware of the target rounding/discretization behavior instead of relying on naive rounding.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. Replaced the compile-folded class-level QAT boolean with a **compile-safe scalar input** (`qat_strength`) threaded through the model.
2. Switched from a binary late-QAT toggle to a **progressive late-QAT ramp**: fake quantization strength is `0` above `LATE_QAT_THRESHOLD` and then increases linearly to `1` as LR scale approaches `0`.
3. Explicitly **primes the late-QAT compiled signature before timed training starts**, so the first positive `qat_strength` step does not trigger a fresh `torch.compile` specialization inside the 600-second budget.
4. Added a small **FlashAttention import fallback** to `scaled_dot_product_attention` so the script is less brittle in environments that have PyTorch but not `flash_attn_interface`.
5. Kept the rest of the strong 2026-03-22 stack intact: 11 layers, XSA on the last 4 layers, partial RoPE, LN scale, EMA, GPTQ-lite int6 export, VE128, BigramHash, SmearGate, and warmdown 3500.

## How to run or evaluate it

From the repository root:

```bash
RUN_ID=compile_safe_late_qat \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
  candidates/202603282057_compile-safe-late-qat/train_gpt.py
```

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603282057_compile-safe-late-qat/train_gpt.py
```

Outcomes:

- Baseline repository compile check passed.
- Candidate `train_gpt.py` compile check passed.
- These checks validate **syntax only**. They do **not** exercise the new runtime-only surfaces introduced by this candidate, including the `torch.compile` call path that threads `qat_strength` through the model and the `scaled_dot_product_attention(..., enable_gqa=...)` fallback used when `flash_attn_interface` is unavailable.
- A minimal runtime smoke test was **not feasible on this runner** because the environment does not currently have `torch` installed, so the script cannot be imported or executed even before reaching CUDA- or FlashAttention-specific logic. The next recommended validation is a tiny real PyTorch run in an environment that at least matches the repository's normal dependency set.

## Main expected risks or tradeoffs

- The progressive ramp is a simplified approximation of more sophisticated differentiable or Hessian-aware low-bit schedules, so gains may be smaller than the recent QAT literature suggests.
- Threading `qat_strength` through the compiled graph still relies on `torch.compile` accepting both warmed signatures as expected; this candidate primes that path before the timer starts, but the exact runtime cost still needs a real GPU run.
- Because this candidate keeps the 2026-03-22 export path unchanged, the win depends on better pre-alignment to the existing GPTQ-lite/int6 export rather than on a more powerful export algorithm.
