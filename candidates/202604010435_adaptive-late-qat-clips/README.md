# Adaptive Late-QAT Clips

## Hypothesis

The current best pure train/export record in this repository (`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`) already closes most of the quantization gap with EMA, tight SWA, GPTQ-lite percentile search, Partial RoPE, LN scale, XSA, and shared value embeddings. The remaining gap is that late QAT still uses a fixed row-max quantizer during training, while export uses a stronger post-training clip search.

This candidate tests a lightweight bridge between those phases: learn a per-row clip factor during late QAT, then feed those learned clip hints into the final GPTQ-lite export search. The goal is to make the training-time fake quantizer look more like the eventual exported quantizer without changing the inference-time architecture.

## Why it is promising for this repository

Repository review showed a very clear trend:

- winners come from quantization-aware model design rather than from raw extra compute,
- fixed STE/QAT plus smarter export quantization already works well,
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly documents that its late-QAT gate was compiled away under `torch.compile`,
- the best non-TTT stack (`1.1233`) still leaves room in the training/export interface rather than in the core architecture.

That makes compile-safe, learned clip-aware late QAT a targeted next step: it preserves the strongest known stack and attacks a quantization mismatch that the repo has not yet explored directly.

## Prior records and candidates that influenced this candidate

There were no prior entries under `candidates/` in this checkout.

The main record influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - base implementation copied for this candidate,
  - strongest pure train/export result in the repo,
  - established GPTQ-lite percentile search, EMA, tight SWA, 11L/512d/MLP3x/XSA4/Partial-RoPE/LN-scale/VE128 stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - specifically the README note that late QAT was dead-code-eliminated under `torch.compile`,
  - motivated making the QAT activation runtime-controlled and compile-safe.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - reviewed as the current best overall record,
  - not used as the base because it adds evaluation-time TTT and parameter-banking infrastructure, which is a broader jump than needed for this candidate.

## External research that informed it

The main external references were:

- `EfficientQAT: Efficient Quantization-Aware Training for Large Language Models` (arXiv:2407.11062)
  - especially the idea of separating regular training from an end-stage phase that optimizes quantization parameters.
- `SiLQ: Simple Large Language Model Quantization-Aware Training` (arXiv:2507.16933)
  - evidence that simple end-to-end QAT changes can produce strong gains without heavy architectural changes.
- `How to Parameterize Asymmetric Quantization Ranges for Quantization-Aware Training` (arXiv:2404.16898)
  - motivation for treating quantization ranges as learnable objects rather than fixed heuristics.

These papers were used as design guidance, not as a blueprint: the candidate keeps the repository's existing symmetric int6 export path and only adds lightweight learnable clip factors that fit the current codebase.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. Added per-row `qat_clip_logits` to each `CastedLinear`.
2. Added compile-safe runtime QAT gating via a tensor buffer (`qat_mix`) instead of a Python class flag.
3. Switched fake quantization to a branchless blend between fp32 weights and fake-quantized weights during training, so late QAT can activate at runtime under `torch.compile`.
4. Added a dedicated optimizer group for clip parameters (`QAT_CLIP_LR`, zero weight decay).
5. Added helper functions to activate late QAT across the model and collect learned clip hints for export.
6. Extended `quantize_int6_per_row()` and `mixed_quantize_int6()` so the exporter tries the learned clip hint alongside the existing GPTQ-lite percentile candidates.
7. Excluded training-only `qat_clip_logits` from export artifacts while allowing evaluation reloads with `strict=False` only for those missing keys.

## How to run or evaluate it

From the candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
QAT_ENABLED=0 QAT_CLIP_LR=0.01 QAT_CLIP_MIN_RATIO=0.25 QAT_CLIP_INIT=0.985 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate is intentionally still self-contained and run-compatible with the existing repository workflow: it changes the training/export quantization path, not the dataset pipeline or evaluation format.

## Main expected risks and tradeoffs

- The compile-safe gate computes fake-quantized weights even before late-QAT activation, then blends them out with `qat_mix=0`. That preserves behavior but may add some overhead.
- Learned clip factors may help the export quantizer, but they may also converge to a distribution that overlaps with GPTQ-lite's existing percentile search and therefore add little.
- Because the new clip parameters are training-only, the evaluation/export path now depends on the learned hints being useful enough to justify their optimizer state and forward-pass overhead.
- Full validation still needs a real CUDA environment with the repo's Python dependencies and FlashAttention-compatible runtime.

## Validation

Validation was run after implementation.

- `python -m compileall candidates/202604010435_adaptive-late-qat-clips/train_gpt.py`
  - Passed.
- CPU-side smoke test for the new QAT clip path
  - Not feasible in this runner.
  - The runner's default Python lacked the repository dependencies (`numpy`, `sentencepiece`, and `torch`).
  - An isolated venv under `/tmp/gh-aw/agent/` was created successfully, but installing CPU `torch` from `https://download.pytorch.org/whl/cpu` failed behind the workflow proxy (`403 Forbidden` / no matching distribution), so an import-level runtime smoke check could not be completed here.
  - Independently of that dependency issue, `main()` also requires CUDA, so a full training start check was not possible on this host.
