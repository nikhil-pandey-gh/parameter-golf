# SVD rank-1 residual patches on top of GPTQ-lite

## Hypothesis

The best pre-TTT 11-layer stack in this repo is already strong enough that a lot of the remaining loss comes from the export path, not the training stack. A tiny number of structured high-precision residual patches should recover more of that loss-per-byte than uniformly raising precision everywhere.

This candidate keeps the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` training recipe and replaces the "pure rowwise int6 only" export assumption with a selective, data-free correction step:

1. GPTQ-lite per-row int6 quantization still produces the base artifact.
2. The script scores each quantized matrix by residual energy.
3. It adds a rank-1 fp16 SVD residual patch only to the top few matrices.

## Why this is promising for this repository

Three repo trends point in the same direction:

- The leaderboard improved rapidly once people attacked quantization/export directly (`fp16` embeddings, mixed int6/int8, zstd/lzma, GPTQ-lite) rather than only changing the training loop.
- The `2026-03-22` record is the strongest clean pre-TTT base, so it is the best place to isolate a new export-side idea.
- The `2026-03-23` winner suggests the training stack itself is already close to saturated; its extra gain mostly comes from LeakyReLU² and legal TTT, not a radically new artifact format.

So the next local delta worth trying is not "another small LR tweak", but "spend a few extra bytes much more intelligently."

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - Chosen as the direct base because it is the strongest pre-TTT export stack and already uses GPTQ-lite clip search.

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - Important evidence that the current architecture/training recipe is already strong enough that export-side improvements are a good next frontier.

- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow`
  - Reinforced the lesson that selective higher precision is often worth the bytes.

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600`
  - Earlier proof that protecting a small sensitive subset of weights can beat uniform low-bit quantization.

## External research that informed it

- **GlowQ: Group-Shared LOw-Rank Approximation for Quantized LLMs** (`arXiv:2603.25385`)
  - Motivated the idea that low-rank correction can recover quantization quality while staying cheaper than restoring every layer at higher precision.

- **Intrinsic Structure as a Proxy for Saliency: SVD-Based Weight Preservation for Mixed-Precision Quantization in Large Language Models** (`arXiv:2512.01343`)
  - Motivated a data-free selection rule based on matrix structure rather than calibration data.

- **HeRo-Q: A General Framework for Stable Low Bit Quantization via Hessian Conditioning** (`arXiv:2601.21626`)
  - Reinforced the broader lesson that low-bit PTQ fails in a few especially sensitive directions, so structured corrections are often more valuable than uniform bit increases.

I intentionally implemented the smallest repository-friendly version of these ideas: rank-1 residual patches selected by post-quantization residual energy, with no new calibration pipeline and no extra training infrastructure.

## What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

- adds **selective rank-1 SVD residual patches** for the top quantization-error matrices,
- keeps the base GPTQ-lite per-row int6 path intact for everything else,
- disables the previously inherited late-QAT default by setting `LATE_QAT_THRESHOLD=0.0` unless explicitly re-enabled,
- adds a **CPU / SDPA fallback** so the script can at least be imported and smoke-tested without FlashAttention 3,
- fixes default dataset/tokenizer paths so the script can be run directly from this candidate directory.

The main new knobs are:

- `LOWRANK_PATCH_ENABLED=1`
- `LOWRANK_PATCH_RANK=1`
- `LOWRANK_PATCH_TOPK=12`
- `LOWRANK_PATCH_MIN_NUMEL=131072`

## How to run or evaluate it

Run from this candidate directory:

```bash
cd candidates/202603290600_svd-rank1-gptq

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
LOWRANK_PATCH_ENABLED=1 LOWRANK_PATCH_RANK=1 LOWRANK_PATCH_TOPK=12 \
LOWRANK_PATCH_MIN_NUMEL=131072 LATE_QAT_THRESHOLD=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For very cheap local debugging, set `COMPILE_ENABLED=0`, use a tiny `ITERATIONS`, and reduce `LOWRANK_PATCH_TOPK`.

## Validation

I ran the following lightweight checks in this workflow:

```bash
cd /home/runner/work/parameter-golf/parameter-golf
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603290600_svd-rank1-gptq/train_gpt.py
```

Outcome: **passed**.

I also attempted a CPU-only import / forward / quantize smoke test for this candidate, but the workflow runner does not have `torch` installed in its Python environment:

```bash
python - <<'PY'
import torch
PY
```

Outcome: **failed with `ModuleNotFoundError: No module named 'torch'`**, so a real runtime smoke test was not feasible in this environment without installing heavyweight dependencies from the network.

## Main expected risks and tradeoffs

- The residual-energy heuristic is data-free, which keeps the implementation small, but it may choose the wrong matrices compared with a calibration-aware method.
- Rank-1 may be too weak; rank-2 or fewer/more selected layers could be better.
- The extra fp16 patch factors may improve roundtrip BPB but could still lose on total submission bytes if `TOPK` is too aggressive.
- Full SVD at export time adds some CPU-side overhead, though only once per final export.
- The CPU fallback is only for smoke/debug; competitive runs still expect CUDA and FlashAttention 3.
