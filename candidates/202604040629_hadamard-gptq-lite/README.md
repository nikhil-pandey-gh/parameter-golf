# Hadamard GPTQ-lite on the LeakyReLU2 + legal TTT base

## Hypothesis

The current best stack is already strong on training-time architecture and eval-time adaptation, so the best remaining low-risk lever is the **int6 export path**. This candidate tests whether a **data-free block-Hadamard rotation search** before GPTQ-lite per-row clip selection can flatten weight outliers enough to reduce post-quantization error and improve final sliding-window BPB without slowing training.

## Why this is promising for this repository

The repo history repeatedly says the same thing: quantization/export details move the leaderboard a lot.

- `2026-03-18_FP16Embed_WD3600` showed that preserving the most sensitive tensors during export collapsed the quantization gap.
- `2026-03-19_MixedQuant_Int6Int8_SlidingWindow` and later int6/QAT runs showed that better low-bit choices unlocked materially stronger models under the same 16 MB cap.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved again with a better **clip-search PTQ** pass.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` is the strongest overall base, but it still depends on the same int6/lzma roundtrip path before final evaluation.

Given that pattern, a better **preconditioning step for the existing GPTQ-lite exporter** is more aligned with the repository's winning trends than a broad new architecture.

## Influencing records and prior candidates

There were **no prior `candidates/` directories** in the repository when this candidate was created.

The main local influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` — chosen base because it is the current best overall stack.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` — established GPTQ-lite percentile search as a durable quantization win.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/` — reinforced that export-time tensor handling can be worth several basis points of BPB.

## External research that informed the idea

- **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) showed that orthogonal rotations can remove outliers without changing the full-precision computation, making low-bit quantization easier.
- **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)) showed that some rotations quantize much better than others, motivating a search over inexpensive rotation choices instead of using one fixed transform.
- **KurTail** ([arXiv:2503.01483](https://arxiv.org/abs/2503.01483)) further strengthened the case that outlier-focused rotation optimization can beat stronger quantization baselines at low bitwidth.
- **PolarQuant** ([arXiv:2603.29078](https://arxiv.org/abs/2603.29078)) is especially relevant here because it is a **weight-only PTQ** result and explicitly reports that **Hadamard rotation alone accounts for most of the gain**, which fits this repo's need for a cheap, no-calibration export tweak.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps the training stack, legal TTT path, and overall artifact format intact, and only changes the int6 export logic:

1. Added `HADAMARD_GPTQ` (default `1`) and `HADAMARD_BLOCK_SIZES` (default `512,256,128`) knobs.
2. For each 2D int6 tensor in the MLP/attention categories, the exporter now compares:
   - the existing GPTQ-lite percentile search on the raw weights, and
   - GPTQ-lite after normalized **column-wise block-Hadamard** rotations for each allowed block size.
3. The exporter keeps whichever option gives the lowest reconstruction MSE after inverse rotation.
4. Rotation metadata is serialized and applied during dequantization before final evaluation.
5. Quantization logs now report which Hadamard block sizes were selected.
6. Fixed inherited exact-metric log labels that still said `int8/zlib` even though the chosen base uses `int6/lzma`.

This is intentionally a **minimal quantization-path change** rather than a new training recipe.

## How to run or evaluate

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
HADAMARD_GPTQ=1 HADAMARD_BLOCK_SIZES=512,256,128 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To compare against the base exporter, rerun with `HADAMARD_GPTQ=0`.

## Validation commands and outcomes

- `python -m compileall candidates/202604040629_hadamard-gptq-lite/train_gpt.py` — **passed**.
- Attempted a small Python import/quantization roundtrip smoke test in an isolated venv — **blocked by missing runtime dependencies on this runner** (`torch` was not available in the local Python environment after creating the venv).
- A full CPU-only `main()` smoke test is **not feasible** for this candidate without materially changing the chosen base, because the script intentionally preserves the record codepath's CUDA/FlashAttention/NCCL requirement.

## Main expected risks or tradeoffs

- The new search only optimizes **weight reconstruction MSE**, which may not map perfectly to final BPB after sliding eval and legal TTT.
- Export time increases because each int6 tensor now tries several block sizes in addition to the existing percentile sweep.
- Some matrices may prefer no rotation at all, so the gain could be small or uneven.
- This is still a **data-free** approximation to the stronger learned-rotation literature; if it helps, the next step would be a slightly smarter per-layer objective rather than a larger architectural rewrite.
