# Signed Block-Hadamard Export

## Hypothesis

The current record line has already harvested most of the obvious training-time wins: 11 layers, MLP3x, seq2048, sliding-window eval, XSA4, partial RoPE, LN scaling, EMA/SWA, GPTQ-lite clipping, and finally LeakyReLU^2 plus legal score-first TTT. The next low-risk lever is the **export path itself**. If the int6 roundtrip can see a smoother basis before per-row quantization, the model should lose fewer bits at the exact stage that decides the 16MB artifact score.

This candidate applies a **deterministic signed block-Hadamard rotation** to large attention/MLP weight matrices **only at export time**, quantizes in that rotated basis, and then inverse-rotates after dequantization. Training, architecture, and evaluation stay otherwise unchanged.

## Why this is promising for this repository

1. The repo's best recent gains already moved from architecture to **quantization-aware export details**: GPTQ-lite alone improved the 2026-03-22 record by about **-0.0006 BPB** with zero training cost.
2. No prior `records/` experiment and no prior `candidates/` directory used **Hadamard/rotation-based PTQ**.
3. Prior dead ends were mostly on the training side here: layer recurrence, plain LR retuning, and late-QAT plumbing that got constant-folded away. This candidate avoids those and only touches the export roundtrip.
4. The change is small enough to keep the repository's existing workflow intact: same `train_gpt.py` entrypoint style, same artifact pipeline, same int6/int8 packing, same TTT option.

## Prior experiments that influenced this candidate

| Source | What it contributed |
|---|---|
| `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` | Chosen base implementation because it is the best overall stack so far (`1.1194` mean post-TTT). This candidate keeps that stack and changes only export-time quantization. |
| `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` | Strong evidence that export-only quantization refinements still matter after the training stack is already good. GPTQ-lite is kept and paired with rotation. |
| `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` | Confirms the winning architectural core: XSA4 + partial RoPE + LN scale. |
| `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` and `2026-03-19_MLP3x_QAT_Int6_SlidingWindow` | Established the repo's successful compression-aware direction: MLP3x, int6 block weights, weight decay, and compression as first-class constraints. |
| `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090` | Helpful reminder that recurrence/depth reuse was a wall-clock loser here, so the candidate should improve the existing strong stack rather than add slower training structure. |

## External research that informed it

| Paper | Relevance |
|---|---|
| [QuaRot, arXiv:2404.00456](https://arxiv.org/abs/2404.00456) | Shows that orthogonal rotations can remove outliers and make PTQ dramatically easier, including essentially lossless 6/8-bit quantization with round-to-nearest. |
| [SpinQuant, arXiv:2405.16406](https://arxiv.org/abs/2405.16406) | Shows that the **choice of rotation matters** and that rotated weight spaces materially improve low-bit accuracy. |
| [SmoothQuant, arXiv:2211.10438](https://arxiv.org/abs/2211.10438) | Provides the broader justification for outlier migration / basis manipulation as a training-free quantization tool. |

This candidate is intentionally a **minimal adaptation** of those ideas for Parameter Golf: no learned rotations, no activation quantization, no new kernels. Just deterministic signed Hadamard blocks in the existing export/dequantize flow.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes:

1. Added three export knobs:
   - `EXPORT_ROTATE_ENABLED` (default `1`)
   - `EXPORT_ROTATE_BLOCK_SIZE` (default `512`)
   - `EXPORT_ROTATE_CATEGORIES` (default `attn,mlp`)
2. Added cached helpers that build a normalized Hadamard matrix and deterministic sign vectors driven by a persisted rotation seed.
3. Before int6/int8 quantization, large 2D tensors in the selected categories are rotated blockwise along the input-channel dimension.
4. Rotation metadata (including the deterministic rotation seed) is stored in the export metadata, and the inverse rotation is applied after dequantization before `load_state_dict`.
5. The existing GPTQ-lite clip search, lzma compression, legal TTT path, and parameter-banked training stack are left unchanged.

Notably, embeddings are **not rotated by default** to avoid tied-weight edge cases and keep the change surgical.

## How to run / evaluate

### Best-shot leaderboard-style run

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
EXPORT_ROTATE_ENABLED=1 EXPORT_ROTATE_BLOCK_SIZE=512 EXPORT_ROTATE_CATEGORIES=attn,mlp \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This base keeps **EMA enabled internally at decay `0.997`** and keys the late fake-quant branch off `LATE_QAT_THRESHOLD`, so those are documented here as behavior rather than extra env toggles.

### Pure-training comparison run

Use the same command with `TTT_ENABLED=0` if you want to isolate whether the new export path improves the pre-TTT stack as well.

## Main expected risks / tradeoffs

1. **The rotation is fixed, not learned.** SpinQuant suggests rotation quality matters; a deterministic signed Hadamard may help less than an optimized rotation.
2. **Export time goes up slightly.** The extra matrix multiplies happen only during export/dequantize, not during training.
3. **Benefit may concentrate in a few matrices.** If only a subset of attention/MLP weights are outlier-limited, the gain could be small or noisy.
4. **Compression may move in either direction.** The goal is lower roundtrip distortion, not necessarily smaller lzma blobs; the score win depends on final BPB, not just file size.

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604081951_signed-hadamard-export/train_gpt.py` | Succeeded. |
| CPU smoke test that imported the candidate and exercised the new rotation/quantization helpers | Not feasible in this runner because the local Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`). |
