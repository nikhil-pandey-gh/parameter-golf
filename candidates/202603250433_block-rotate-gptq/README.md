# 202603250433 Block-Rotate GPTQ-lite

## Hypothesis

The best non-TTT record stack in this repo already does a strong job on architecture, training schedule, and mixed-precision export, but the remaining gap still looks heavily quantization-driven. A lightweight, data-free rotation step before mixed int6/int8 export should flatten per-row outliers enough to reduce round-trip error without paying extra training-time cost or introducing a new calibration pipeline.

Concretely, this candidate adds deterministic **block-Hadamard rotations with a tiny sign-search** on large 2D tensors before quantization, then inverts the rotation after dequantization. The goal is to preserve the strong `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` training recipe while improving the exported artifact quality.

## Why this is promising for this repository

The repo history strongly suggests that export quality is one of the last big levers:

- the baseline and early records improved a lot from better post-training quantization, fp16 embedding passthrough, and sliding-window evaluation;
- the strongest non-TTT runs kept pushing mixed int6/int8 export, GPTQ-lite clip search, EMA/SWA, and artifact-aware tuning;
- even the latest top runs still treat quantization as central rather than solved.

In short: this challenge rewards ideas that improve the compressed artifact without slowing down 10-minute training. Rotation-aware PTQ fits that shape well.

## Prior repo work that influenced this candidate

### Chosen base implementation

This candidate starts from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

I chose it because it is the strongest recent **non-TTT**, quantization-focused stack in the repo: 11 layers, XSA, EMA, partial RoPE, VE, mixed int6/int8 export, and GPTQ-lite clip search. That makes it the cleanest place to test a new export-side idea.

### Other record influences

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - reinforced that the current best stack is already strong on architecture/training and still sensitive to export details.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - shows that once the training stack is strong, marginal gains increasingly come from clever evaluation/export tricks rather than wholesale architectural rewrites.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
  - important reminder that the embedding/head path is especially quantization-sensitive.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - documents that naive layer recurrence was a dead end under a strict wall-clock budget, which pushed me away from deeper architecture changes.

There was **no existing `candidates/` directory** in this repo when this candidate was created, so there were no earlier candidate iterations to inherit from.

## External research that informed the idea

- **QuaRot** (`arXiv:2404.00456`) argues that orthogonal rotations can remove outliers and make even 6/8-bit quantization close to lossless in favorable settings.
- **SpinQuant** (`arXiv:2405.16406`) shows that not all rotations are equal: some random rotations are much better than others, and learned rotations improve PTQ quality further.
- **OptRot** (`arXiv:2512.24124`) pushes the idea further with cheap, data-free objectives for weight-quantization rotations, outperforming plain Hadamard baselines.
- **WUSH** (`arXiv:2512.00956`) and **Block Rotation is All You Need for MXFP4 Quantization** (`arXiv:2511.04214`) both motivate practical blockwise transforms instead of assuming one global fixed transform is always best.

This candidate does **not** attempt to reproduce the full learned/data-dependent methods from those papers. Instead, it takes the smallest adaptation that seems appropriate for this repo:

- no calibration set,
- no extra training,
- no new artifact tensors,
- deterministic per-tensor sign-search,
- blockwise Hadamard rotations that fit common tensor widths in this codebase.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate adds:

1. **Rotation-aware mixed quantization**
   - Large 2D tensors in selected categories (`attn`, `mlp`, `embed`) can be pre-rotated with a deterministic block-Hadamard transform before quantization.
   - A tiny per-tensor sign-search (`ROTATION_TRIALS`, default `4`) picks the best rotation by direct reconstruction MSE.
   - The chosen rotation is stored as small metadata only (`block_size`, `trial`), then inverted after dequantization.

2. **Blockwise engineering choice**
   - The default block size is `128`, which fits the common `512` / `1536` / `2048` widths used by this repo's attention and MLP matrices.
   - This is a practical compromise between full-matrix Hadamard transforms and more complex learned transforms from the literature.

3. **CPU-safe fallback attention path**
   - If `flash_attn_interface` is unavailable, the script falls back to `torch.nn.functional.scaled_dot_product_attention`.
   - This is mainly to make lightweight smoke testing easier and keep the candidate more self-contained.

4. **A dedicated `SMOKE_TEST=1` path**
   - Builds a tiny CPU model, runs a forward/backward pass, quantizes it, dequantizes it, reloads it, and checks that logits stay finite.

The training recipe itself is intentionally unchanged so the effect stays concentrated in the export path.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603250433_block-rotate-gptq
```

The script resolves its default dataset/tokenizer paths from the repository root, so running it from inside this candidate directory works without extra path flags as long as the standard repo `data/` tree is present.

Full training/eval command (same base stack, with the new export path enabled by default):

```bash
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
ROTATION_ENABLED=1 ROTATION_BLOCK_SIZE=128 ROTATION_TRIALS=4 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional ablations:

```bash
# disable the new idea entirely
ROTATION_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# smaller / larger search
ROTATION_TRIALS=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
ROTATION_TRIALS=8 torchrun --standalone --nproc_per_node=8 train_gpt.py

# different block size
ROTATION_BLOCK_SIZE=64 torchrun --standalone --nproc_per_node=8 train_gpt.py
ROTATION_BLOCK_SIZE=256 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Local smoke path:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation run for this candidate

### Completed successfully

- `python -m compileall candidates/202603250433_block-rotate-gptq/train_gpt.py`
- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603250433_block-rotate-gptq/train_gpt.py`

### Attempted but not feasible in this workflow environment

- `cd candidates/202603250433_block-rotate-gptq && SMOKE_TEST=1 python train_gpt.py`

That smoke command is implemented in the candidate, but this workflow runner does not have the full runtime stack installed. In particular, the Python environment used here is missing `torch`, so a real forward-pass smoke test could not be executed inside this job.

## Main expected risks and tradeoffs

- **The rotation search is still heuristic.** SpinQuant and OptRot suggest that rotation choice matters a lot; this candidate only does a tiny deterministic sign-search, not learned or data-dependent optimization.
- **Embeddings may still dominate.** The repo repeatedly finds that the tied embedding/output path is unusually sensitive. Rotation may help, but it may not be enough compared with fp16 passthrough or a more radical embedding-specific compression scheme.
- **Export time increases.** The new search is CPU-side and only affects serialization, but it does add extra quantization work for each large 2D tensor.
- **The best block size is unknown.** `128` is a practical default, not a proven optimum for this model family.
- **No score claim yet.** This candidate is meant to be the next strong experiment to run, not a verified leaderboard result.
