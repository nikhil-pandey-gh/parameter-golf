# Candidate: Hadamard GPTQ-lite + LeakyReLU^2

## Hypothesis

The strongest low-risk next step is to start from the strong `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack and improve the **export bottleneck**, not the main training loop. The top records already sit very close to the 16MB limit, and recent record progress shows that cheap post-training/export improvements can still matter. This candidate therefore adds **blockwise Hadamard preconditioning inside GPTQ-lite int6 export** so each large matrix can choose whether a simple orthogonal rotation reduces reconstruction MSE before packing. On top of that, it also adopts the later repo-proven **LeakyReLU(0.5)^2** MLP activation.

## Why this is promising for this repository

Repository evidence points to two facts:

1. The current best non-TTT stacks are heavily limited by the quality of the compressed/exported model, not just the trained bf16 checkpoint.
2. The best recent wins often came from **small, composable, zero-or-low-cost tweaks** rather than broad architectural rewrites.

This candidate follows that pattern:

- it keeps the proven 11-layer / XSA / partial-RoPE / EMA / GPTQ-lite stack from the March 22 record,
- adds a **zero-training-cost** quantization refinement inspired by rotation-based PTQ work,
- and folds in **LeakyReLU^2**, which the March 23 record showed was one of the highest-signal cheap training changes.

I avoided more aggressive recurrence / looped-depth ideas because the repo’s own evidence is negative there under a fixed wall-clock budget.

## Prior records and experiments that influenced this candidate

Primary base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Most relevant repository influences:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - motivated swapping the MLP activation to `LeakyReLU(0.5)^2`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - reinforced partial RoPE + LN scaling as part of the strong reusable stack
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - showed the 11L/XSA/EMA/int6 line was the right branch to build on
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - documented a negative result for layer recurrence under fixed time, which pushed this candidate away from depth reuse and toward export-side improvements

There was **no existing `candidates/` directory** in the repository when this candidate was created, so there were no prior candidate iterations to avoid duplicating.

## External research that informed it

Primary sources:

- **QuaRot** — Croci et al., arXiv:2404.00456
  - orthogonal rotations can remove outliers and make quantization easier without changing the represented function
- **SpinQuant** — Liu et al., arXiv:2405.16406
  - rotations are not all equal; better rotations improve PTQ quality substantially
- **SignRound / AutoRound** — Cheng et al., arXiv:2309.05516
  - low-cost post-training optimization of clipping / rounding can close a real fraction of the quantization gap

This candidate intentionally implements the **lightest-weight version** of that direction that fits this repo well: deterministic blockwise Hadamard preconditioning at export time, searched against the no-rotation baseline by reconstruction MSE. It does **not** introduce learned rotations, calibration passes, or a second training stage.

## What changed versus the chosen base implementation

Compared to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes four focused changes:

1. **LeakyReLU(0.5)^2 MLP**
   - replaces `relu(x)^2` with `leaky_relu(x, 0.5)^2`
   - configurable via `LEAKY_RELU_SLOPE`

2. **Hadamard-preconditioned GPTQ-lite int6 export**
   - for eligible large rank-2 tensors in the int6 path, the exporter tries:
     - no rotation,
     - blockwise Hadamard with group size 128,
     - blockwise Hadamard with group size 256,
   - then keeps the option with the lowest reconstruction MSE
   - configurable via `INT6_HADAMARD_GROUP_SIZES`

3. **Dequantization aware of the chosen rotation**
   - the export metadata records whether a matrix used Hadamard preconditioning
   - dequantization applies the inverse transform (same normalized Hadamard, since it is involutory)

4. **Local validation ergonomics**
   - optional FlashAttention fallback to `scaled_dot_product_attention`
   - `SMOKE_TEST=1` path for CPU-only local sanity checks when `torch` is available
   - lazy/optional imports for dependencies not needed by the smoke path

## How to run or evaluate it

### Training / full evaluation

The script resolves its default dataset/tokenizer paths relative to the repository root, so it can be launched either from the repo root or from this candidate directory.

From the repo root:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
LEAKY_RELU_SLOPE=0.5 INT6_HADAMARD_GROUP_SIZES=0,128,256 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 candidates/202603312226_hadamard-gptq-lite/train_gpt.py
```

From the candidate directory:

```bash
cd candidates/202603312226_hadamard-gptq-lite
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
LEAKY_RELU_SLOPE=0.5 INT6_HADAMARD_GROUP_SIZES=0,128,256 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### CPU-only smoke check

If a local environment already has `torch` installed:

```bash
SMOKE_TEST=1 python candidates/202603312226_hadamard-gptq-lite/train_gpt.py
```

This exercises:

- model construction,
- the fallback attention path,
- Hadamard-aware int6 export,
- dequantization,
- and a post-roundtrip logits forward pass.

## Validation run for this workflow

Commands executed in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603312226_hadamard-gptq-lite/train_gpt.py
SMOKE_TEST=1 python candidates/202603312226_hadamard-gptq-lite/train_gpt.py
```

Outcome:

- `python -m compileall ...` **passed**.
- The CPU smoke test could **not** be completed on this runner because the environment does not have `torch` installed.
- I also attempted to install a temporary CPU-only `torch` into `/tmp/gh-aw/agent/torchdeps`, but that was blocked by the workflow network proxy (`403 Forbidden`), so I could not complete a runtime smoke pass in this environment.

## Main expected risks and tradeoffs

1. **Rotation may help some matrices and hurt others.**
   - This is why the exporter compares no-rotation versus multiple Hadamard group sizes and keeps the best MSE option per matrix.

2. **Reconstruction MSE is only a proxy.**
   - A matrix that looks better by MSE may still not yield the best final `val_bpb` after full roundtrip evaluation.

3. **Compression size could move in either direction.**
   - Better quantization error does not guarantee a smaller compressed artifact; rotated codes may compress slightly better or slightly worse.

4. **LeakyReLU^2 may interact with this stack differently than it did with the March 23 stack.**
   - It was strong in repo evidence, but it has not yet been re-ablated specifically on the March 22 GPTQ-lite base.

5. **The CPU smoke path is for sanity only.**
   - Real leaderboard-relevant performance still depends on the CUDA / FlashAttention training and evaluation path.
