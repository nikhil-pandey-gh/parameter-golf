# Signed Hadamard GPTQ-lite

## Hypothesis

The strongest recent gains in this repository have come from fitting a better model into the same 16 MB artifact budget, especially through stronger post-training quantization. This candidate tests whether a lightweight **rotation-aware PTQ** pass can shrink the int6 reconstruction gap of the repository's strongest clean 11-layer stack by spreading weight outliers before GPTQ-lite clip search.

The concrete variant here is intentionally minimal: apply a deterministic **signed block-Hadamard rotation** to large 2D attention and MLP weights immediately before int6 quantization, then invert that rotation after dequantization when loading for evaluation. That keeps the runtime model architecture unchanged while borrowing the core "remove outliers with orthogonal rotations" idea from recent PTQ work.

## Why this is promising for this repository

Repository review showed a very clear trend:

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

all improved primarily by making the quantized artifact more efficient or less lossy, not by radically changing the trainer. The best non-TTT base already uses EMA, tight SWA, partial RoPE, LN scaling, XSA, BigramHash, SmearGate, and GPTQ-lite clip search. That makes the export path the highest-leverage place to try the next idea.

There were no pre-existing experiments under `candidates/` when this candidate was created.

## Prior records that influenced this candidate

- **Base implementation**: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest clean non-TTT stack,
  - already optimized around GPTQ-lite style per-row clip search,
  - simpler to modify safely than the newer score-first TTT / parallel-Muon submission.

- **Related architectural stack**:
  - `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`

These records established that the 11L / MLP3x / XSA4 / EMA / partial-RoPE family is the right base. This candidate deliberately leaves that stack alone and only changes PTQ behavior.

## External research that informed it

- **QuaRot** — [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
  - Shows that orthogonal rotations can remove quantization-hostile outliers while preserving model function.

- **SpinQuant** — [arXiv:2405.16406](https://arxiv.org/abs/2405.16406)
  - Demonstrates that some rotations are much more quantization-friendly than others and that rotation choice can materially improve PTQ quality.

- **KurTail** — [arXiv:2503.01483](https://arxiv.org/abs/2503.01483)
  - Reinforces the same core lesson in a lower-overhead PTQ setting: reshaping outlier structure before quantization is often higher leverage than another small clip-threshold tweak.

This candidate does **not** attempt full activation-aware, learned, or fused-in-graph rotation machinery. Instead, it implements the smallest version that fits this repository cleanly: deterministic offline rotations on the exported int6 tensors.

## What changed vs the chosen base

Starting from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. Adds deterministic PTQ controls:
   - `PTQ_ROTATION_ENABLED=1`
   - `PTQ_ROTATION_BLOCK_SIZE=512`
   - `PTQ_ROTATION_MIN_BLOCK_SIZE=128`
   - `PTQ_ROTATION_SIGN_SEED=1337`

2. Adds a **signed block-Hadamard transform** for large 2D tensors.

3. Applies that transform only to large int6-quantized attention / MLP matrices before GPTQ-lite percentile clip search.

4. Stores rotation metadata in the quantization manifest so the loader can invert the transform after dequantization.

5. Logs how many tensors were rotated during export.

Everything else intentionally stays aligned with the `2026-03-22` base: same model family, same optimizer stack, same EMA/SWA schedule, same evaluation path, same export structure.

## How to run

From this candidate directory:

```bash
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
PTQ_ROTATION_ENABLED=1 PTQ_ROTATION_BLOCK_SIZE=512 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to ablate the idea directly, rerun with `PTQ_ROTATION_ENABLED=0`.

## Expected risks and tradeoffs

- This is a **lightweight transform-coding variant**, not full QuaRot / SpinQuant. Gains may be smaller than the papers because we are only rotating exported weight tensors.
- The benefit is likely concentrated in the already-quantized MLP / attention weights. If the remaining gap is dominated by something else, the effect may be small.
- Fixed 512-wide blocks are a good fit for this repo's dominant hidden sizes, but may not be the best choice for every matrix.
- The added code path only changes export / reload behavior, so training-time metrics may stay flat even if quantized eval improves.

## Validation

### Commands run

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603260644_hadamard-gptq-lite/train_gpt.py
```

### Outcomes

- Syntax compilation passed for:
  - `train_gpt.py`
  - `train_gpt_mlx.py`
  - `data/`
  - `candidates/202603260644_hadamard-gptq-lite/train_gpt.py`

- A true CPU startup smoke test was **not feasible** on this runner:
  - the environment does not have the repo runtime dependencies installed (notably `torch` / `numpy` imports were unavailable for execution), and
  - the record-style training script also hard-requires CUDA + FlashAttention at runtime.

So the candidate is validated here at the syntax / static level, with the expectation that functional evaluation happens in the normal GPU-equipped training environment used for repository submissions.
