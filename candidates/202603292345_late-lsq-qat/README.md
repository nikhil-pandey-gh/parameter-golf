# 202603292345_late-lsq-qat

## Hypothesis

The strongest training-only opportunity left in this repository is to make late-stage quantization-aware training actually work on the already-strong 11-layer GPTQ-lite stack. The March 21 record explicitly noted that its `Late QAT` branch never activated because `torch.compile` constant-folded the class-level flag away, so this candidate fixes that failure mode and upgrades the QAT path from fixed row-max clipping to a small LSQ-lite variant with learned per-layer clip multipliers.

## Why this is promising for this repository

The best non-TTT result in the repo is the March 22 11-layer EMA + GPTQ-lite stack, and its README shows the project is already close enough to the compression frontier that small quantization improvements matter. External research also points in the same direction: LSQ argues that learned quantization scales can recover low-bit accuracy, while OmniQuant shows that lightweight learnable quantization parameters can close post-training gaps without redesigning the model.

That makes late-stage QAT a good fit here:

- it targets the exact artifact that gets scored,
- it keeps the architecture and export format familiar,
- it stays inside the repository's existing `CastedLinear` / mixed-int6 export structure,
- and it avoids growing the final artifact in any meaningful way.

## Repository evidence that motivated this candidate

The main local influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest training-only stack in the repo,
  - already combines 11 layers, XSA, Partial RoPE, LN scaling, VE, GPTQ-lite, EMA, and long warmdown.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - explicitly documents that late QAT was dead-code-eliminated by `torch.compile` and therefore contributed nothing.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - shows the repo's current best overall score now depends on small improvements layered onto an already optimized stack, so fixing a training-only quantization path is still worthwhile.

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

Primary sources consulted:

- Esser et al., **"Learned Step Size Quantization"**, arXiv:1902.08153
- Shao et al., **"OmniQuant"**, arXiv:2308.13137
- Croci et al., **"QuaRot"**, arXiv:2404.00456
- Liu et al., **"SpinQuant"**, arXiv:2405.16406

The candidate is closest to LSQ/OmniQuant in spirit: instead of introducing rotations or new export formats, it adds a tiny learnable clipping control to each quantized linear layer and only turns that path on late in training.

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Fixed late-QAT activation semantics**
   - replaced the fragile class-level late-QAT switch with per-module enable flags,
   - when the LR scale falls below `LATE_QAT_THRESHOLD`, the script re-enables QAT on all `CastedLinear` modules and recompiles the training graph so the QAT branch cannot stay dead.

2. **Added LSQ-lite learnable clip multipliers**
   - every `CastedLinear` now owns a scalar `qat_clip`,
   - the clip multiplier is optimized during late QAT,
   - it rescales each layer's detached per-row maxima before STE quantization.

3. **Moved QAT controls into the repository's existing control-tensor path**
   - `qat_clip` is treated like other low-dimensional control parameters for fp32/fp16 handling and export.

4. **Raised the late-QAT default slightly**
   - `LATE_QAT_THRESHOLD` default changed from `0.15` to `0.18` to give the learned clip controls a little more warmdown runway.

## How the QAT path works

During normal training, the model follows the same strong March 22 stack.

Once the wallclock-aware LR multiplier drops below `LATE_QAT_THRESHOLD`, each `CastedLinear` starts using an STE low-bit proxy inside the forward pass:

- detached per-row maxima set the base scale,
- a learned scalar `qat_clip` rescales those maxima,
- weights are clipped to `[-31, 31]` in scale units,
- the rounded branch uses a straight-through estimator,
- and the whole graph is recompiled when QAT turns on so the branch is actually live.

This keeps the change narrow and directly aligned with the repository's int6 export path.

## How to run

Example candidate command:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_WD=0.04 ADAM_WD=0.04 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.18 \
QAT_CLIP_INIT=0.92 QAT_CLIP_MIN=0.50 QAT_CLIP_MAX=1.00 \
EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- `QAT_ENABLED=0` keeps training in the strong base regime until late warmdown.
- `LATE_QAT_THRESHOLD=0.18` activates QAT when the wallclock-aware LR scale gets small enough.
- `QAT_CLIP_*` controls the LSQ-lite clip multiplier range.

## Validation

Commands run for this candidate in this workflow:

```bash
python -m compileall candidates/202603292345_late-lsq-qat/train_gpt.py
```

Outcome:

- `compileall` **passed**.

Attempted additional smoke validation:

- I attempted a lightweight CPU-side import/forward smoke test.
- That was **not feasible in this workflow runner** because the available Python environment does not include `torch`, while this script also expects the repository's CUDA/FlashAttention training stack.
- I therefore limited validation here to syntax compilation and code inspection.

## Main expected risks and tradeoffs

- Recompiling the training graph when QAT turns on may cost a small amount of warmdown time.
- Learned clip multipliers may saturate at the clamp bounds if the chosen range is too tight.
- If the quantization gap is already dominated by effects outside `CastedLinear` modules, this candidate may underperform a more invasive export-side method such as rotations.
- Because this candidate targets training-only quality rather than TTT, it may improve the March 22-style stack more than the repo's March 23 end-to-end leaderboard score.
