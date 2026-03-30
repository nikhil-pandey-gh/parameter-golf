# Quadratic MLP Equalization on the LeakyReLU^2 + Legal TTT stack

## Hypothesis

The current strongest stacks in this repo already squeeze a lot out of training-time compute and evaluation-time tricks, but they still pay a non-trivial gap when exporting to mixed `int6` weights. This candidate adds a **lossless export-time MLP equalization pass** that is specific to the repo's `relu^2` / `leaky_relu^2` MLPs.

For each MLP pair,

- `h = up @ x`
- `y = down @ phi(h)`
- `phi(z) = relu(z)^2` or `leaky_relu(z, 0.5)^2`

we can rescale hidden channels with any positive vector `s` and preserve the exact float function:

- `up' = diag(s) @ up`
- `down' = down @ diag(1 / s^2)`

because `phi(s * z) = s^2 * phi(z)` for `s > 0`.

This lets us rebalance per-channel ranges **before GPTQ-lite int6 quantization** without paying extra training cost or changing the float model.

## Why this is promising for this repository

The recent records cluster around the same frontier:

- 11 layers / 512 width with compact front-end tricks like `BigramHash` and `SmearGate`
- strong evaluation (`sliding window`, then `legal TTT`)
- export-aware training or post-training compression (`int6`, late QAT, GPTQ-lite, EMA)

The strongest non-TTT training/export stack is the 2026-03-22 GPTQ-lite + EMA record, and the current top overall result is the 2026-03-23 LeakyReLU^2 + Legal TTT + Parallel Muon record. Both still rely on aggressive mixed quantization at export time, so a **better weight-shaping step directly on the export path** is one of the lowest-risk remaining levers.

Unlike layer recurrence or heavier training-time QAT, this candidate leaves wall-clock training throughput unchanged.

## Prior records that influenced this candidate

The main local influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current stack; provides the base implementation, `leaky_relu(0.5)^2`, legal score-first TTT, and the banked export path
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - shows that post-training GPTQ-lite improvements can still move the needle on top of a strong 11-layer architecture
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - reinforces that small export-aware quality improvements are still worth chasing late in the stack
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful negative result: recurrent/depth-reuse style ideas can lose badly under fixed wall-clock budgets

## External research that informed it

This candidate adapts ideas from quantization literature into this repo's very specific MLP nonlinearity:

- **Cross-Layer Equalization for data-free quantization**: Nagel et al., 2019
  - https://arxiv.org/abs/1906.04721
- **SmoothQuant**: equivalent offline transformations to shift quantization difficulty while preserving the function
  - Xiao et al., 2022 / ICML 2023
  - https://arxiv.org/abs/2211.10438
- **AWQ**: activation-aware channel protection via equivalent scaling transforms
  - Lin et al., 2023 / MLSys 2024
  - https://arxiv.org/abs/2306.00978

The twist here is that the repo's `relu^2` / `leaky_relu^2` MLPs are **degree-2 homogeneous**, so the balancing rule becomes cubic rather than the more familiar square-root rule from ReLU-style equalization.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

- added `QCLE_ENABLED`, `QCLE_MAX_SCALE`, `QCLE_BLEND`, and `QCLE_EPS` export controls
- added `quadratic_equalize_mlp_pairs(...)`, which:
  - unbanks the MLP weights
  - computes per-hidden-channel scales
  - applies the exact `up * s`, `down / s^2` transformation layer-by-layer
  - logs aggregate scale statistics during export
- kept the existing GPTQ-lite `int6` path intact after equalization
- added a FlashAttention fallback to PyTorch SDPA when FlashAttention 3 is unavailable
- added `SMOKE_TEST_ONLY=1` support so the script can instantiate the model, verify exact float equivalence of the Q-CLE transform, and run an int6 roundtrip on CPU without dataset/tokenizer setup

## How the scaling is chosen

For each hidden channel, let:

- `u` = max absolute weight in the corresponding `up` row
- `d` = max absolute weight in the corresponding `down` column

Then this candidate uses:

- `s = (d / u)^(blend / 3)`

followed by geometric recentering and clamping to `[1 / QCLE_MAX_SCALE, QCLE_MAX_SCALE]`.

The `1/3` exponent is the degree-2 analogue of classic cross-layer equalization: it balances the growth of the `up` row (`* s`) against the shrinkage of the `down` column (`/ s^2`).

## How to run

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
QCLE_ENABLED=1 QCLE_MAX_SCALE=4.0 QCLE_BLEND=1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Quick local smoke check:

```bash
SMOKE_TEST_ONLY=1 SMOKE_SEQ_LEN=8 SMOKE_BATCH_SIZE=1 python train_gpt.py
```

## Validation

Commands run during this workflow:

```bash
python -m compileall candidates/202603301749_quadratic-cle/train_gpt.py
SMOKE_TEST_ONLY=1 SMOKE_SEQ_LEN=8 SMOKE_BATCH_SIZE=1 python candidates/202603301749_quadratic-cle/train_gpt.py
```

Outcomes:

- `compileall`: passed
- CPU smoke path: passed in a temporary venv after installing `numpy`, `sentencepiece`, and `torch`
- smoke output:
  - `smoke:device:cpu`
  - `smoke:base_loss:6.944032`
  - `smoke:qcle_loss:6.944032`
  - `smoke:qcle_exact_diff:0.000000e+00`
  - `smoke:int6_loss:6.969493`
  - `smoke:qcle_layers:11 scale_mean:1.0006 scale_min:0.8493 scale_max:1.1084`

## Main expected risks / tradeoffs

- The transformation is **exact in float**, but it can still make some channels worse for the downstream quantizer if the cubic balance heuristic is poorly tuned.
- The repo's quantizer is per-row, while the MLP down-projection effect is column-oriented; the balancing heuristic is principled but still approximate relative to the real quantization objective.
- If `QCLE_MAX_SCALE` is too loose, a few channels may become over-amplified and hurt compression or numerical stability.
- If the current frontier is already dominated more by evaluation than by export quality, the gain may be smaller than hoped.
