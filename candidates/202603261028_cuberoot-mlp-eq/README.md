# Candidate: Cubic-Root MLP Equalization

## Hypothesis

The current best stack is already strong in training and evaluation, so the next cheap gain may come from reducing **int6 roundtrip damage** rather than adding more runtime-heavy architecture. This candidate adds an **exact, export-time channel equalization pass** for the MLPs.

For the donor stack's MLP,

```python
phi(x) = leaky_relu(x, negative_slope=0.5) ** 2
```

so for any positive per-hidden-channel scale `s`, the following identity holds exactly:

```python
y = W_down @ phi(W_up x)
  = (W_down / s^2) @ phi((s * W_up) x)
```

That means we can rebalance `W_up` rows against `W_down` columns **without changing the full-precision function at all**, then quantize the rebalanced weights. The candidate uses a **cubic-root** balance rule,

```python
s_i = (down_stat_i / up_stat_i) ** (1/3)
```

because the up-path scales linearly while the down-path compensates quadratically.

## Why this is promising for this repository

The repo history suggests the biggest remaining headroom is in quantization-aware export:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved the best pre-TTT stack mostly through better post-training quantization and averaging.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` already stacked most of the obvious training-side wins, leaving export quality as a relatively attractive next lever.
- MLP weights are a large fraction of the int6 artifact, and this repo's squared MLP nonlinearity gives an unusually clean exact symmetry to exploit.

This candidate therefore tries to buy a quantization win with:

- no new runtime parameters,
- no change to the full-precision model function,
- no extra evaluation-time compute,
- only a small export-time preprocessing step.

## Records and prior candidates that influenced it

There was **no existing `candidates/` directory** when this candidate was created, so the influences were all from `records/`.

Primary donor records:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - source for the base architecture, LeakyReLU^2 MLP, parameter banking, legal TTT, and current strongest overall stack.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strong evidence that careful post-training quantization still matters materially on this leaderboard.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - evidence that zero-parameter changes with clean mathematical motivation can still move BPB.

## External research that informed it

This candidate is mainly inspired by the line of work on **equivalent transformations for post-training quantization**:

- **SmoothQuant** (Xiao et al., 2023, arXiv:2211.10438)
  - key takeaway: move quantization difficulty across adjacent operations using a mathematically equivalent offline transform.
- **AWQ: Activation-aware Weight Quantization** (Lin et al., 2024, arXiv:2306.00978)
  - key takeaway: channel scaling can protect salient weights without changing model semantics.
- **Data-Free Quantization Through Weight Equalization and Bias Correction** (Nagel et al., 2019, arXiv:1906.04721)
  - key takeaway: scale-equivariance can be exploited to equalize weight ranges before quantization, even without retraining.

The twist here is specific to this repo: because the MLP activation is **LeakyReLU squared**, the exact balancing rule becomes **cubic-root** rather than the more standard square-root-style equalization used for degree-1 activations.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added three environment knobs:
   - `MLP_EQUALIZE_ENABLED` (default `1`)
   - `MLP_EQUALIZE_QUANTILE` (default `0.999`)
   - `MLP_EQUALIZE_MAX_SCALE` (default `4.0`)
2. Added `apply_square_mlp_channel_equalization(...)`, which:
   - reads each layer's `blocks.{i}.mlp.fc.weight` and `blocks.{i}.mlp.proj.weight`,
   - computes hidden-channel stats from absolute-value quantiles,
   - applies cubic-root balancing with geometric-mean normalization and scale clipping,
   - writes back an exactly equivalent full-precision MLP pair.
3. Applied the transform **only on the export / quantization path** after unbanking and before `mixed_quantize_int6(...)`.
4. Left the training graph, TTT path, and full-precision model architecture unchanged.

If `MLP_EQUALIZE_ENABLED=0`, the candidate falls back to the donor stack's original export behavior.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603261028_cuberoot-mlp-eq

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MLP_EQUALIZE_ENABLED=1 MLP_EQUALIZE_QUANTILE=0.999 MLP_EQUALIZE_MAX_SCALE=4.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful A/B:

```bash
MLP_EQUALIZE_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script logs an `mlp_equalize ...` line before export-time quantization so the chosen settings are visible in the run log.

## Validation run for this candidate

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603261028_cuberoot-mlp-eq/train_gpt.py
```

Outcome:

- Passed.

Pure-Python numerical smoke check for the new algebra:

```bash
python - <<'PY'
import math
import random

random.seed(0)
B, T, D, H = 2, 3, 6, 8

def rand_tensor(*shape):
    if len(shape) == 1:
        return [random.gauss(0.0, 1.0) for _ in range(shape[0])]
    return [rand_tensor(*shape[1:]) for _ in range(shape[0])]

def linear_1d(vec, weight):
    return [sum(v * w for v, w in zip(vec, row, strict=True)) for row in weight]

def linear_3d(x, weight):
    return [[linear_1d(token, weight) for token in batch] for batch in x]

def leaky_relu_sq(x, negative_slope=0.5):
    return [[[(v if v >= 0.0 else negative_slope * v) ** 2 for v in token] for token in batch] for batch in x]

def quantile(vals, q):
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac

def max_abs_diff(a, b):
    best = 0.0
    for batch_a, batch_b in zip(a, b, strict=True):
        for tok_a, tok_b in zip(batch_a, batch_b, strict=True):
            for va, vb in zip(tok_a, tok_b, strict=True):
                best = max(best, abs(va - vb))
    return best

x = rand_tensor(B, T, D)
up = rand_tensor(H, D)
down = rand_tensor(D, H)
up_stat = [max(1e-6, quantile([abs(v) for v in row], 0.999)) for row in up]
down_stat = []
for col in range(H):
    down_stat.append(max(1e-6, quantile([abs(down[row][col]) for row in range(D)], 0.999)))
scale = [(d / u) ** (1.0 / 3.0) for u, d in zip(up_stat, down_stat, strict=True)]
geo = math.exp(sum(math.log(s) for s in scale) / len(scale))
scale = [min(4.0, max(0.25, s / geo)) for s in scale]
eq_up = [[v * scale_i for v in row] for scale_i, row in zip(scale, up, strict=True)]
eq_down = []
for row in down:
    eq_down.append([v / (scale[col] ** 2) for col, v in enumerate(row)])
y0 = linear_3d(leaky_relu_sq(linear_3d(x, up)), down)
y1 = linear_3d(leaky_relu_sq(linear_3d(x, eq_up)), eq_down)
print(max_abs_diff(y0, y1))
PY
```

Outcome:

- Passed with exact numerical agreement (`0.0` max absolute error in the pure-Python smoke test).

Why no full CPU startup smoke test was run:

- This workflow environment does **not** currently have `torch` importable.
- `flash_attn_interface` is not present.
- the cached challenge dataset directory (`data/datasets/fineweb10B_sp1024`) is not present.

So a real start-to-first-batch execution check was not feasible in this runner without adding heavyweight setup that is outside the repository's lightweight validation pattern.

## Main expected risks and tradeoffs

- The transform is mathematically exact in full precision, but it can still hurt if the chosen per-channel statistic (`0.999` abs quantile) is the wrong proxy for this repo's GPTQ-lite int6 error.
- Equalization helps MLP quantization, but the global score may still be dominated by attention or embedding export error.
- Clamping scales too tightly may leave gains on the table; clamping too loosely may help one matrix while hurting another.
- Because the latest record already uses legal TTT, any export-side gain may be partially masked or amplified by post-quant TTT adaptation.

If this candidate is neutral, the next experiments I would try are:

1. sweep `MLP_EQUALIZE_QUANTILE` between `0.995`, `0.999`, and `1.0`,
2. sweep `MLP_EQUALIZE_MAX_SCALE` between `2.0`, `4.0`, and `8.0`,
3. apply equalization only to the deepest MLPs,
4. combine this with the existing dormant training-only MTP path in the donor script.
