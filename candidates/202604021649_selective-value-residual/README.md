# Selective Value Residual on the XSA Tail

## Hypothesis

The current frontier stack gets strong gains from deep-layer Exclusive Self Attention (XSA), but XSA intentionally removes the component aligned with a token's own value vector. A **selective value-residual path** should restore a cheap identity-preserving channel without undoing XSA everywhere: capture a shallow value state once, then inject it only into the deepest XSA-heavy blocks.

The expected result is a small but real reduction in validation BPB with almost no training-speed cost and essentially no artifact-size cost.

## Why this is promising for this repository

The record history in `records/` shows a clear pattern:

- deeper 11-layer stacks beat the old 9-layer baseline once quantization/export became cheap enough,
- **XSA on the deepest layers** kept paying off,
- tiny control-parameter additions like LN scaling, EMA, and GPTQ-lite clipping produced meaningful late-stage gains,
- the latest March 23 frontier (`1.1194` post-TTT mean) is already optimized for throughput, so the best next idea is likely a **near-zero-cost architectural refinement** rather than a big new subsystem.

This candidate targets exactly that regime. It adds only a few learned scalars (`vr_lambda`) and reuses the already-present banked attention/value path, so it should preserve the latest training speed profile while testing a new inductive bias.

## Prior experiments that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - current frontier with parameter banking, Parallel Muon, LeakyReLU(0.5)^2, GPTQ-lite + lzma, and legal score-first TTT.
- **Direct architectural context:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - established the strong 11L XSA/Partial-RoPE/VE stack before TTT.
- **XSA introduction:** `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
  - showed deep-layer XSA works in this repo and is worth keeping.
- **No prior candidates existed** under `candidates/` when this candidate was created.

## External research that informed it

1. **Exclusive Self Attention** — Zhai et al., arXiv:2603.09078  
   <https://arxiv.org/abs/2603.09078>  
   XSA improves sequence modeling by explicitly removing self-value-aligned information from attention outputs, with larger gains at longer context.

2. **ResFormer / SVFormer (Value Residual Learning)** — Zhou et al., arXiv:2410.17897  
   <https://arxiv.org/abs/2410.17897>  
   The paper argues that hidden-state residuals alone do not preserve token-level information well in deep transformers, and that **value residual connections** improve information flow. Its SVFormer variant specifically shares the first layer's value embedding across deeper layers, which is especially relevant here.

These two ideas fit together naturally for this repo: XSA deliberately suppresses self-copying, while a selective value residual can reintroduce a controlled self-information path only where the model is most aggressively contextualized.

## What changed versus the chosen base implementation

Starting point: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Selective value residual enabled by default**
   - `VALUE_RESIDUAL` now defaults to `1`.
   - New flag: `VALUE_RESIDUAL_LAST_N` (default `4`).

2. **First-layer capture, deep-layer injection**
   - Block 0 captures the source value state.
   - The residual mix is applied only to the last `VALUE_RESIDUAL_LAST_N` layers (default: the deepest 4 blocks), matching the part of the stack where XSA is already active.

3. **Attention backend fallback**
   - Added a `scaled_dot_product_attention` fallback when `flash_attn_interface` is unavailable.
   - This keeps GPU behavior aligned with the base when FlashAttention is installed, but makes local import/smoke validation easier in lighter environments.

4. **Extra logging**
   - Logs the active value-residual layers and whether FlashAttention was available.

Everything else is intentionally left aligned with the March 23 frontier: parameter banks, Parallel Muon, LeakyReLU(0.5)^2, XSA tail, VE, GPTQ-lite + lzma export, and legal score-first TTT.

## How to run / evaluate

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
VALUE_RESIDUAL=1 VALUE_RESIDUAL_LAST_N=4 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a quick syntax-only check:

```bash
python -m compileall train_gpt.py
```

## Validation

Commands run while preparing this candidate:

```bash
python -m compileall candidates/202604021649_selective-value-residual/train_gpt.py

python - <<'PY'
import importlib.util
from pathlib import Path
import torch
path = Path('candidates/202604021649_selective-value-residual/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
...
PY
```

Outcomes:

- `python -m compileall .../train_gpt.py` **succeeded**.
- The minimal CPU import/forward smoke test was **not feasible in this workflow container** because the Python environment does not have the repository's ML dependencies installed (`torch`, `sentencepiece`, and even `numpy` were absent), so the attempt failed before the model code itself could run.

## Main risks / tradeoffs

- The residual path might reintroduce too much self-information and partially cancel XSA's benefit.
- The best depth for injection may be narrower than the last 4 layers; the right setting may turn out to be `2` or `3`.
- Because this change is so cheap, the upside is likely incremental rather than dramatic.
- The candidate is grounded in compatible prior work, but it has not yet been trained on the real leaderboard setup, so the true interaction with TTT and GPTQ-lite quantization remains uncertain.

## Suggested next experiments if this helps

1. Sweep `VALUE_RESIDUAL_LAST_N` over `2, 3, 4`.
2. Restrict the residual to only the XSA blocks instead of a simple deepest-`N` rule.
3. Combine this with a smaller selective-TTT recipe that adapts only low-dimensional control tensors.
