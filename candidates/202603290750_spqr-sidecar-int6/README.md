# 202603290750_spqr-sidecar-int6

## Hypothesis

The current best reusable stacks in this repository already look strong on architecture and training, but they still pay a meaningful accuracy tax at export time because almost all large MLP and attention weights are forced through dense per-row int6 quantization. A tiny SpQR-style residual sidecar should let the model preserve only the few worst per-row reconstruction errors while leaving the bulk of each matrix in the existing GPTQ-lite int6 format, improving post-quantization quality without blowing the 16MB artifact budget.

## Why this is promising for this repository

- The leaderboard has largely converged on 10-11 layer models with XSA, partial RoPE, LN scaling, EMA/SWA, and aggressive low-bit export, so export quality is one of the cleanest remaining levers.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` already showed that better per-row clip selection is worth a real gain at essentially zero training cost.
- The repository review found no existing record or prior candidate using sparse outlier sidecars, SpQR-style residual storage, AWQ-style protected channels, or rotation-based outlier handling.

## Which records or prior candidates influenced it

- Base implementation: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- Architecture context: `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- Frontier context: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

There were no existing `candidates/` folders in the repository root when this candidate was created.

## External research that informed it

- **SpQR** — Egiazarian et al., 2023, arXiv:2306.03078. The motivating idea is to isolate a tiny set of outlier weights that dominate low-bit degradation and store them in higher precision.
- **GPTQ** — Frantar et al., 2023, arXiv:2210.17323. Current repo practice is already GPTQ-lite-adjacent, so this candidate keeps the same weight-only PTQ framing.
- **AWQ** — Tang et al., 2024, arXiv:2306.00978. AWQ reinforces the same underlying thesis: protecting a very small fraction of salient weights/channels can disproportionately reduce quantization error.
- **QuaRot** — Croci et al., 2024, arXiv:2404.00456. Recent quantization work continues to point at outliers as a major source of degradation, even when the mitigation strategy differs.

## What changed versus the chosen base implementation

This candidate starts from the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` training script and keeps its training/model stack intact:

- 11 layers, 512d, 8 heads / 4 KV heads
- 3x MLP with relu-squared
- BigramHash, SmearGate, shared value embeddings
- partial RoPE, LN scaling, XSA on the last 4 layers
- EMA export path and GPTQ-lite per-row percentile search

The changes are intentionally surgical:

1. **Sparse int6 residual sidecar**
   - Added `INT6_OUTLIER_TOPK` (default `2`).
   - After GPTQ-lite per-row int6 quantization, each eligible 2D MLP/attention matrix stores the top-`k` largest absolute residual entries per row in a tiny sidecar (`int16` column indices + `float16` residual values).
   - Dequantization reconstructs the dense int6 tensor and then adds back the sparse residual sidecar.

2. **CPU-safe attention fallback for smoke testing**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch `scaled_dot_product_attention`.
   - This is meant for lightweight import/forward validation only; the intended leaderboard path still uses FlashAttention 3 on CUDA.

3. **Portable repo-root discovery for default paths**
   - The script now walks upward until it finds the repository-level `data/` directory instead of assuming a fixed parent depth.
   - That keeps the default `DATA_PATH` / `TOKENIZER_PATH` working both from this `candidates/` location and from a future `records/.../` location if the candidate graduates.
   - Explicit `DATA_PATH` / `TOKENIZER_PATH` environment overrides bypass repo-root discovery, so the module can still be imported from a relocated copy when those paths are supplied manually.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603290750_spqr-sidecar-int6

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
INT6_OUTLIER_TOPK=2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script discovers its repo root by walking upward until it finds the repository `data/` directory, so the default `DATA_PATH` and `TOKENIZER_PATH` continue to work when launched directly from inside `candidates/202603290750_spqr-sidecar-int6/` and remain portable to a future `records/.../` location. If you copy the script elsewhere, set `DATA_PATH` and `TOKENIZER_PATH` explicitly to bypass repo-root discovery.

For a quick local structural smoke test, import the module and run a tiny CPU forward pass through `GPT`; that path uses the SDPA fallback when FlashAttention 3 is unavailable.

## Validation

Commands run so far:

```bash
python -m compileall candidates/202603290750_spqr-sidecar-int6/train_gpt.py

# in an isolated temp venv because the shared system Python did not include torch
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('candidates/202603290750_spqr-sidecar-int6/train_gpt.py')
spec = importlib.util.spec_from_file_location('cand_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model = mod.GPT(
    vocab_size=32,
    num_layers=2,
    model_dim=32,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2.0,
    tie_embeddings=True,
    tied_embed_init_std=0.02,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.0,
    mtp_num_heads=0,
    mtp_loss_weight=0.0,
    bigram_vocab_size=0,
    bigram_dim=16,
    xsa_last_n=1,
    rope_dims=8,
    ln_scale=True,
    dtg=False,
    ve_enabled=False,
    ve_dim=8,
    ve_layers='',
)
input_ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
target_ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
loss = model(input_ids, target_ids)
print(float(loss.detach()))
PY

python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('candidates/202603290750_spqr-sidecar-int6/train_gpt.py')
spec = importlib.util.spec_from_file_location('cand_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

weight = torch.randn(512, 256) * 0.7
state = {'blocks.0.mlp.fc.weight': weight.clone()}
plain_w, plain_m = mod.mixed_quantize_int6(state, {'mlp', 'attn'}, outlier_topk=0)
plain = mod.dequantize_mixed_int6(plain_w, plain_m, state)['blocks.0.mlp.fc.weight']
side_w, side_m = mod.mixed_quantize_int6(state, {'mlp', 'attn'}, outlier_topk=2)
side = mod.dequantize_mixed_int6(side_w, side_m, state)['blocks.0.mlp.fc.weight']
print(float((weight - plain).abs().mean()))
print(float((weight - side).abs().mean()))
print('blocks.0.mlp.fc.weight.side_idx' in side_w)
PY
```

Outcome:

- `compileall` succeeded.
- In an isolated temporary virtual environment, a CPU import-and-forward smoke test succeeded with a tiny 2-layer / 32-dim `GPT` instantiation and produced a finite loss (`3.4799` in the latest validation rerun after the lazy-path fix).
- A toy quantization round-trip on a `512 x 256` synthetic MLP weight matrix confirmed that the sidecar path is populated and improves reconstruction error versus dense int6 alone (`mean abs error 0.01684 -> 0.01656`, sidecar present = `True`).
- A relocated-copy import smoke test also passed once `DATA_PATH` and `TOKENIZER_PATH` were provided explicitly, confirming that manual overrides bypass repo-root discovery instead of failing at import time.

## Main expected risks or tradeoffs

- The sidecar may recover quantization error but still cost too many bytes after serialization and compression.
- A fixed top-`k` per row is simple and robust, but may not be the best allocation strategy across attention vs. MLP or early vs. late layers.
- Because this candidate focuses on export quality rather than changing the training stack, its absolute BPB gain may be smaller than more aggressive ideas like legal TTT or larger architectural rewrites.
