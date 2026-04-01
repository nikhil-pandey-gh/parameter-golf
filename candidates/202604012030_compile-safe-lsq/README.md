# Compile-Safe LSQ Late QAT

## Hypothesis

The strongest non-TTT stack in this repo already gets most of its gains from better export quality, but its "late QAT" path still uses a fixed row-max STE toggle that was previously shown to be vulnerable to `torch.compile` constant-folding. This candidate replaces that brittle boolean gate with a compile-safe tensor ramp and swaps the fixed fake-quant scale for learnable per-row LSQ-style scales on block linears.

The bet is that a smooth, learnable late-QAT path will reduce the int6 roundtrip gap without spending extra artifact bytes, which is a better fit for this challenge than adding more architecture complexity.

## Why this is promising here

- Recent records repeatedly show that **quantization/export quality** is still a major bottleneck, even after longer training and stronger architectures.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly notes that its late-QAT flag was a **no-op under `torch.compile`**, so there is still a clean unresolved training lever in this code path.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` is already a strong non-TTT base with EMA, GPTQ-lite clip search, partial RoPE, XSA, VE, and artifact headroom; that makes it the best place to test a tighter quantization-aware training loop.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - Chosen base implementation.
   - Kept its 11L/XSA4/partial-RoPE/LN-scale/EMA/GPTQ-lite stack unchanged outside the QAT path.
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
   - Important negative result: the README documents that the previous late-QAT switch was effectively dead under `torch.compile`.
3. `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - Reinforced that quantization-sensitive parameters and post-training roundtrip loss deserve direct attention.

## External research

- **LSQ** — Esser et al., *Learned Step Size Quantization* (ICLR 2020): https://arxiv.org/abs/1902.08153
- **LSQ+** — Bhalgat et al., *LSQ+: Improving low-bit quantization through learnable offsets and better initialization*: https://arxiv.org/abs/2004.09576
- **PACT** — Choi et al., *PACT: Parameterized Clipping Activation for Quantized Neural Networks*: https://arxiv.org/abs/1805.06085

These papers all point in the same direction: fixed clipping/rounding heuristics leave accuracy on the table, while learned quantizer parameters and smoother transitions into quantized training can recover more low-bit quality.

## What changed vs the base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. `CastedLinear` now has a **learnable per-row `qat_scale` parameter** and an LSQ-style straight-through path instead of a fixed row-max fake quantizer.
2. Late QAT is now controlled by a **compile-safe `qat_mix` tensor buffer** that linearly ramps from float weights to quantized weights between `LATE_QAT_START` and `LATE_QAT_END`.
3. `qat_scale` parameters use a dedicated **low-LR AdamW optimizer** (`QAT_SCALE_LR`) instead of piggybacking on the main scalar optimizer.
4. When late QAT actually activates, export reuses the learned LSQ row scales for the large block weights that are actually quantized; otherwise it cleanly falls back to the base GPTQ-lite percentile search.
5. `qat_scale` itself is still treated as a **training-only parameter** and excluded from export so the final artifact budget stays focused on deploy-time weights.

Everything else in the stack stays intentionally close to the 03-22 base: model shape, EMA, XSA, VE, partial RoPE, LN scaling, and evaluation flow. The GPTQ-lite percentile search remains the fallback int6 path for weights that do not use an LSQ scale override.

## How to run

From this candidate directory:

```bash
cd candidates/202604012030_compile-safe-lsq
RUN_ID=compile_safe_lsq \
QAT_ENABLED=1 \
LATE_QAT_START=0.30 \
LATE_QAT_END=0.05 \
QAT_SCALE_LR=0.001 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults in `train_gpt.py` already point at this LSQ-style late-QAT setup, so the extra env vars above are mainly there to make the intended experiment explicit.

## Validation

### Commands run

```bash
python -m compileall candidates/202604012030_compile-safe-lsq/train_gpt.py
python - <<'PY'
import importlib.util
import sys
import types
import torch

stub = types.ModuleType("flash_attn_interface")
stub.flash_attn_func = lambda q, k, v, causal=True: torch.zeros_like(q)
sys.modules["flash_attn_interface"] = stub

path = "candidates/202604012030_compile-safe-lsq/train_gpt.py"
spec = importlib.util.spec_from_file_location("candidate_train_gpt", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

### Outcomes

- `python -m compileall ...` **succeeded**.
- The import-based CPU smoke check was **not feasible in this workflow environment** because the available Python interpreter does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).

## Expected risks and tradeoffs

- LSQ scale learning may be **sensitive to `QAT_SCALE_LR`** and the late-QAT ramp window.
- The candidate now uses learned LSQ scales for the main quantized block weights, but smaller fp16-passthrough tensors still follow the base export rules, so some quantization asymmetry remains by design.
- The added LSQ path increases training-time compute slightly, so it may need tuning if step throughput regresses too much.
- This version only learns quantizer scales for **block linear weights**, not the tied embedding/head; LSQ+ on the embedding is a natural follow-up if this direction looks promising.
