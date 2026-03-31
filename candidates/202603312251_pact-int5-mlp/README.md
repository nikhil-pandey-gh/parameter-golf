# PACT-style mixed-precision QAT on the 11L EMA/XSA/VE stack

## Hypothesis

The repository's strongest trend is that **better compression buys better models**: once post-training export quality improves, the freed bytes get reinvested into deeper or wider networks. Earlier records showed that **int5 MLP weights** can buy meaningful artifact headroom, but that result came from an older stack. This candidate tests whether a more modern base can make int5 MLP viable again by replacing fixed export heuristics with **compile-safe, learned clipping for weight QAT**.

Concretely, the candidate keeps the mature 11-layer EMA/XSA/partial-RoPE/value-embedding stack from `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, but changes the quantization story:

- `MLP` weights train/export toward **int5**.
- attention weights train/export toward **int6**.
- each large linear gets a learned scalar clip multiplier, so clipping is tuned during training instead of chosen only by a fixed export-time heuristic.
- QAT uses module-owned tensor state instead of the old class-attribute toggle, avoiding the `torch.compile` constant-folding failure that previously made late QAT a no-op.

## Why this is promising for this repository

Three repository patterns point toward this candidate:

1. `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md` showed that **int5 MLPs** are compressible enough to fund meaningful capacity under the 16MB cap.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` showed that **clipping policy matters even on a strong late-generation stack**.
3. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` documented that the earlier late-QAT path could silently do nothing under `torch.compile`.

This candidate aims to combine those lessons: keep the strong modern architecture, keep the strong optimizer/export pipeline, but make the low-bit pressure on the MLP explicit and learned during training.

## Prior records and candidates that influenced this run

There were no existing `candidates/` directories in the repo when this was created.

The main record influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the implementation base because it is a clean pre-TTT strong stack with EMA, XSA4, partial RoPE, VE, and mature mixed-precision export.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - motivates revisiting **int5 MLP** on a newer architecture.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - motivates making QAT compile-safe instead of relying on a global boolean gate.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - motivates the switch from `relu^2` to **LeakyReLU(0.5)^2** in the MLP.

## External research that informed it

The quantization idea is primarily inspired by:

- **PACT: Parameterized Clipping Activation for Quantized Neural Networks** (2018, arXiv:1805.06085)
- **Learned Step Size Quantization** (LSQ, 2019, arXiv:1902.08153)
- **Scaling Law for Quantization-Aware Training** (2025, arXiv:2505.14302)

This candidate does not implement full LSQ exactly. Instead, it takes the part that best fits this repository's constraints: a **small learned clipping parameter per large weight matrix**, wired directly into the existing PyTorch training/export flow with minimal extra infrastructure.

## What changed versus the chosen base implementation

Relative to `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Compile-safe learned clipping for QAT**
   - `CastedLinear` now owns:
     - a learned `qat_clip_logit`,
     - a `qat_mix` buffer, and
     - a per-module bit-width.
   - The training path uses STE-style fake quantization without relying on a class-level boolean that can be constant-folded away.

2. **Mixed precision defaults**
   - `ATTN_QUANT_BITS=6`
   - `MLP_QUANT_BITS=5`
   - export is now driven by the learned clip multipliers gathered from the trained modules.

3. **LeakyReLU(0.5)^2 MLP**
   - carries forward the strongest recent activation change while keeping the rest of the MLP structure unchanged.

4. **FlashAttention fallback**
    - if `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA.
    - non-flash SDP backends stay enabled in that case, so the fallback is usable instead of only syntactic.
    - this is mostly for local validation/import robustness; the intended leaderboard path is still the CUDA/FlashAttention route.

## How to run or evaluate it

From the repository root:

```bash
RUN_ID=pact_int5_mlp \
QAT_ENABLED=1 \
ATTN_QUANT_BITS=6 \
MLP_QUANT_BITS=5 \
MLP_LEAKY_SLOPE=0.5 \
QAT_CLIP_INIT=0.98 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
WARMDOWN_ITERS=3500 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 \
  candidates/202603312251_pact-int5-mlp/train_gpt.py
```

If the full-time QAT overhead is too expensive, the candidate still exposes a late-QAT path:

```bash
QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.15
```

That mode is now compile-safe because enabling/disabling QAT changes module buffers instead of a class attribute, and the late-QAT switch is synchronized across DDP ranks before activation.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202603312251_pact-int5-mlp/train_gpt.py
```

Outcome:

- **Passed**.

Attempted smoke validation:

```bash
python - <<'PY'
# import the module and run a tiny CPU forward pass
PY
```

Outcome:

- **Not feasible in this runner** because the available Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
- The script was still adjusted to import cleanly when FlashAttention is missing, so a CPU import/forward smoke test should work in any normal PyTorch environment.

## Main expected risks and tradeoffs

- **QAT overhead**: always-on fake quantization may reduce steps completed in 600 seconds.
- **Int5 may still be too aggressive** for the MLPs even with learned clipping.
- **Per-matrix learned clipping is intentionally simple**; it is cheaper than full LSQ/per-channel schemes, but also less expressive.
- **Exporter/training mismatch is reduced, not eliminated**: the learned clip multipliers align train and export more closely, but export is still rowwise low-bit packing while training uses a lightweight fake-quant approximation.
