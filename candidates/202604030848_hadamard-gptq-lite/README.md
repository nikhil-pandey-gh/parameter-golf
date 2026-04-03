# Hadamard-Rotated GPTQ-lite

## Hypothesis

The current 11-layer EMA + GPTQ-lite stack is already strong on training quality, so the next cheap win is likely to come from **reducing export-time quantization error rather than changing the core model**. This candidate applies a **data-free blockwise Walsh-Hadamard rotation to the input dimension of large 2D weights before per-row quantization**, then inverts that rotation after dequantization. The goal is to smooth row-wise outliers and make the existing GPTQ-lite clip search land on a better int6 approximation under the same 16MB budget.

## Why it is promising here

- This repository's best pure training stack is already concentrated around **11 layers + XSA + partial RoPE + LN scale + EMA + GPTQ-lite**; the remaining gap is increasingly a **quantization/export** problem rather than a missing large architectural change.
- The base implementation already uses **per-row weight quantization** (`int6` for attention/MLP, `int8` elsewhere), so a **rotation in weight space** is a natural, low-infrastructure extension.
- The dominant matrix widths in this repo are especially rotation-friendly: many large weights have an input width of **512**, and the MLP projection path uses **1536 = 3 x 512**, which makes blockwise Hadamard transforms easy to apply without changing the runtime model architecture.

## Prior repository work that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the main base because it is the strongest non-TTT stack and already isolates the value of GPTQ-lite clip search and EMA.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` showed that the strongest low-cost architectural wins have already been harvested in this family (partial RoPE, LN scale), pushing the next iteration toward export quality.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` reinforced that the top of the board is now a tightly optimized stack, so a new candidate should probe an **untried axis** rather than re-litigate the same local tweaks.

## External research that informed it

- **QuaRot** (Ashkboos et al., arXiv:2404.00456) shows that orthogonal rotations can remove outliers and make low-bit quantization much easier, including reporting lossless 6-bit and 8-bit settings without calibration.
- **SpinQuant** (Liu et al., arXiv:2405.16406) shows that the choice of rotation matters a lot for quantized accuracy; random or poorly chosen rotations can vary substantially, while better rotations materially shrink the accuracy gap.
- **OptRot** (Gadhikar et al., arXiv:2512.24124) is especially relevant to this repository because it targets **GPTQ-style weight quantization** and argues that reducing weight outliers with cheap, fusible rotations is a strong route for PTQ quality.

This candidate is intentionally a **minimal repo-compatible adaptation** of that line of work: instead of introducing learned rotations, calibration pipelines, or activation/KV quantization, it adds a simple export-time Hadamard basis change around the existing weight-only quantizer.

## What changed versus the chosen base implementation

Base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. Added **blockwise Walsh-Hadamard helpers** for large 2D weights.
2. Enabled **rotation-aware quantization metadata** so the exported model can be dequantized back into the original basis exactly for evaluation.
3. Applied the rotation to both the repo's `int6` GPTQ-lite path and its `int8` fallback path for large matrices.
4. Added candidate logging for the rotated PTQ configuration.

Nothing outside this candidate directory is modified.

## How to run / evaluate

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
ROTATED_PTQ=1 ROTATED_PTQ_MIN_BLOCK=64 ROTATED_PTQ_MAX_BLOCK=512 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The rotation path is enabled by default. `ROTATED_PTQ_MIN_BLOCK` and `ROTATED_PTQ_MAX_BLOCK` control which matrix widths are eligible.
If `DATA_PATH` and `TOKENIZER_PATH` are not set, the script resolves them from the repository root based on the location of this file, so it can be launched directly from the candidate directory.

## Validation

Commands run in this repository:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604030848_hadamard-gptq-lite/train_gpt.py
python - <<'PY'
import importlib.util
print(importlib.util.find_spec("torch"))
PY
```

Outcomes:

- `compileall` completed successfully for the baseline scripts, `data/`, and this candidate script.
- A real CPU runtime smoke test was **not feasible in this runner** because `torch` is not installed here (`find_spec("torch") -> None`), and this training script also expects the CUDA/FlashAttention runtime used by the challenge environment.

## Main expected risks / tradeoffs

- Fixed Hadamard rotations are a **cheap proxy** for the learned/data-aware rotations in SpinQuant or OptRot, so they may help some matrices and hurt others.
- The chosen block sizes are heuristic; the strongest block size may differ across attention, MLP, and embedding weights.
- If the main bottleneck in this stack is still training loss rather than export error, rotation-aware PTQ may produce only a small gain.
- Because this candidate changes only the export basis, the biggest upside is likely in **post-training quantization gap**, not raw pre-quant validation BPB.
