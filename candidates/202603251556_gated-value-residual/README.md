# Candidate: Gated Attention + Value Residual on the March 23 banked stack

## Hypothesis

Turning on **per-head attention gating** and **cross-layer value residual mixing** in the strongest current banked/XSA/LeakyReLU² stack should improve tiny-model sample efficiency without paying a meaningful artifact-size or systems-complexity cost.

The intuition is that the current top stack already has QK normalization, layerwise scaling, shared value embeddings, and efficient deep-layer attention refinement. Recent work suggests that adding a lightweight gate on attention outputs and a persistent value path across depth can further stabilize information flow and reduce wasted attention mass, especially in small models where every head and every residual path matters.

## Why this looks promising for this repository

This repository's best runs have converged on a fairly stable recipe:

- deeper 11-layer banked models,
- QK-normalized attention,
- partial RoPE + layerwise normalization scale,
- EMA/SWA-style late averaging,
- XSA in deep layers,
- aggressive post-training quantization,
- and recently LeakyReLU(0.5)^2 plus legal score-first TTT.

The remaining room for improvement appears to be in **small, low-risk architectural refinements** that do not disturb the optimized training systems path. This candidate keeps the March 23 Parallel Muon + parameter-banking implementation intact and only activates two already-implemented but previously undocumented architectural switches.

The repo-wide review suggested March 22 as the cleanest pre-TTT training base, but March 23 is the smallest-risk implementation fork for this idea because it already contains the exact `gated_attention` and `value_residual` paths, plus the faster banked optimizer/eval plumbing that the current top stack depends on.

## Prior records and experiments that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Supporting trends:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` showed that the 11-layer EMA + GPTQ-lite family is a strong pre-TTT base.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` established partial RoPE + layerwise scaling as reliable zero/near-zero parameter wins.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` is especially relevant as a cautionary datapoint: naive layer recurrence under a fixed wallclock budget hurt badly there, so this candidate avoids a broad shared-depth rewrite despite recurrence looking attractive in recent literature.

There were no prior `candidates/` directories to review when this candidate was created.

## External research that informed the choice

- **Value Residual Learning** (`https://arxiv.org/abs/2410.17897`) argues that standard hidden-state residuals do not always preserve token-level information well through depth, and reports that adding value residual connections can reach similar loss with fewer parameters or less data.
- **IMU-1: Sample-Efficient Pre-training of Small Language Models** (`https://arxiv.org/abs/2602.02522`) explicitly reports a recipe combining **QK-norm attention, per-head gating, value residuals, and LayerNorm scaling** for small/medium language models.
- **Forgetting Transformer: Softmax Attention with a Forget Gate** (`https://arxiv.org/abs/2503.02130`) reinforces the broader thesis that lightweight gates in attention can improve information routing while remaining compatible with FlashAttention-style kernels.
- For broader gated-attention context, see **Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free** (`https://arxiv.org/abs/2505.06708`).

The important point is not that this candidate fully reproduces any of those papers. It is that the repo already contains a cheap, compatible implementation of two ideas repeatedly highlighted by recent compact-LM work, and those ideas have not yet been surfaced in a record or candidate directory.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

- Enabled `GATED_ATTENTION` by default.
- Enabled `VALUE_RESIDUAL` by default.
- Enabled `TTT_ENABLED` by default so the out-of-the-box run follows the strongest current evaluation style.
- Set `TTT_FREEZE_BLOCKS=0` by default so legal TTT adapts all blocks, matching the stronger March 23 recipe.
- Set repo-root-relative default `DATA_PATH` and `TOKENIZER_PATH` so the script can be launched directly from this candidate directory.
- Set a candidate-specific default `RUN_ID` and `ITERATIONS=9000` to match the current 10-minute-style record regime more closely.
- Added one small log line to make the candidate-specific switches obvious in logs.

What did **not** change:

- No edits to the root `train_gpt.py`.
- No edits to any existing record directory.
- No rewrite of the parameter banks, Parallel Muon path, XSA path, quantization path, or legal score-first TTT implementation.

## How to run or evaluate it

Prerequisite: this candidate expects the repository's cached SP1024 challenge assets to exist under `data/datasets/fineweb10B_sp1024/` and `data/tokenizers/`. On a fresh checkout, populate them first from the repository root:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

See `data/README.md` for the broader data workflow and smaller smoke-download options.

From the repository root:

```bash
cd candidates/202603251556_gated-value-residual

torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides for faster iteration:

```bash
cd candidates/202603251556_gated-value-residual

TTT_ENABLED=0 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=60 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

If you prefer explicit paths instead of the new defaults:

```bash
cd candidates/202603251556_gated-value-residual

DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- The extra gates could simply be redundant with the existing QK norm + XSA + value-embedding stack.
- Value residual mixing may interact badly with deep-layer XSA if the preserved value stream becomes too dominant.
- Because the March 23 stack already includes legal TTT, the training-side architectural gain might be partially hidden or amplified by evaluation-side adaptation.
- The defaults are intentionally strong, but not yet tuned. If the idea is promising, the first follow-up sweeps should be gate bias, value-residual initialization, and whether gating should be enabled in all layers or only the deepest XSA layers.

## Validation run for this candidate

The following lightweight checks were run locally in this workflow after adding the files:

1. `python -m compileall candidates/202603251556_gated-value-residual/train_gpt.py`
   - **Outcome:** passed.
2. Attempted a CPU-only smoke-test preflight by checking whether `torch` was available in the local Python interpreter before attempting any model import or forward-pass smoke test.
   - **Outcome:** not feasible in this environment because the available Python interpreters do not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`), so a truthful model-construction smoke test could not be completed here without adding new infrastructure.

Exact commands used:

```bash
python -m compileall candidates/202603251556_gated-value-residual/train_gpt.py

python3 - <<'PY'
import torch
print(torch.__version__)
PY
```
