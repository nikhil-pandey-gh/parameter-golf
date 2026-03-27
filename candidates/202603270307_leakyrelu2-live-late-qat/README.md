# Candidate: LeakyReLU^2 + Live Late QAT on the GPTQ-lite 11L stack

## Hypothesis

Take the strongest non-TTT training stack in the repo (`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`) and make two tightly coupled changes:

1. Replace `relu^2` with `LeakyReLU(0.5)^2` in the MLP.
2. Make late int6 fake-quantization actually go live under `torch.compile` by recompiling the training wrapper when late QAT is enabled.

The hypothesis is that the activation change improves optimization and feature usage in the 11-layer GPTQ-lite stack, while a real late-QAT phase reduces the train/deploy mismatch that still matters in this artifact-constrained regime.

## Why this is promising for this repository

The repo's best recent non-TTT result is the 11-layer EMA + GPTQ-lite stack at `1.1233` BPB, and the best overall result shows that `LeakyReLU(0.5)^2` was a cheap but meaningful gain on a nearby architecture family.

At the same time, the prior partial-RoPE record explicitly documented that its late-QAT path was dead because `torch.compile` constant-folded the class flag. The 2026-03-22 stack still uses the same class-level `CastedLinear._qat_enabled` pattern, so the safest next step is not a broad architectural rewrite but a precise attempt to combine the best mature stack with a known-good activation tweak and a compile-safe QAT transition.

## Prior experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen as the base because it is the strongest pure training/quantization stack in the repo.
  - Reused: 11L shape, XSA, partial RoPE, LN scale, VE, SmearGate, BigramHash, EMA, GPTQ-lite, mixed int6/int8 export.

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Motivation for the MLP change.
  - Its README reports `LeakyReLU(0.5)^2` as a roughly `-0.0021` pre-TTT gain on a nearby 11-layer family.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Motivation for the QAT fix.
  - Its README explicitly states that late QAT had no effect because `torch.compile` constant-folded the QAT flag.

There were no prior `candidates/` directories in this repository when this candidate was created.

## External research that informed it

- **Primer: Searching for Efficient Transformers for Language Modeling** (`arXiv:2109.08668`)
  - Primer reports that squaring ReLU activations is one of the simple changes that improves autoregressive Transformer efficiency and quality.
  - This supports continuing to treat squared-rectifier MLPs as a high-signal part of the search space here.

- **Delving Deep into Rectifiers** (`arXiv:1502.01852`)
  - PReLU/rectifier work motivates preserving gradient flow through negative pre-activations at very low extra cost.
  - This is the main reason to test `LeakyReLU(0.5)^2` instead of plain `relu^2` on the strongest GPTQ-lite stack.

- **BitNet b1.58** (`arXiv:2402.17764`)
  - BitNet reinforces the broader point that training with low-bit awareness can preserve or recover quality that would otherwise be lost at deployment time.
  - This supports fixing the repo's broken late-QAT path instead of treating low-bit export as purely post-training.

- **SpinQuant** (`arXiv:2405.16406`)
  - SpinQuant shows how sensitive LLM quality can be to quantization-friendliness and the remaining gap between full precision and low-bit deployment.
  - This is directly relevant because the repo's best stacks still depend on aggressive int6 export under a 16 MB artifact budget.

## What changed versus the chosen base implementation

Base file:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **LeakyReLU^2 MLP**
   - Added `MLP_NEGATIVE_SLOPE` hyperparameter with default `0.5`.
   - Changed the MLP from:
     - `relu(fc(x))` then square
   - to:
     - `leaky_relu(fc(x), negative_slope=0.5)` then square

2. **Compile-safe late QAT**
   - Kept the existing late-QAT trigger semantics (`LATE_QAT_THRESHOLD`).
   - When the trigger fires, the script now:
     - sets `CastedLinear._qat_enabled = True`
     - rebuilds the compiled training wrapper
   - This is meant to avoid the prior constant-folding failure mode where the compiled graph never saw the QAT branch become active.

3. **Candidate-only runtime robustness**
   - Added a `flash_attn_interface` import fallback.
   - If FlashAttention 3 is unavailable, the attention path falls back to PyTorch SDPA.
   - This does not change the intended GPU path when FA3 is installed, but it makes the file importable in lighter environments.

## How to run or evaluate it

From this candidate directory:

```bash
NUM_LAYERS=11 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
MLP_NEGATIVE_SLOPE=0.5 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults in `train_gpt.py` already target the 11-layer GPTQ-lite family, so the most important candidate-specific knob is `MLP_NEGATIVE_SLOPE=0.5`.
EMA is always enabled in this script with the same fixed `0.997` decay used by the base record.

## Validation commands and outcomes

Ran:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603270307_leakyrelu2-live-late-qat/train_gpt.py
```

Outcome:

- Passed.

Attempted CPU smoke test:

```bash
python3 - <<'PY'
import importlib.util
from pathlib import Path
import torch
PY
```

Outcome:

- Not feasible in this runner because `torch` is not installed in the available Python environment (`ModuleNotFoundError: No module named 'torch'`).
- Because of that environment limitation, I could not run a real import-and-forward smoke test here.

## Main expected risks and tradeoffs

- **Late-QAT recompilation cost:** enabling QAT now forces a one-time recompilation during training, which may slightly perturb the 10-minute budget.
- **Activation/quantization interaction:** `LeakyReLU^2` helped on the TTT/parallel-Muon family, but it has not yet been isolated on the GPTQ-lite + zstd stack.
- **No on-runner torch validation:** syntax was validated, but an actual forward pass was not possible in this environment because PyTorch is unavailable here.
- **Still a narrow intervention:** this candidate deliberately avoids a larger architecture change, so upside may be smaller than a more radical parameter-sharing or evaluation-time idea if those were tuned successfully.
