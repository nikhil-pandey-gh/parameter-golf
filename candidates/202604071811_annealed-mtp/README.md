# Annealed training-only MTP on the 1.1194 stack

## Hypothesis

The strongest next local idea is to turn the dormant **multi-token prediction (MTP)** path into a real training signal on top of the current best stack, while making sure it does **not** survive into the exported artifact.

Concretely: use a small number of auxiliary future-token heads during training, warm-start those heads from the main token projection so they produce trunk gradients immediately, and **anneal the MTP weight to zero during late warmdown / late-QAT** so the final checkpoint specializes back to the one-step BPB objective that the leaderboard scores.

## Why this is promising here

The repo trendline is clear:

- the biggest wins came from **capacity under the artifact cap** (int6/int5 + bigger MLP / more layers),
- then **quantization-friendly finishing** (EMA, GPTQ-lite, late-QAT),
- then **small but real model/eval refinements** on top of the best 11-layer stack.

The current best record is `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`, which already includes a latent MTP implementation and already strips `mtp_heads` from export. But submitted configs kept `MTP_NUM_HEADS=0`, and the copied base branch did not place the MTP head weights into any optimizer group. That meant the auxiliary path was present in code but not a credible candidate as-is.

This candidate fixes that in the smallest possible way:

1. **actually train the MTP heads,**
2. **give them useful initialization,**
3. **anneal them away before final export/eval,**
4. **preserve the existing export-time MTP stripping** so artifact bytes stay focused on the main model.

## Which records influenced it

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best current mean score (**1.1194 BPB**),
  - strongest proven stack for LeakyReLU(0.5)^2, XSA tail, partial RoPE, EMA/SWA, legal TTT, and Parallel Muon.
- **Quant-friendly finishing:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - reinforced that the last stretch of training/export handling is where small extra gains still exist.
- **Earlier 11-layer XSA line:** `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - confirmed the 11L + XSA + EMA stack is a stable base worth extending.

There were **no prior `candidates/` directories** in the repository when this candidate was created, so this does not repeat an earlier candidate branch.

## External research that informed it

- **Fabian Gloeckle et al., _Better & Faster Large Language Models via Multi-Token Prediction_ (arXiv:2404.19737, 2024).**
  - Trains the model to predict multiple future tokens with independent output heads on top of a shared trunk.
  - Reports better sample efficiency and especially strong gains on generative tasks.
- **DeepSeek-AI, _DeepSeek-V3 Technical Report_ (arXiv:2412.19437, 2025).**
  - Explicitly adopts an MTP objective for stronger performance.
  - Motivates MTP as a way to **densify training signals**, improve data efficiency, and help the model **pre-plan representations** for future tokens.

Those papers are a good fit for this repo because the challenge is heavily **time-budgeted**: if MTP improves sample efficiency during the fixed 10-minute training window, the gain can survive even after the auxiliary heads are removed from export.

## What changed vs the chosen base

Compared with `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

1. turns on a candidate default of **`MTP_NUM_HEADS=2`** and **`MTP_LOSS_WEIGHT=0.15`**,
2. adds **`MTP_DECAY_SCALE=0.15`** so the MTP loss weight decays with the late LR / warmdown scale,
3. adds the **MTP head weights to the AdamW auxiliary parameter group** so they are actually optimized,
4. **warm-starts the MTP heads from the main token projection** (tied embedding weights) instead of leaving them as a dead zero-init path,
5. keeps the existing **export-time exclusion of `mtp_heads`** so the final artifact still contains only the main model,
6. adds a **FlashAttention -> PyTorch SDPA fallback** so the module can at least be imported and smoke-instantiated without FlashAttention.

## How to run

From the repository root:

```bash
cd candidates/202604071811_annealed-mtp

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
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_DECAY_SCALE=0.15 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`train_gpt.py` resolves its default dataset and tokenizer paths from the script location, so the command above still points at the repository-level `data/` tree after `cd` into the candidate directory.

The intended scoring behavior is unchanged from the base branch: the exported checkpoint still drops `mtp_heads`, then evaluates the quantized roundtrip and the legal score-first TTT path.

## Main risks / tradeoffs

- **Training throughput risk:** extra MTP heads add forward/backward work and optimizer state, which could reduce the number of 10-minute steps.
- **Objective mismatch risk:** if the auxiliary loss remains too strong too late, it can help sample efficiency early but hurt final one-step BPB; that is why this branch anneals MTP toward zero during late warmdown.
- **Initialization risk:** warm-starting MTP heads from the token projection is intended to avoid a zero-gradient cold start, but it may also over-couple auxiliary heads to the main token space.
- **Best hyperparameters are still uncertain:** the first follow-up ablations should be `MTP_NUM_HEADS in {1,2,4}` and `MTP_DECAY_SCALE in {0.10, 0.15, 0.20}`.

## Validation run here

### Commands

```bash
python -m compileall candidates/202604071811_annealed-mtp/train_gpt.py
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('candidates/202604071811_annealed-mtp/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

### Outcomes

- `python -m compileall ...` **succeeded**.
- A minimal CPU import/smoke attempt was **not feasible in this workflow environment** because the available Python interpreter did not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
- The candidate therefore has **syntax validation only** in this run, but the code now includes a PyTorch SDPA fallback specifically so a CPU smoke import/forward is possible in an environment that does have `torch`.
