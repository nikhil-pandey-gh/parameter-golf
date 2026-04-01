# LeakyReLU² + MTP Warmstart on the 11L GPTQ-lite/EMA stack

## Hypothesis

A small amount of training-only multi-token prediction (MTP) should improve sample efficiency for this challenge's fixed 10-minute training budget without increasing exported artifact size, as long as the auxiliary head is kept lightweight and excluded from export.

This candidate activates a single auxiliary future-token head on top of the strongest pre-TTT stack in the repo, and warm-starts that head from the main readout instead of starting it from zeros. The hope is to get some of the representation-learning benefit reported in recent MTP papers while minimizing step-time and main-head interference.

## Why this is promising for this repository

Recent winning records in this repo have already harvested many of the obvious low-bit, attention, and evaluation tricks: XSA on late layers, EMA/SWA, partial RoPE, GPTQ-lite clip search, legal score-first TTT, and LeakyReLU². By contrast, the repo's code lineage contains dormant `MTP_NUM_HEADS` support, but the shipped record logs consistently show `mtp_num_heads:0`, so this direction appears implemented-but-untried in practice.

That makes MTP attractive here for two reasons. First, it targets training efficiency rather than post-training export quality, which is complementary to the repo's quantization-heavy progress. Second, it can be added with essentially zero artifact-cost because the auxiliary head is excluded from export.

## Prior records that influenced this candidate

This candidate is primarily based on:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the strongest clean pre-TTT stack with GPTQ-lite export, EMA, partial RoPE, LN scale, BigramHash, SmearGate, and shared value embeddings.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the LeakyReLU(0.5)^2 MLP activation, which was one of the few clearly positive late-stage training-time changes.

I intentionally did **not** stack this candidate on top of legal TTT. MTP is a training-time intervention, and isolating it on the strongest pre-TTT base keeps the candidate smaller, easier to reason about, and less likely to accidentally blow the evaluation-time budget.

## External research that informed it

- Fabian Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (2024)
  - <https://arxiv.org/abs/2404.19737>
  - motivates MTP as a sample-efficiency improvement during language-model training.
- Anastasios Gerontopoulos et al., *Multi-Token Prediction Needs Registers* (2025)
  - <https://arxiv.org/abs/2505.10518>
  - highlights that lightweight, low-overhead MTP variants can still be useful when they are careful about how auxiliary prediction is injected.
- Somesh Mehra et al., *On multi-token prediction for efficient LLM inference* (2025)
  - <https://arxiv.org/abs/2502.09419>
  - argues that hidden states are strongly specialized for next-token prediction, which is why this candidate uses only one auxiliary head rather than a large MTP fanout.
- John Kirchenbauer et al., *Multi-Token Prediction via Self-Distillation* (2026)
  - <https://arxiv.org/abs/2602.06019>
  - helped motivate warm-starting the auxiliary head from the main readout instead of zero-initializing it.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

- enabled a **single** MTP head by default:
  - `MTP_NUM_HEADS=1`
  - `MTP_LOSS_WEIGHT=0.15`
- added `MTP_INIT_FROM_MAIN=1` and initialize the auxiliary MTP head from the main readout weights instead of leaving it zero-initialized.
- swapped the MLP activation from `ReLU²` to `LeakyReLU(0.5)^2`, following the current top record's positive ablation.
- made `DATA_PATH` and `TOKENIZER_PATH` default to repo-relative paths so the script works when run from inside this candidate directory.

The rest of the stack is intentionally unchanged: 11 layers, XSA on late layers, EMA, GPTQ-lite export, BigramHash, SmearGate, shared value embeddings, partial RoPE, LN scale, and the same export/eval path.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202604010315_leaky-mtp-warmstart
SEED=1337 \
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.15 \
MTP_INIT_FROM_MAIN=1 \
python train_gpt.py
```

A closer challenge-style run on GPU hardware would typically look like:

```bash
cd candidates/202604010315_leaky-mtp-warmstart
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 MTP_INIT_FROM_MAIN=1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If the MTP head meaningfully helps, the natural next experiment is to port the same change onto the legal-TTT / Parallel-Muon stack from `2026-03-23`.

## Main expected risks and tradeoffs

- **Step-time overhead:** even a single auxiliary head adds extra vocab projections during training, so it may reduce the number of optimization steps completed in 600 seconds.
- **Main-head interference:** MTP is not free; too much auxiliary weight can hurt next-token specialization, especially in smaller models.
- **Quantization interaction is unknown:** the repo's best progress recently has come from export quality, while this idea improves the pre-export objective. It may help little if the quantization bottleneck dominates.
- **Best pairing with TTT is still unknown:** this candidate deliberately does not answer whether MTP also improves the later score-first TTT path.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202604010315_leaky-mtp-warmstart/train_gpt.py
```

Outcome:

- Passed.

Attempted minimal CPU smoke test:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch
...
PY
```

Outcome:

- Not feasible in this container because `torch` is not installed (`ModuleNotFoundError: No module named 'torch'`).
- I therefore limited validation here to syntax compilation. A real runtime smoke should be run in the normal challenge environment where PyTorch and FlashAttention are available.
