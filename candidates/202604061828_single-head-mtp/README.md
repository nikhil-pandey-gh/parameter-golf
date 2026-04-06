# Single-Head MTP on the Parallel-Muon Stack

## Hypothesis

A **single training-only multi-token prediction (MTP) head** should improve sample efficiency during the 600-second training budget without increasing the exported artifact size, because the auxiliary head is stripped before serialization and never used at eval time.

## Why this is promising here

The repo's strongest recent runs already stack most of the obvious architecture and quantization wins: 11 layers, 2048 context, partial RoPE, XSA, EMA, GPTQ-lite, and the later LeakyReLU(0.5)^2 activation. That makes a **training-efficiency** idea more attractive than another small architecture tweak.

This candidate focuses on a path that was already partially wired into the record lineage but not actually active in the latest stack:

1. `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` introduced MTP scaffolding and already excluded `mtp_heads` from export.
2. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the strongest current base stack, but its Parallel-Muon refactor dropped the MTP heads from optimizer parameter groups, so enabling MTP there would have been effectively inert.
3. No prior `candidates/` directory existed, so this is the first candidate iteration in the repo.

## External research

The implementation is guided by three papers:

1. **Gloeckle et al., _Better & Faster Large Language Models via Multi-token Prediction_ (arXiv:2404.19737)**: argues that MTP improves sample efficiency and downstream quality while sharing one trunk across multiple future-token heads.
2. **Gerontopoulos et al., _Multi-Token Prediction Needs Registers_ (arXiv:2505.10518)**: shows that parameter-light MTP variants can work well when they avoid large architectural overhead.
3. **Zhao et al., _Self-Distillation for Multi-Token Prediction_ (arXiv:2603.23911)**: highlights that multi-head MTP is harder to train jointly, which motivates the conservative choice here of **one auxiliary head** instead of a wider horizon sweep.

## What changed vs. the chosen base

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Turn MTP on by default** with `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.15`.
2. **Fix the optimizer plumbing** so `mtp_heads` are included in the Muon matrix parameter group. This is the key correctness fix: without it, zero-initialized MTP heads never update, so the auxiliary loss cannot influence the trunk.
3. **Keep MTP artifact-neutral** by preserving the existing export path that strips `mtp_heads` from the serialized model.
4. **Add a CPU-safe grouped-attention fallback** using `torch.nn.functional.scaled_dot_product_attention` when `flash_attn_interface` is unavailable, so the module is importable in lighter environments for smoke tests.

I intentionally left legal TTT disabled by default so the training-side effect can be measured independently. The TTT path remains in the script and can still be layered back on after a clean ablation.

## How to run or evaluate

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional follow-up once the training-only ablation is understood:

```bash
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0
```

## Main risks and tradeoffs

1. **Step-time overhead**: even one extra vocab projection head adds training FLOPs, so any sample-efficiency win has to beat the small loss in steps completed.
2. **Loss balancing**: `MTP_LOSS_WEIGHT=0.15` is deliberately conservative, but the optimum may be lower or slightly higher.
3. **Single-head may be too timid**: if one future token helps but the effect is small, the next experiment should sweep `MTP_NUM_HEADS in {1,2}` and `MTP_LOSS_WEIGHT`.
4. **Eval interaction remains uncertain**: the main hope is lower pre-quant loss and better hidden states; whether that stacks cleanly with sliding-window eval and legal TTT still needs a GPU run.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202604061828_single-head-mtp/train_gpt.py
```

Outcome: **passed**.

Attempted smoke check:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

path = Path('candidates/202604061828_single-head-mtp/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
...
PY
```

Outcome: **blocked by missing environment dependencies** in this container (`torch`, `numpy`, and `sentencepiece` are not installed here). The code now includes a CPU-safe attention fallback so the smoke test should work in a normal repo environment once those standard dependencies are present.
