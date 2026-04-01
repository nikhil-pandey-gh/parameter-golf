# Training-Only Multi-Token Prediction on the Banked LeakyReLU² + TTT Stack

## Hypothesis

The current best record family is already heavily optimized on architecture, quantization, and evaluation. Under a strict 10-minute training budget, the cleanest remaining gain may come from **better sample efficiency**, not more artifact bytes. This candidate enables a **training-only multi-token prediction (MTP) auxiliary head** on top of the current banked SOTA stack so the shared trunk learns to predict farther ahead during training, while the extra head is still **excluded from the exported artifact**.

The key bet is that a small amount of future-token supervision can improve the pre-TTT checkpoint enough to survive the existing int6 export path and then compound with the repository's existing sliding-window + legal TTT evaluation recipe.

## Why this is promising for this repository

A few repository trends point toward sample-efficiency ideas being attractive here:

- The best records already converge on the same core stack: 11 layers, 512 dim, 3x MLP, XSA, partial RoPE, EMA/SWA, GPTQ-lite/int6 export, and finally legal score-first TTT.
- Full recurrence and slower activations like SwiGLU were already reported as net negatives in the 10-minute budget because they trade away too many training steps.
- Multiple record codepaths contain MTP support, but the reviewed runs still log `mtp_num_heads:0`, so the idea is present in code but apparently still untested as an actual training configuration.
- In the latest banked/Parallel-Muon record, the auxiliary MTP heads were defined and excluded from export, but they were not wired into the matrix optimizer path, so enabling them would not have trained them correctly without an explicit fix.

That makes MTP unusually attractive here: it is a **training-time-only change**, does **not consume artifact bytes after export**, and directly targets the repository's most obvious remaining bottleneck: how much useful learning the model can squeeze into 600 seconds.

## Prior records that influenced this candidate

This candidate is built directly from:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`

The first provides the strongest current overall stack. The latter two are important because they already contain the intended MTP loss/export pattern, which helped identify the missing optimizer wiring in the banked SOTA branch.

There was **no pre-existing `candidates/` directory** at review time, so this idea is novel relative to the repo's existing records rather than prior candidate iterations.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"**, arXiv:2404.19737, 2024.
  - Main takeaway used here: predicting multiple future tokens with independent heads on a shared trunk can improve **sample efficiency** during pretraining.
  - URL: <https://arxiv.org/abs/2404.19737>

- Anastasios Gerontopoulos et al., **"Multi-Token Prediction Needs Registers" (MuToR)**, arXiv:2505.10518, 2025.
  - Main takeaway used here: low-overhead MTP variants can be effective without large architectural disruption.
  - URL: <https://arxiv.org/abs/2505.10518>

- Somesh Mehra et al., **"On multi-token prediction for efficient LLM inference"**, arXiv:2502.09419, 2025.
  - Main caution incorporated here: frozen next-token trunks are specialized for NTP; MTP works best when trained jointly with the backbone rather than bolted on later.
  - URL: <https://arxiv.org/abs/2502.09419>

This repository's setup is a particularly good match for that caution because the candidate trains the MTP head **from scratch jointly with the trunk**, instead of retrofitting it onto an already-trained model.

## What changed vs the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in `train_gpt.py`:

1. **Enabled one auxiliary MTP head by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`

2. **Fixed the banked optimizer wiring so enabled MTP heads actually train**
   - The auxiliary `mtp_heads` weights are now added to the Parallel Muon matrix parameter set when MTP is enabled.
   - The export path is unchanged: MTP heads are still removed before serialization, so the submission artifact does not pay for them.

3. **Added a safe SDPA fallback when FlashAttention is unavailable**
   - This is only for lightweight import/CPU smoke validation.
   - Leaderboard-intended GPU runs still use `flash_attn_interface` when available.

## How to run / evaluate

From the candidate directory on the standard challenge environment with dependencies installed:

```bash
cd candidates/202604010638_mtp-aux-heads
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
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

For a cleaner ablation of the training change alone, rerun the same command with `TTT_ENABLED=0` and compare the pre-TTT / post-export metrics against the base record.

## Validation run in this workflow

Commands attempted here:

```bash
python -m compileall candidates/202604010638_mtp-aux-heads/train_gpt.py
```

Outcome:

- **Passed**.

Attempted lightweight CPU smoke test:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('candidates/202604010638_mtp-aux-heads/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

Outcome:

- **Not feasible in this runner** because the workflow environment does not currently have the repository runtime dependencies installed (`torch`, `numpy`, and `sentencepiece` were all missing, with the first failure being `ModuleNotFoundError: No module named 'torch'`).
- I therefore limited validation to syntax/bytecode compilation and kept the code changes small and localized.

## Main expected risks / tradeoffs

- **Step-time risk**: even one auxiliary future-token head adds training compute, so total steps in 600 seconds may fall slightly.
- **Scale risk**: MTP gains are strongest in larger-model literature; this repo's 16MB regime may see smaller improvements.
- **Metric interaction risk**: the final leaderboard number is heavily shaped by export quality and legal TTT, so trunk-only gains may be partially diluted.
- **Tuning risk**: `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.15` are conservative defaults, not a fully tuned optimum.

## Suggested next experiments

1. Sweep `MTP_NUM_HEADS` in `{1, 2}` with loss weights in `{0.10, 0.15, 0.20}`.
2. Measure pre-TTT gains first (`TTT_ENABLED=0`) before paying the full evaluation cost.
3. If step-time loss is negligible, try 2 heads while keeping export unchanged.
4. If MTP helps pre-TTT but not post-export, combine it with a stronger exporter experiment next.
