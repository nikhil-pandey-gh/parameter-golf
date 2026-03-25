# Forward-Curriculum MTP with Export-Pruned Auxiliary Heads

## Hypothesis

The current best local stack is already strong on evaluation tricks, legal score-first TTT, and compression-aware export. A promising next step is to improve **training-time sample efficiency** without paying extra artifact bytes or inference latency. This candidate enables a small **forward-curriculum multi-token prediction (MTP)** auxiliary loss during training, then prunes the auxiliary heads before serialization so the final artifact stays on the same deployment path.

## Why it is promising for this repository

- Recent records already harvested many obvious gains from sliding-window eval, mixed-bit export, EMA/SWA, XSA, partial RoPE, and legal TTT.
- Repo history also shows that naive recurrence and several broader architectural changes were too expensive or unstable under the 10-minute cap.
- The current best script already contained dormant MTP support and already excluded `mtp_heads.*` from export, so MTP is one of the few low-infrastructure ways to target **training efficiency** instead of artifact design.

## Prior records and experiments that influenced it

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - chosen base stack: LeakyReLU squared MLP, Parallel Muon, legal TTT, parameter banks.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - confirms that export quality is already highly optimized and incremental.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - highlighted the need to avoid compile-time dead branches for training-time ideas.
- Repo-wide synthesis from earlier records:
  - sliding-window eval was a huge early jump,
  - mixed int6/int8/fp16 export unlocked deeper/wider models,
  - mature 11-layer stacks now dominate the local frontier.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737)
  - argues that MTP improves sample efficiency by supervising multiple future tokens from a shared trunk.
- Ansar Aynetdinov and Alan Akbik, **"Pre-Training Curriculum for Multi-Token Prediction in Language Models"** (arXiv:2505.22757)
  - especially relevant here: smaller language models struggle with naive MTP, while a **forward curriculum** helps them benefit from the auxiliary objective.
- Anastasios Gerontopoulos et al., **"Multi-Token Prediction Needs Registers"** (arXiv:2505.10518)
  - reinforces that low-overhead MTP-style supervision is a plausible direction even when parameter budgets are tight.

## What changed versus the base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate:

1. Enables a **1-head MTP auxiliary loss by default**.
2. Adds a **forward curriculum** using `MTP_START_STEP` and `MTP_RAMP_STEPS` so the auxiliary loss ramps in gradually instead of appearing at full weight from step 0.
3. Keeps the final artifact budget-safe by reusing the existing **export-pruning path** that removes `mtp_heads.*` before serialization and quantization.
4. Fixes the optimizer wiring so MTP head weights are actually optimized when enabled.
5. Adds a **FlashAttention fallback** to standard PyTorch SDPA so the module can be imported for lightweight CPU smoke checks when `flash_attn_interface` is unavailable.
6. Aligns defaults more closely with the current best stack by setting `BIGRAM_VOCAB_SIZE=1536`, `TTT_ENABLED=1`, and `TTT_FREEZE_BLOCKS=0`.

## How to run or evaluate it

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

```bash
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 MTP_START_STEP=0 MTP_RAMP_STEPS=1500 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Commands run (from this candidate directory):

```bash
python -m compileall train_gpt.py
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

model = module.GPT(
    vocab_size=64,
    num_layers=2,
    model_dim=32,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2,
    tie_embeddings=True,
    tied_embed_init_std=0.01,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.0,
    mtp_num_heads=1,
    mtp_loss_weight=0.1,
    bigram_vocab_size=32,
    bigram_dim=8,
    xsa_last_n=1,
    rope_dims=4,
    ln_scale=True,
    ve_enabled=False,
).float()

x = torch.randint(0, 64, (2, 16))
y = torch.randint(0, 64, (2, 16))
loss = model(x, y)
print(float(loss))
PY
```

Outcomes:

- `compileall`: passed
- CPU forward smoke test: not run in this workflow container because both `/usr/bin/python` and `/usr/bin/python3` are missing the repo runtime dependencies (`torch`, `sentencepiece`).

## Main expected risks or tradeoffs

- Even a 1-head MTP loss adds train-time compute, so the net effect depends on whether representation gains outweigh the lost steps.
- The curriculum is grounded in recent small-model MTP work, but the exact best schedule for this unusual 11-layer Parameter Golf stack is still unknown.
- This candidate validates importability and forward correctness locally, not the final 8xH100 quality curve, so the next step after merge should be a real GPU ablation against `MTP_NUM_HEADS=0`.
