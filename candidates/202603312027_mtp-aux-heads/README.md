# Candidate: MTP Auxiliary Heads on the 2026-03-23 Stack

## Hypothesis

Turning on dormant **multi-token prediction (MTP)** auxiliary heads should improve **sample efficiency** inside the repository's fixed 600-second training budget, while leaving the exported artifact size unchanged because the extra heads are already excluded from serialization.

The repo's strongest record stacks already use most of the obvious architecture, quantization, and evaluation tricks. MTP targets a different bottleneck: getting more useful learning signal per token during the short training window.

## Why this is promising for this repository

- The strongest inspected record, `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`, already carries MTP plumbing in code, but all reviewed records appear to leave `MTP_NUM_HEADS=0`.
- The challenge is dominated by a **10-minute training cap**, so techniques that improve training efficiency without adding export bytes are unusually attractive here.
- This candidate keeps the strongest known training/eval stack mostly intact: LeakyReLU(0.5)^2, parameter banking + Parallel Muon, partial RoPE, XSA, VE, EMA/SWA, GPTQ-lite-style int6 export, and optional legal TTT.

## Prior repository work that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Strongest inspected overall result (`val_bpb: 1.1194` mean) and already includes dormant MTP hooks.
- **Pure training/export reference:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Shows that post-training/export improvements still matter, and also carries the same dormant MTP path.
- **Contextual architecture refinements:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Validated partial RoPE and layerwise scaling on the 11-layer family.
- **Quantization bottleneck evidence:** `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`
  - More training alone was not enough; export-aware improvements still matter.

There were **no prior `candidates/` directories** in the repo when this candidate was created.

## External research that informed it

- **Fabian Gloeckle et al., 2024, _Better & Faster Large Language Models via Multi-token Prediction_** (`arXiv:2404.19737`)
  - Trains a shared trunk with multiple future-token heads and reports improved sample efficiency.
- **Zechun Liu et al., 2024, _MobileLLM_** (`arXiv:2402.14905`)
  - Reinforces that small/sub-billion models are highly architecture- and efficiency-sensitive.
- **Mostafa Elhoushi et al., 2024, _LayerSkip_** (`arXiv:2404.16710`)
  - Another strong efficiency-oriented direction considered here, but it would require broader training/inference changes than simply activating the existing MTP path.

Among those options, MTP was the best fit because it is **already wired into the strongest local codebase**, is **training-only**, and does **not consume export bytes**.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes two targeted changes:

1. **Turns on MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.1`
   - The candidate also routes `mtp_heads` through the replicated AdamW path so the auxiliary loss actually trains both the heads and the shared trunk.
   - The existing export path already strips `mtp_heads` from the saved artifact.

2. **Adds a local-development attention fallback**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch `scaled_dot_product_attention`.
   - This does **not** change the intended H100 path when FlashAttention 3 is installed, but it makes import/smoke workflows more robust in lighter environments.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603312027_mtp-aux-heads

MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.1 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
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

For a cleaner training-only ablation, rerun with `TTT_ENABLED=0` so the measured change is isolated to MTP.

## Main expected risks and tradeoffs

- **Compute tradeoff:** two auxiliary vocab heads add training compute and may reduce total steps if the extra signal does not pay for itself.
- **Objective mismatch:** MTP improves training efficiency, but the leaderboard is still next-token BPB after quantized export (and often after TTT).
- **Interaction risk:** MTP may interact nonlinearly with LeakyReLU^2, XSA, VE, or TTT.
- **Possible saturation:** the 2026-03-23 stack is already strong, so the upside may be small unless MTP gives real efficiency gains at this scale.

## Validation

Commands run for this candidate:

```bash
python -m compileall candidates/202603312027_mtp-aux-heads/train_gpt.py
```

Outcome:

- `compileall` **passed**.

Attempted smoke validation:

```bash
python - <<'PY'
import torch
PY
```

Outcome:

- Not feasible in this workflow runner because both `python` and `python3` are missing the `torch` dependency (`ModuleNotFoundError: No module named 'torch'`).
- The candidate therefore adds a FlashAttention fallback to keep import/smoke testing straightforward in environments that have PyTorch installed but not `flash_attn_interface`.
