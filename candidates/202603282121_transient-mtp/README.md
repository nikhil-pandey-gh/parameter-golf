# Transient MTP on the current 11L banked stack

## Hypothesis

Add **training-only multi-token prediction (MTP)** heads to the current best stack so the trunk learns a richer future-aware objective during the 10-minute training budget, while still exporting the same next-token model artifact.

This is attractive for Parameter Golf because the auxiliary heads are useful **during training** but are already excluded from the final exported state dict, so the candidate mainly pays extra training compute rather than extra artifact bytes.

## Why this is promising for this repository

The recent leaderboard trend is clear: the best runs are already close to saturating the easy artifact-side wins, stacking together partial RoPE, LN scaling, XSA, EMA, GPTQ-lite, LeakyReLU squared, and legal TTT:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Those runs improve the same 11-layer family mostly through better regularization, quantization, evaluation, and small architectural refinements. MTP attacks a different bottleneck: **sample efficiency under a fixed wallclock**.

The current best script already contains dormant MTP plumbing and already strips `mtp_heads.*` from export, but the recent logs show it was never actually used:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log` logs `mtp_num_heads:0`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train.log` logs `mtp_num_heads:0`

That makes MTP a good next candidate: it is **new to this repo’s record history**, already close to implementation, and fits the challenge’s fixed-size export regime unusually well.

## External research that informed this candidate

The main source is:

- Fabian Gloeckle et al., **“Better & Faster Large Language Models via Multi-Token Prediction”** (2024), `https://arxiv.org/abs/2404.19737`

Key reason this paper fits here:

- it reports that predicting multiple future tokens with independent heads on a shared trunk improves **sample efficiency**
- the extra heads can be viewed as an **auxiliary training task**
- inference can keep the normal next-token interface, which maps well onto this repo’s “train rich, export lean” setup

I also considered more structural changes such as cross-layer sharing from ALBERT / Universal Transformer:

- ALBERT, `https://arxiv.org/abs/1909.11942`
- Universal Transformer, `https://arxiv.org/abs/1807.03819`

Those are promising longer-term, but MTP is the lower-risk fit for this repository right now because the codebase already contains most of the necessary scaffolding.

## Chosen base implementation

This candidate is based directly on:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

That base already includes the strongest public stack in this repository:

- 11 banked layers at 512d
- XSA on the deepest 4 layers
- partial RoPE (`16/64`)
- LN scale
- shared value embeddings
- LeakyReLU(0.5)^2 MLP
- EMA + tight SWA
- GPTQ-lite style int6 export
- legal score-first TTT path

## What changed vs the chosen base

This candidate keeps the main stack intact and makes the MTP path real:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS` now defaults to `2`
   - `MTP_LOSS_WEIGHT` now defaults to `0.15`

2. **Actually optimize the MTP heads**
   - the base script created `mtp_heads`, used them in the loss, and excluded them from export
   - but it did **not** attach them to any optimizer
   - this candidate adds a separate `AdamW` optimizer for `mtp_heads` with `MTP_HEAD_LR`

3. **Initialize MTP heads from the tied embedding**
   - the base script zero-initialized them like export heads
   - for an auxiliary loss, that delays trunk-side gradient signal
   - this candidate copies the tied token embedding into each MTP head at init, so the auxiliary objective can shape the trunk immediately

4. **Keep MTP training-only**
   - the export path still drops `mtp_heads.*`
   - eval still instantiates the dequantized model with `mtp_num_heads=0`
   - artifact size therefore stays aligned with the standard next-token model

5. **Add an explicit SDPA fallback**
   - if `flash_attn_interface` is unavailable, attention falls back to `torch.scaled_dot_product_attention`
   - the CUDA + FlashAttention path remains unchanged when available
   - this only exists to make lightweight import/forward smoke tests possible in less specialized environments

## How to run

From the repository root:

```bash
cd candidates/202603282121_transient-mtp

RUN_ID=transient_mtp \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_HEAD_LR=0.008 \
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

For quick syntax validation only:

```bash
python -m compileall candidates/202603282121_transient-mtp/train_gpt.py
```

## Main expected risks and tradeoffs

- **Training compute:** MTP adds extra logits and losses during training, so it may reduce raw step count.
- **Auxiliary objective balance:** `MTP_LOSS_WEIGHT=0.15` is a reasonable starting point, not a tuned optimum.
- **Horizon count:** 2 heads is intentionally conservative; more horizons may help sample efficiency but may cost too much wallclock.
- **Interaction with TTT:** MTP changes the trained trunk representation, so its interaction with the repo’s legal TTT recipe is still uncertain.
- **Potential over-regularization late in training:** if the auxiliary loss stays too strong during warmdown, it could blunt last-mile next-token specialization.

## Validation run in this workflow

### Completed

```bash
python -m compileall candidates/202603282121_transient-mtp/train_gpt.py
```

Outcome: **passed**

### Attempted but blocked

I attempted a minimal CPU import/forward smoke test using a toy `GPT(...)` instantiation loaded from the candidate file. That was blocked on this runner because `torch` is not installed in the active Python environment.

Evidence:

```bash
python - <<'PY'
import importlib.util
print(importlib.util.find_spec('torch'))
PY
```

Output:

```text
None
```

`requirements.txt` in the repository does list `torch`, so this is an environment limitation of the current workflow runner rather than a missing dependency declaration.
