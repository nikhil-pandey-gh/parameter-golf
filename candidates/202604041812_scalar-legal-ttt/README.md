# Scalar-Only Legal TTT on the 2026-03-23 Record Stack

## Hypothesis

The current best stack already shows that **legal score-first test-time training (TTT)** can buy a real validation gain, but adapting the entire network is expensive and may be broader than necessary. This candidate tests whether a **parameter-efficient TTT subspace** — the model's existing scalar/control tensors such as `attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, `skip_weights`, `smear.gate`, `bigram.scale`, and VE scales — can retain much of the adaptation benefit with lower evaluation-side compute and less risk of overfitting each chunk.

## Why this is promising for this repository

- The strongest existing record is `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`, so TTT is already a proven high-leverage axis here.
- Earlier repo history also includes `records/track_10min_16mb/2026-03-17_LoRA_TTT/`, which suggests **parameter-efficient** adaptation is viable, even if that specific LoRA variant was not the final frontier.
- This repo's current strong stacks already expose a rich set of low-dimensional control parameters; adapting those is much cheaper than mutating the banked attention/MLP weights.
- A scalar-only update rule is easy to ablate against the existing full-network TTT path because the script keeps `TTT_PARAM_MODE=all` available as a fallback.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - chosen base implementation
   - contributes legal score-first TTT, LeakyReLU(0.5)^2, parameter banking, Parallel Muon, and the rest of the current best stack
2. `records/track_10min_16mb/2026-03-17_LoRA_TTT/`
   - motivates keeping the adaptation step parameter-efficient instead of updating every weight
3. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - provides the stable 11-layer pre-TTT backbone lineage that the current record still builds on

## External research that informed it

- **LoRA** — Hu et al., 2021, arXiv:2106.09685  
  Low-rank adaptation shows that strong adaptation often lives in a much smaller update subspace than full fine-tuning.
- **BitFit** — Ben Zaken et al., 2021, arXiv:2106.10199  
  Updating only a very small subset of parameters can remain competitive, which supports trying an even cheaper TTT path over scalar/control tensors only.

These papers are not TTT papers themselves, but they directly motivate the candidate's main bet: **for a pretrained LM, adaptation quality can come from a narrow parameter subspace** rather than full-network mutation.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added `TTT_PARAM_MODE` with:
   - `scalar` (new default): adapt only scalar/control tensors
   - `all`: recover the original record-style full-parameter TTT behavior
2. Switched defaults to make this candidate run its own intended eval path by default:
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`
   - `TTT_PARAM_MODE=scalar`
3. Added validation for an empty TTT parameter selection so the candidate fails loudly instead of silently skipping adaptation.
4. Added logging so runs record which TTT parameter mode was used.

## Run / evaluate

From the candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_PARAM_MODE=scalar TTT_LR=0.002 TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To compare directly against the original full-parameter TTT path, rerun with:

```bash
TTT_PARAM_MODE=all
```

## Main risks and tradeoffs

- Scalar-only TTT may be **too constrained** and leave some of the full-network TTT gain on the table.
- The best learning rate for full-parameter TTT may not transfer; scalar-only adaptation may want a larger `TTT_LR` or more epochs.
- Because this keeps the record stack otherwise unchanged, the candidate's upside is likely to come from **TTT efficiency and stability**, not from a radically different pre-training curve.

## Validation

Commands run in this repo:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202604041812_scalar-legal-ttt/train_gpt.py
```

Outcome:

- both compile checks succeeded
- no CPU runtime smoke test was feasible in this container because `torch` is not installed locally and this record-derived script hard-requires CUDA plus FlashAttention
