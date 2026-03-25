# Scheduled Sparse MTP on the 2026-03-23 stack

## Hypothesis

The current best local stack already contains dormant multi-token-prediction (MTP) heads and excludes them from the exported checkpoint, so the challenge is not artifact size but wall-clock cost. My hypothesis is that **sparse, early-phase MTP** can improve sample efficiency enough to help the 10-minute regime, while **disabling it before late warmdown** recovers endgame optimization for the main next-token objective.

In short: spend some early training budget on future-token supervision, but do not pay for it all the way to the finish line.

## Why this is promising for this repository

Recent local wins have come from techniques that are either:

- effectively free at export time, or
- training-only signals that improve the final compressed model without increasing artifact size.

Scheduled sparse MTP fits both constraints:

- the auxiliary heads are **dropped from export**, so the artifact budget is unchanged;
- the auxiliary loss is **sample-efficiency oriented**, which matters when training is capped at 600 seconds;
- sparse token sampling and an explicit warmdown cutoff keep the added compute bounded.

## Local record review that influenced this candidate

No `candidates/` directory existed when this candidate was prepared, so the comparison set was the root baseline plus the existing `records/` tree.

The main local influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - strongest current stack overall;
  - already includes dormant MTP support and strips `mtp_heads` from export;
  - serves as the implementation base here.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - strongest training-only stack before the repo jumps to legal TTT;
  - confirms that the best local recipe is already an 11-layer compressed model with EMA, XSA, partial RoPE, and GPTQ-lite export.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - explicitly documents a `torch.compile` constant-folding issue for late-QAT;
  - this motivated using **separate compiled forward paths** for MTP-on vs MTP-off rather than relying on a runtime boolean branch that compile could freeze.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`
  - reinforces the theme that the 10-minute track is extremely sensitive to wall-clock efficiency and schedule shape.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`)
  - argues that predicting multiple future tokens improves sample efficiency using auxiliary output heads on a shared trunk.
- Lorenzo Noci et al., **"Thinking into the Future: Latent Lookahead Training for Transformers"** (`arXiv:2603.20219`)
  - reinforces the idea that future-looking supervision can help models allocate computation more effectively on difficult predictions.
- John Kirchenbauer et al., **"Multi-Token Prediction via Self-Distillation"** (`arXiv:2602.06019`)
  - shows continuing interest in multi-token objectives precisely because they can improve behavior without needing deployment-time auxiliary models.

These papers do not prescribe this exact implementation; the candidate adapts their common theme to this repository's unusual combination of strict wall-clock and strict artifact limits.

## What changed versus the chosen base

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enabled MTP by default for this candidate**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`

2. **Made MTP sparse over token positions**
   - new `MTP_STRIDE` hyperparameter (default `4`);
   - each auxiliary head only scores every `MTP_STRIDE`-th eligible position, with staggered offsets across horizons.

3. **Scheduled MTP off before the endgame**
   - new `MTP_STOP_SCALE` hyperparameter (default `0.35`);
   - once the main LR multiplier falls below that scale, training switches from the compiled MTP loss path to a compiled main-loss-only path.

4. **Refactored the model into explicit forward paths**
   - `_forward_backbone(...)`
   - `forward_main(...)`
   - `forward_with_mtp(...)`
   - this avoids relying on a single runtime branch inside one compiled graph;
   - the training forward paths are recompiled once when late QAT activates so fake-quantization is not frozen out by `torch.compile`.

5. **Made path defaults candidate-directory safe**
   - default dataset and tokenizer paths now resolve from the repository root, so the script can be launched from inside this candidate directory.

6. **Added a non-FlashAttention fallback**
   - when `flash_attn_interface` is unavailable, the script falls back to causal SDPA with GQA expansion;
   - this is primarily for safer local import/smoke scenarios and should not be the preferred fast path on the real challenge hardware.

## How to run / evaluate

From this directory:

```bash
cd candidates/202603252048_scheduled-sparse-mtp
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_STRIDE=4 MTP_STOP_SCALE=0.35 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To isolate the training-only effect without legal TTT, set `TTT_ENABLED=0` and compare the pre-TTT validation path first.

## Main expected risks / tradeoffs

- If the auxiliary loss is still too expensive, the model may lose too many optimization steps before `MTP_STOP_SCALE` is reached.
- If sparse sampling is too aggressive, the future-token signal may become too weak to help.
- If MTP is turned off too early, the auxiliary supervision may not have enough time to matter.
- If it is turned off too late, the model may spend too much of warmdown optimizing a side objective.
- The SDPA fallback is correctness-oriented, not performance-oriented.

## Validation run for this candidate

The following lightweight checks were run in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603252048_scheduled-sparse-mtp/train_gpt.py
```

Outcome:

- both syntax/bytecode checks passed.

Attempted smoke test:

- I attempted a minimal CPU import-and-forward smoke check for `candidates/202603252048_scheduled-sparse-mtp/train_gpt.py`.
- That was **not feasible in this runner** because `torch` is declared in `requirements.txt` but is not installed in the current environment, so the module could not be imported for runtime execution.
