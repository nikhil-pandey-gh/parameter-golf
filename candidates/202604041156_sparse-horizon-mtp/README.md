# Sparse Horizon MTP on the 11L Legal-TTT Stack

## Hypothesis

The current best lineage has already squeezed a lot out of evaluation strategy, mixed-bit compression, and small architectural tweaks. The next cheap place to buy quality is **sample efficiency during the fixed 600s training run**: add **training-only multi-token prediction (MTP)** heads so each hidden state learns to predict a short future horizon, but keep the extra compute under control with **token-strided supervision** and **horizon-decayed weighting**.

The final artifact budget should stay effectively unchanged because the auxiliary MTP heads are still stripped from the exported checkpoint.

## Why this looks promising for this repository

- The strongest record line already converged on a shared stack: 11 layers, 3x MLP, XSA on late layers, partial RoPE, EMA/SWA, and aggressive post-training compression.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is currently the strongest local base. Its README shows the latest gain came from a cheap optimization-side change (LeakyReLU^2), not a broad rewrite.
- Both `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` and `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` already contain dormant MTP code paths, but no record README reports an actual MTP run.
- In the 2026-03-23 code snapshot, the MTP heads are created and exported away, but they are **not added to any optimizer**. Turning `MTP_NUM_HEADS>0` on there would not really train those heads. This candidate restores that path and makes it cheap enough to try.
- Prior non-record work in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` showed that slower, deeper, or recurrent variants can easily lose more wall-clock training steps than they gain in modeling power. That pushes this candidate toward a lightweight auxiliary loss instead of another recurrent/shared-layer experiment.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **MTP scaffold reference:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Do-not-repeat signal:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` documented layer recurrence as a negative result.

There were no prior `candidates/` directories in the repository when this candidate was created.

## External research that informed it

- **Better & Faster Large Language Models via Multi-token Prediction** (arXiv:2404.19737) argues that predicting multiple future tokens can improve pretraining sample efficiency with independent horizon heads on top of a shared trunk.
- **LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding** (arXiv:2404.16710) is a useful adjacent result: training-time auxiliary supervision on internal representations can pay off even when inference still uses the full model.
- **Self-Distillation for Multi-Token Prediction** (arXiv:2603.23911) highlights that farther MTP heads are harder to optimize jointly. That pushed this candidate toward a short horizon plus **decayed weighting** instead of equally-weighted deeper horizons.
- **Multi-Token Prediction Needs Registers** (arXiv:2505.10518) suggests stronger MTP variants may need architectural changes, but adding register tokens would be a much larger deviation from the existing single-file trainer than this repository seems to want for fast iteration.

## What changed versus the chosen base

This candidate copies the 2026-03-23 record stack and makes four targeted changes:

1. **Turns MTP on by default** with `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15`.
2. **Adds token-strided MTP supervision** via `MTP_TOKEN_STRIDE=4`, so auxiliary heads only score every fourth position instead of every token.
3. **Adds horizon-decayed weighting** via `MTP_HORIZON_DECAY=0.5`, so the nearest future head gets the strongest gradient and farther heads contribute less.
4. **Restores optimizer updates for MTP heads** by adding them back to the Muon parameter set. Without this, zero-initialized MTP heads never move and the auxiliary path is effectively dead.

Two small config alignments from the base record are also made explicit in defaults:

- `BIGRAM_VOCAB_SIZE=1536`
- `TTT_FREEZE_BLOCKS=0`

The export path is unchanged: `mtp_heads` are still excluded from the final artifact.

## How to run

From this candidate directory:

```bash
cd candidates/202604041156_sparse-horizon-mtp
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_TOKEN_STRIDE=4 MTP_HORIZON_DECAY=0.5 \
BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script resolves `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root by default, so it can be launched from this candidate directory without rewriting those paths.

## Main risks and tradeoffs

- **Wall-clock risk:** even strided MTP adds extra logits and extra optimizer work, so the net result depends on whether the quality gain beats the step-count loss.
- **Head instability:** farther-horizon heads can be noisy; this is why the default setup keeps only two heads and decays the second head's weight.
- **Compile-path risk:** this candidate deliberately avoids more invasive runtime toggles or late-activation tricks after the repo already hit compile-folding pitfalls with earlier QAT experiments.
- **No artifact help:** unlike a quantization or factorization change, this idea buys training signal, not bytes. It needs a real BPB gain to matter.

## Validation

Commands run during candidate creation:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202604041156_sparse-horizon-mtp/train_gpt.py
```

Outcomes:

- `python -m compileall train_gpt.py train_gpt_mlx.py data` succeeded.
- `python -m compileall candidates/202604041156_sparse-horizon-mtp/train_gpt.py` succeeded.
- A CPU-only runtime smoke test was **not** run because this trainer hard-requires CUDA in `main()` and imports Hopper-specific `flash_attn_interface` at module import time, so there is no existing low-cost CPU execution path to validate without changing the runtime contract.
