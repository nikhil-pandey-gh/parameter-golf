# Frontier MTP on the 2026-03-23 stack

## Hypothesis

Enable a **real, training-only multi-token prediction (MTP) auxiliary head** on top of the current frontier stack. The bet is that a single extra future-token head improves sample efficiency during the fixed 600s training window, while keeping the exported artifact essentially unchanged because the auxiliary head is stripped before serialization.

This targets a clear gap in the repo history: the strongest record scripts already carry MTP loss/export code, but the logged frontier runs all kept `mtp_num_heads:0`, so the idea was never actually tested in a leaderboard-strength configuration.

## Why this is promising here

- Recent repo gains mostly came from evaluation tricks, quantization, activation tuning, and optimizer refinements. A **training-objective** improvement is still underexplored.
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) reports higher sample efficiency from predicting multiple future tokens with auxiliary heads.
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) explicitly highlights a **multi-token prediction training objective for stronger performance**.
- Repo evidence argues against heavier architectural bets like depth recurrence under a 10-minute wallclock cap, so a training-only auxiliary is a safer fit than another large structural rewrite.

## Prior repo influence

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- **Relevant prior records:**
  - `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: current best overall stack
  - `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: best non-TTT stack
  - `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: partial RoPE + LN scale
  - `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`: earlier 11-layer XSA/EMA frontier
- **Prior candidates:** none existed in `candidates/` before this run.

## What changed vs the chosen base

1. **Default MTP is now on:** `MTP_NUM_HEADS` defaults to `1` instead of `0`.
2. **MTP heads are actually trainable on the parameter-banked stack:** the copied 2026-03-23 script declared `mtp_heads` and used them in the loss, but did not add their weights to any optimizer group. This candidate wires those weights into the replicated AdamW path so they receive gradients, all-reduce, and parameter updates.
3. **Export behavior stays training-only:** `mtp_heads.*` are still excluded from `export_sd`, so the candidate preserves the intended “learn with MTP, ship without MTP heads” artifact strategy.

## How to run

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.2 \
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

Useful ablation:

```bash
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Evaluation / expected behavior

- Training should log a nonzero `mtp_num_heads` and nonzero `mtp_params`.
- Export should still log `export_excluding_mtp_params:...`, confirming the auxiliary head is not counted in the shipped artifact.
- The main comparison is against the same stack with `MTP_NUM_HEADS=0`.

## Main risks and tradeoffs

- **More head FLOPs:** one extra vocab projection can reduce steps completed in 600s, so the gain must come from better sample efficiency, not raw throughput.
- **Small effect size:** MTP may help only slightly once the stack is already very strong and TTT dominates the final score.
- **Optimizer choice:** on this parameter-banked stack the MTP heads ride the replicated AdamW path rather than Parallel Muon; that is deliberate for simplicity, but it may not be optimal.
- **Best setting is unclear:** `MTP_NUM_HEADS=1` is the safest first guess, but `MTP_LOSS_WEIGHT` and the number of auxiliary heads likely need ablation.

## Validation

Commands run:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604090516_frontier-mtp/train_gpt.py
python - <<'PY'
# attempted CPU-only import/forward smoke
...
PY
```

Outcomes:

- `compileall` completed successfully.
- A minimal CPU smoke test was **not feasible in this runner** because the available Python environment does not currently have `torch` installed (`requirements.txt` lists it, but it is absent from the workflow environment), so the candidate could not be imported for a live forward pass here.
