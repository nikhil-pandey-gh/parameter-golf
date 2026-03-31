# Weighted MTP on the LeakyReLU2 + Legal-TTT stack

## Hypothesis

The repo's best runs are heavily wallclock-limited rather than artifact-limited during training. A lightweight **multi-token prediction (MTP)** auxiliary loss should improve sample efficiency inside the same 600-second budget, while costing essentially **zero final artifact bytes** because the auxiliary heads are excluded from export.

This candidate keeps the current strongest public stack shape from `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, but turns on **weighted two-head MTP** during training. The nearer future target gets full auxiliary weight and the farther target gets half weight, so the trunk is nudged toward longer-horizon structure without letting the extra heads dominate optimization.

## Why this is promising for this repository

Three repo patterns point in the same direction:

- The strongest records repeatedly win by squeezing more useful learning into the fixed 10-minute budget rather than by massively changing model size.
- Recent scripts already contain dormant MTP plumbing, but every checked record log still shows `mtp_num_heads:0`, so this direction appears untried in a scored run.
- The current export path already drops `mtp_heads.*` from the saved artifact and evaluates with `mtp_num_heads=0`, so the auxiliary loss is almost free at submission time.

That combination makes MTP unusually attractive here: it directly attacks the training bottleneck without fighting the 16 MB artifact cap.

## Prior records that influenced this candidate

### Chosen code base

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

This base already contains the current high-performing training/eval stack:

- LeakyReLU(0.5)^2 MLP
- parameter-banked Parallel Muon
- 11-layer XSA / partial-RoPE / VE line
- legal score-first TTT
- int6 + lzma export

### Supporting prior evidence

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - establishes the strong 11-layer EMA / XSA / partial-RoPE / export-focused line.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - includes the same dormant MTP hooks, but logs still show `mtp_num_heads:0`, which suggests the mechanism exists but was never actually tried in the tracked runs.

## External research that informed this candidate

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737)
  - argues that predicting multiple future tokens from a shared trunk improves sample efficiency and can help induction-style behaviors.
- Anastasios Gerontopoulos et al., **"Multi-Token Prediction Needs Registers"** (arXiv:2505.10518)
  - suggests horizon structure matters; this candidate keeps the implementation simpler than register tokens, but uses **decaying head weights** to reflect that near-horizon targets should matter more.
- John Kirchenbauer et al., **"Multi-Token Prediction via Self-Distillation"** (arXiv:2602.06019)
  - further supports MTP as an active direction rather than a one-off 2024 result.

## What changed versus the chosen base implementation

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes four focused changes:

1. **Turns on MTP by default**
   - `MTP_NUM_HEADS` now defaults to `2`.
   - `MTP_LOSS_WEIGHT` now defaults to `0.15`.

2. **Adds weighted MTP horizons**
   - New `MTP_HEAD_WEIGHTS` env var, defaulting to `1.0,0.5`.
   - The auxiliary heads are still contiguous future-token heads, but the farther horizon is intentionally down-weighted.

3. **Keeps export artifact-clean**
   - The script already excluded `mtp_heads.*` from export; this candidate keeps that behavior and makes it central to the hypothesis.
   - The dequantized eval model is still instantiated with `mtp_num_heads=0`, so submission bytes stay focused on the actual next-token model.

4. **Adds a CPU smoke-test path**
   - `CPU_SMOKE_TEST=1` instantiates a tiny CPU model, runs a forward/backward pass, strips the MTP heads from export, loads the export into an eval model with `mtp_num_heads=0`, and checks logits.
   - A small SDPA fallback is used when FlashAttention is unavailable, so this smoke path works without GPU kernels.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202603312102_weighted-mtp

BIGRAM_VOCAB_SIZE=1536 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_HEAD_WEIGHTS=1.0,0.5 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a quick local CPU sanity check only:

```bash
CPU_SMOKE_TEST=1 python train_gpt.py
```

## Validation run for this candidate

Validation performed in this workflow:

- `python -m compileall candidates/202603312102_weighted-mtp/train_gpt.py`
  - **Passed**.
- `CPU_SMOKE_TEST=1 python candidates/202603312102_weighted-mtp/train_gpt.py`
  - **Passed** in a temporary venv after installing `numpy`, `sentencepiece`, and `torch`, because the runner's system Python did not have the repo runtime dependencies preinstalled.
  - Output: `cpu_smoke_test:ok`

## Main expected risks / tradeoffs

- **Step-time overhead**: two auxiliary vocab heads are not free, and the main uncertainty is whether the extra learning signal beats the throughput loss in this very short training regime.
- **Horizon choice**: the script currently uses the existing contiguous future-head setup (`t+2`, `t+3` relative to the base next-token target), not a more elaborate register-token or leap-token formulation.
- **Interaction with TTT**: stronger pretrained features could amplify legal TTT gains, but MTP might also shift representation geometry in a way that changes the best TTT hyperparameters.
- **Late-QAT remains inherited**: this candidate's core hypothesis is MTP-driven sample efficiency, not a new claim about the inherited late-QAT path.

## Suggested next experiments if this direction works

1. Sweep `MTP_NUM_HEADS` in `{1, 2, 3}` while keeping export pruning unchanged.
2. Compare `MTP_HEAD_WEIGHTS=1.0`, `1.0,0.5`, and `1.0,0.5,0.25`.
3. If pre-quant quality improves but roundtrip quality lags, combine this candidate with a stronger activation-aware export idea.
