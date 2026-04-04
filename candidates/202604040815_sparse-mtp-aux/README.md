# Sparse MTP Auxiliary Loss on the 11L EMA/GPTQ-lite Stack

## Hypothesis

A **training-only sparse multi-token prediction (MTP) auxiliary loss** can improve sample efficiency on the repository's strongest non-TTT 11-layer stack without increasing exported artifact size. The key adaptation for this challenge is to supervise only a tail slice of each sequence, so the extra vocab-head softmax work stays small enough for the 10-minute training budget.

## Why this is promising here

The record history converged on a consistent recipe:

- sliding-window evaluation is a large win, but it is already saturated in the strong stacks,
- quantization/export quality matters enormously,
- 11 layers + MLP3x + XSA + partial RoPE + LN scale + EMA/GPTQ-lite is the strongest pure training/export family,
- throughput-sensitive ideas such as naive recurrence or SwiGLU can lose if they slow steps too much.

That makes MTP a good next bet for this repository: it adds extra supervision during training, but the extra heads are excluded from export so they do not consume artifact budget.

## Prior repo runs that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Final stack direction:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **11-layer / XSA / EMA / partial-RoPE evolution:** `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` and `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Quantization-aware capacity growth:** `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/` and `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`
- **Dead-end warning for depth reuse:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research

- Fabian Gloeckle et al., **[Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)**. The paper shows that predicting multiple future tokens from a shared trunk can improve sample efficiency and downstream quality.

This candidate keeps the core idea but adapts it to the Parameter Golf regime by making the auxiliary loss sparse instead of supervising every position.

## What changed versus the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Enabled MTP by default** with `MTP_NUM_HEADS=2`.
2. Added **tail-focused sparse supervision** via:
   - `MTP_WINDOW_TOKENS=256`
   - `MTP_STRIDE=2`
3. Logged the new MTP settings explicitly at startup.
4. Kept the existing export behavior that **drops `mtp_heads` from the saved/exported state dict**.

Everything else stays intentionally close to the strong 11-layer base:

- 11 layers, 512 model dim, 8 heads / 4 KV heads
- MLP 3x
- BigramHash + SmearGate
- XSA on the last 4 layers
- partial RoPE + LN scale
- VE128 on layers 9 and 10
- EMA + tight SWA collection
- GPTQ-lite mixed int6 export
- sliding-window evaluation

## How to run

From this candidate directory:

```bash
cd candidates/202604040815_sparse-mtp-aux

RUN_ID=sparse_mtp_aux \
SEED=1337 \
NUM_LAYERS=11 \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.2 \
MTP_WINDOW_TOKENS=256 \
MTP_STRIDE=2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate inherits the same dataset/tokenizer defaults and export path style as the chosen base implementation.

## Main risks and tradeoffs

- The added softmax heads may still slow steps enough to erase any sample-efficiency gain.
- Tail-only supervision may leave performance on the table if full-sequence MTP is needed.
- MTP gains in the literature are strongest on larger models; this tiny-model regime may realize only a small effect.
- Because the export path excludes `mtp_heads`, any gain must transfer through the shared trunk rather than through the auxiliary heads directly.

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604040815_sparse-mtp-aux/train_gpt.py` | Passed |
| CPU import/forward smoke test with a stubbed FlashAttention module | Not feasible in this workflow container because `torch` is not installed, so a real module import and forward pass would require bootstrapping the full ML runtime first |
