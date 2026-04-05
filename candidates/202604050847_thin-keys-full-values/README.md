# Thin Keys, Full Values

## Hypothesis

The current 11-layer static stack is probably overpaying for **selection bandwidth** in attention. If queries and keys only need enough dimensions to choose relevant tokens, while values still need the full head width to carry content, then shrinking **Q/K head dim** from 64 to 16 should free parameters and reduce attention FLOPs with much less quality loss than shrinking values.

## Why it is promising here

Recent records have already pushed the main local knobs hard: XSA, EMA/SWA, partial RoPE, LN scaling, GPTQ-lite clip search, late-stage quantization tuning, and eval-time tricks. The strongest static base is still the 11L GPTQ-lite + EMA stack from `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, so a fresh gain is more likely to come from an underexplored structural change than from another small schedule tweak.

This candidate targets exactly that gap: it keeps the mature 11L recipe intact and changes only the attention projection geometry.

## Prior repo influence

- **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Supporting static stack lineage:** `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` and `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Why not fork 2026-03-23 directly:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` adds TTT and optimizer/systems complexity. This candidate instead probes a new attention parameterization on the strongest simpler base.

There were no prior experiments under `candidates/` when this candidate was created.

## External research

- **Thin Keys, Full Values** argues that selection needs far fewer dimensions than value transfer, and reports that training from scratch with quarter-size keys can match full-attention perplexity while using fewer parameters and less compute: <https://arxiv.org/abs/2603.04427>
- **Grouped-Query Attention** is already validated in this repo at 4 KV heads; the paper shows that reducing K/V-side capacity can preserve quality surprisingly well, which supports going one step further on the Q/K side: <https://arxiv.org/abs/2305.13245>
- **MHA2MLA** further supports the broader premise that K/V-side compression composes with existing architectures and compression schemes, though this candidate stays simpler than MLA: <https://arxiv.org/abs/2502.14837>

## What changed vs the chosen base

Starting from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. adds a new `QK_HEAD_DIM` hyperparameter, defaulting to `16`,
2. changes attention so `q` and `k` project to `num_heads * QK_HEAD_DIM` and `num_kv_heads * QK_HEAD_DIM`,
3. keeps `v` and the attention output at the original full 64-dim head width,
4. applies RoPE over the thinner Q/K representation,
5. falls back to PyTorch `scaled_dot_product_attention` when Q/K and V head dims differ.

Everything else stays aligned with the strong 3/22 static stack: 11 layers, MLP 3x, SmearGate, BigramHash, XSA-on-last-4, partial RoPE, LN scale, EMA + tight SWA, GPTQ-lite int6 export, and shared value embeddings.

## How to run

From this directory:

```bash
QK_HEAD_DIM=16 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The most useful follow-up sweep is `QK_HEAD_DIM in {16, 24, 32}`.

## Validation

- `python -m compileall candidates/202604050847_thin-keys-full-values/train_gpt.py` — passed
- Minimal CPU smoke test was **not feasible in this environment** because the repository runtime dependencies needed to import the training script (`torch`, and typically `flash_attn_interface` on the record path) were not installed here. This candidate keeps the smoke-safe built-in SDPA fallback for the thin-QK path, but an import-time forward-pass smoke test still requires the repo runtime stack.

## Main risks and tradeoffs

- `QK_HEAD_DIM=16` may be too aggressive for this 11L stack; if selection capacity becomes the bottleneck, `24` may be the better setting.
- The candidate gives up the explicit FlashAttention-3 call on the default thin-QK path and relies on PyTorch SDPA instead; that should still be efficient, but exact H100 step-time needs measurement.
- Saved parameters are not yet reinvested into bigger lexical features or extra depth; if thin Q/K works, the next logical experiment is to spend part of that budget on a larger BigramHash table or another depth/width bump.
