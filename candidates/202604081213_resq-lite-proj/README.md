# ResQ-lite projection rescue on top of the 2026-03-23 best stack

## Hypothesis

The current record line already squeezed a lot out of training-side changes, so the next cheap gain is likely in the export path. This candidate keeps the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` training and evaluation stack, but replaces uniform post-training int6 export with a tiny **byte-budgeted low-rank residual rescue** on the most quantization-sensitive projection matrices.

The working bet is:

- keep the strong 11L + LeakyReLU^2 + XSA + Partial RoPE + legal TTT base,
- quantize as before with GPTQ-lite-style per-row percentile search,
- then spend a very small extra byte budget on rank-1 residual factors for the projection matrices whose int6 reconstruction error is most expensive per stored parameter.

If that works, post-quant sliding-window BPB should improve without touching the 600 second training budget.

## Why this is promising for this repo

Repository evidence points to compression/export as a persistent bottleneck:

- `2026-03-18_FP16Embed_WD3600` showed that smarter export of tied embeddings materially shrank the post-quantization gap.
- `2026-03-19_MixedQuant_Int6Int8_SlidingWindow` and later int5/int6 records showed that selective precision reallocation buys real BPB gains.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved again with a pure export-time quantizer upgrade.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` is now strong enough that another full architecture rewrite looks riskier than a surgical export improvement.

The repo review also found **no prior `candidates/` directory**, so this candidate is not duplicating an earlier candidate branch.

## Prior records that most influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
   - chosen as the base because it is the current best record-track submission.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
   - showed that export-only quantization improvements are still worth chasing.
3. `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600`
   - clear evidence that a little extra precision in the right place can beat uniform low-bit export.
4. `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
   - reinforced the broader pattern that better byte allocation matters as much as raw model quality.

## External research that informed it

1. **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (Lin et al., 2024 / arXiv:2306.00978)
   - salient weights/channels matter disproportionately, so protecting a tiny subset can pay off.
2. **Channel-Wise Mixed-Precision Quantization for Large Language Models** (Chen et al., 2024 / arXiv:2410.13056)
   - channel-wise precision allocation can improve accuracy under a tight memory budget.
3. **ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals** (Saxena et al., 2024 / arXiv:2412.14363)
   - low-rank residuals are a byte-efficient way to rescue quantization-sensitive subspaces.
4. **Scaling Law for Quantization-Aware Training** (Chen et al., 2025 / arXiv:2505.14302)
   - highlights FC2/projection outliers as a key quantization bottleneck.
5. **Precision Where It Matters** (Maisonnave et al., 2025 / arXiv:2504.21553)
   - architecture-specific projection layers can concentrate the spikes worth protecting.

## What changed versus the chosen base implementation

Base file:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added a **FlashAttention fallback** to `scaled_dot_product_attention` so import-level CPU smoke tests work even when `flash_attn_interface` is unavailable.
2. Added `RESQ_*` hyperparameters:
   - `RESQ_ENABLED`
   - `RESQ_RANK`
   - `RESQ_BYTE_BUDGET`
   - `RESQ_TARGET_PATTERNS`
3. Extended the export path so that after GPTQ-lite int6 quantization:
   - candidate projection matrices are scored,
   - a truncated low-rank residual is computed,
   - matrices are ranked by residual energy captured per stored parameter,
   - only the best residual factors that fit the configured byte budget are kept.
4. Extended dequantization so those residual factors are added back before round-trip evaluation.
5. Added logging of which matrices received residual rescue and how much of the budget was used.

This is intentionally a **minimal export-path change**. Training, EMA/SWA, sliding-window eval, legal TTT, and the rest of the model stack are left intact.

## How to run

From the candidate directory:

```bash
cd candidates/202604081213_resq-lite-proj

DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
RESQ_ENABLED=1 RESQ_RANK=1 RESQ_BYTE_BUDGET=24576 \
RESQ_TARGET_PATTERNS=mlp.proj.weight,attn.proj.weight \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The candidate still prefers the FlashAttention path on real GPU runs.
- The residual rescue is export-only, so it should not change step time during training.
- EMA uses the baked-in `0.997` path from the base script; there is no separate `EMA_*` runtime flag in this file.
- Late QAT is triggered by `LATE_QAT_THRESHOLD`; there is no extra `LATE_QAT=1` switch in this candidate.
- The script accepts the older `RESQ_PARAM_BUDGET` env var as a compatibility alias, but the candidate now budgets **bytes**, not coefficient count.

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604081213_resq-lite-proj/train_gpt.py` | Passed. |
| CPU import/forward smoke in a temporary venv using `/tmp/gh-aw/agent/pg-venv/bin/python` | Passed; a tiny 2-layer model produced a finite loss and `logits_shape=(2, 16, 128)`. |
| Quantizer-branch smoke in the same venv with a toy banked model and randomized projection weights | Passed; `ResQ-lite` selected both toy `mlp.proj.weight` matrices under a `4096`-byte residual budget. |

I did **not** run a full `main()` training smoke here because the script hard-requires CUDA plus the repository tokenizer and shard dataset layout; the CPU-only validation above was the lowest-cost way to verify the new code paths without fabricating extra infrastructure.

## Main expected risks and tradeoffs

- The residual budget is tight by design; if it is too small, the rescue path may select nothing.
- If the budget is too large, the compressed artifact can drift past the 16MB target.
- Rank-1 residuals may be too weak if the dominant quantization error is not close to low-rank.
- Targeting `mlp.proj` and `attn.proj` is a research-informed heuristic, but the true worst tensors may differ on this exact stack.
- The FlashAttention fallback is for portability and smoke testing, not the expected fast path for leaderboard runs.
