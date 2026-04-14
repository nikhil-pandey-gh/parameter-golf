# Bank-aware mixed-bit GPTQ-lite on the banked 11L stack

## Hypothesis

The current record stack already has the strongest training-time system improvements in the repo: 11 layers, LeakyReLU^2, XSA, partial RoPE, VE, and Parallel Muon with parameter banks. The missing piece is that the newer banked export path no longer gets the same low-bit treatment as the earlier GPTQ-lite and mixed-bit records. This candidate restores that by applying **bank-aware row-wise clip-search quantization** directly to the 3D parameter banks, using **int5 for MLP banks** and **int6 for attention banks**.

## Why this is promising here

- Repo history says export quality is still a major lever: FP16 embeddings, mixed int5/int6, and GPTQ-lite clip search all produced real gains.
- The strongest banked run improved training throughput, but its export path is a looser fit than the older quantization-focused records. That makes export the cleanest place to recover quality without paying more training-time overhead.
- Mixed-bit allocation already worked in the repo: int5 MLP + int6 attention funded a deeper model in the 10-layer record, while GPTQ-lite clip search improved the best non-TTT 11-layer stack.
- This candidate keeps the newest banked training recipe intact and only changes the artifact formation path plus one small budget reallocation (`BIGRAM_VOCAB_SIZE=3072` by default).

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - strongest banked training stack
   - Parallel Muon + parameter banks
   - LeakyReLU^2, XSA4, partial RoPE, VE
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - GPTQ-lite percentile search
   - EMA + warmdown3500 on the 11-layer frontier
3. `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
   - mixed int5/int6 export
   - larger bigram hash funded by export savings
4. `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
   - explicit evidence that quantization sensitivity can dominate small-model quality

## External research that informed it

- **GPTQ**: one-shot weight quantization works best when quantization is tailored to each matrix instead of using one coarse global scale. https://arxiv.org/abs/2210.17323
- **AWQ**: low-bit quality depends on protecting the right channels/rows rather than treating all weights equally. https://arxiv.org/abs/2306.00978
- **SmoothQuant**: equivalent rescaling can materially reduce quantization error without changing the forward function. https://arxiv.org/abs/2211.10438
- **A Survey on Transformer Compression**: parameter-efficient architecture work and smarter quantization remain the two most reliable compression levers for Transformers. https://arxiv.org/abs/2402.05964

This candidate stays on the conservative side of that literature: no calibration set, no new kernels, no activation rewriting, just a bank-aware version of the repo's existing GPTQ-lite / mixed-bit logic.

## What changed vs the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

1. Added a safe FlashAttention fallback based on `scaled_dot_product_attention` so the candidate has a CPU-friendly smoke path when PyTorch is available.
2. Added `SMOKE_TEST=1`, which instantiates a tiny CPU model, runs one forward pass, and verifies low-bit round-trip dequantization shapes.
3. Changed bank parameter classification so `qo_bank` / `kv_bank` are treated as attention tensors and `mlp_up_bank` / `mlp_down_bank` as MLP tensors during export.
4. Replaced the old 2D-only GPTQ-lite helper with a version that also handles 3D bank tensors by flattening them into row groups and doing percentile search per row.
5. Switched bank export policy to **mixed-bit**:
   - attention banks: int6
   - MLP banks: int5
   - small / control tensors: passthrough
   - everything else: existing int8 fallback
6. Bumped the default `BIGRAM_VOCAB_SIZE` to `3072`, reusing a bit of the saved artifact budget in the direction the latest record's ablation already suggested.

## How to run or evaluate it

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to compare against the absolute frontier, you can still layer the repo's legal TTT flags on top of this export variant.

## Main expected risks or tradeoffs

- Int5 MLP banks may be too aggressive on the banked 11-layer stack and could erase part of the gain from finer row-wise scaling.
- Percentile search over large bank tensors increases export-time work, even though it does not slow training.
- The extra bigram budget may help less than expected if the export savings are already consumed by control tensors and embeddings.
- The best overall leaderboard scores are increasingly evaluation-aware, so an export-only win may still trail TTT-heavy runs.

## Validation

### Commands

```bash
python -m compileall candidates/202604140837_banked-mixedbit-gptq/train_gpt.py
SMOKE_TEST=1 python candidates/202604140837_banked-mixedbit-gptq/train_gpt.py
```

### Outcome

- `python -m compileall ...` passed in this workflow environment.
- A true CPU smoke run was **not completed here** because the workflow runner did not have PyTorch installed, and installing a CPU torch wheel was blocked by the runner's package proxy. The candidate script does include a `SMOKE_TEST=1` path plus a FlashAttention fallback so the smoke test can run on a machine with the repo's normal PyTorch dependency available.
