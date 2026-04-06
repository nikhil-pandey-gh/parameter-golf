# Selective Ternary MLP on the LeakyReLU² + Legal TTT Stack

## Hypothesis

The current best stack is already very strong on training dynamics and evaluation, but it is still artifact-budget constrained. The hypothesis here is that **middle-layer MLP banks can be pushed to ternary {-1, 0, +1} late in training and at export time** with less quality loss than attention or embedding weights, freeing enough bytes to **reinvest in a larger BigramHash table**.

In short: **spend the low-bit budget where prior records say the model is least sensitive, then spend the saved bytes where prior records already showed a gain**.

## Why this is promising for this repository

Three repository patterns point in the same direction:

1. **Quantization/export quality has repeatedly moved BPB**. Mixed int6/int8 export, FP16 embeddings, GPTQ-lite percentile search, and EMA/SWA all improved leaderboard entries by shrinking the quantization gap.
2. **MLP weights are the most compressible large tensors in this codebase**. The `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` record showed that pushing MLP weights lower than attention funded extra capacity while still improving score.
3. **Bigger hashed bigram tables have already helped**. The best 10L mixed-int5 record grew BigramHash aggressively, and the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` README records a positive `2048 -> 3072` bigram ablation.

That makes selective ternary MLP export a good fit: it directly targets the repo's most expensive-but-most-compressible tensors, while keeping the winning 11-layer/XSA/Partial-RoPE/TTT recipe intact.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - Best current mean score.
  - Carries the strongest known stack: LeakyReLU(0.5)^2, legal score-first TTT, parameter banking + parallel Muon, XSA, Partial RoPE, LN scale, VE, EMA/SWA, GPTQ-lite export.
- **Quantization baseline:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - Shows that export-time quantization improvements still matter on the best 11-layer stack.
- **Low-bit MLP precedent:** `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
  - Strong evidence that MLP weights tolerate more aggressive quantization than attention.
- **Bigram capacity precedent:** `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`
  - Shows Bigger BigramHash is a real lever, not just a byte sink.
- **Dead-end avoided:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`
  - Documents that naive layer recurrence/looping was a bad fit under the fixed wall-clock budget, so this candidate stays on the proven feed-forward depth schedule.

## External research that informed it

- **TernaryLM** ([arXiv:2602.07374](https://arxiv.org/abs/2602.07374))
  - Motivating paper for native ternary training with adaptive scaling.
  - Especially relevant here because it reports that **middle transformer layers are the sparsest / most ternary-tolerant layers**, which maps cleanly onto selective middle-layer MLP quantization.
- **Tiny Autoregressive Recursive Models** ([arXiv:2603.08082](https://arxiv.org/abs/2603.08082))
  - Reviewed as an alternative direction.
  - It explicitly cautions that the full autoregressive recursive/TRM path did **not** show reliable gains, which lines up with this repo's own negative recurrence results.
- **MoEUT: Mixture-of-Experts Universal Transformers** ([arXiv:2405.16039](https://arxiv.org/abs/2405.16039))
  - Also reviewed as a parameter-sharing route.
  - Useful background for why shared-depth ideas remain interesting in general, but still less attractive here than low-bit compression because this repo is more artifact-bound than parameter-count-bound.

## What changed versus the chosen base

This candidate copies `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` and makes four focused changes:

1. **Selective ternary MLP path**
   - Adds late-stage fake quantization for the middle MLP banks via `TERNARY_MLP_ENABLED`, `TERNARY_MLP_LAYERS`, and `TERNARY_QAT_THRESHOLD`.
   - Default selected layers are `3,4,5,6,7` in the 11-layer stack.
2. **Mixed export now supports ternary MLP rows**
   - Export keeps the existing GPTQ-lite int6/int8 path for attention/embeddings/control tensors.
   - Selected middle-layer MLP matrices are quantized with a small adaptive ternary scale search and stored alongside the existing mixed-precision tensors.
3. **Byte reinvestment**
   - Default `BIGRAM_VOCAB_SIZE` is raised to **3072** to spend the expected compression savings on a lever that already showed positive ablations in this repository.
4. **Self-contained usability improvements**
   - Default data/tokenizer paths resolve from the repository root, so `train_gpt.py` can be run from inside this candidate directory.
   - FlashAttention import is optional, with a PyTorch SDPA fallback for non-Hopper / smoke-test environments.
   - Adds a `SMOKE_TEST=1` path that instantiates a reduced CPU model, runs forward/backward, and checks quantize/dequantize round-trip startup without dataset access.

## How to run or evaluate it

From the repository root:

```bash
torchrun --standalone --nproc_per_node=8 \
  candidates/202604061121_selective-ternary-mlp/train_gpt.py
```

From inside the candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
BIGRAM_VOCAB_SIZE=3072
TERNARY_MLP_ENABLED=1
TERNARY_MLP_LAYERS=3,4,5,6,7
TERNARY_QAT_THRESHOLD=0.20
LATE_QAT_THRESHOLD=0.15
```

Quick startup smoke check (from inside this candidate directory):

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation run in this workflow

1. `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604061121_selective-ternary-mlp/train_gpt.py`
   - **Passed**
2. `SMOKE_TEST=1 python candidates/202604061121_selective-ternary-mlp/train_gpt.py`
   - **Passed**
   - Output: `smoke_test:ok layers:3 dim:128 loss:6.9259 roundtrip_loss:6.9349`

## Main risks and tradeoffs

- **Ternary may still be too aggressive**, especially for `mlp_down_bank`, even in middle layers.
- **Compression wins are model-distribution dependent**: the extra BigramHash capacity only pays off if ternary export really saves enough bytes in practice.
- **Late-QAT thresholds may need tuning**: too early risks destabilizing training; too late risks not learning enough low-bit robustness.
- **This is still a candidate, not a verified record**: the design is grounded in repository evidence and recent literature, but it still needs real 8xH100 training runs to confirm the BPB / artifact tradeoff.
