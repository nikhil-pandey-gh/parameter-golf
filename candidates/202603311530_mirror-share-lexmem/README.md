# Mirror-Shared Lexical Memory + Light MTP

## Hypothesis

The current repo is already very strong at squeezing extra score out of evaluation and quantization. The next useful training-side move is to **stop paying artifact bytes for redundant core weights**, then reinvest that headroom into two things that matter more for this challenge:

1. **Cheap lexical memory** that barely affects step time but gives the tiny model more token/bigram-specific capacity.
2. **Higher-value training signals** via a small **2-token multi-token prediction (MTP)** auxiliary loss whose heads are dropped before export.

Concretely, this candidate mirror-shares the banked transformer weights across symmetric layers, keeps per-layer control tensors unique, expands lexical tables, and uses the saved bytes to export lexical tables in fp16 while only pushing the MLP banks down to int6.

## Why this is promising for this repository

The record history strongly suggests three things:

- Most large gains came from **better use of the 16MB artifact budget**, especially mixed precision export, wider or deeper models funded by compression, and cheap lexical/context tricks like `BigramHash` and `ValueEmbedding`.
- **Naive recurrence/looping** is not enough on its own under a 10-minute cap, because extra compute costs too many optimizer steps.
- The repo has not really explored **artifact-aware parameter sharing** as a way to keep step time similar while moving bytes toward more helpful parameters and gentler export precision.

This candidate therefore uses **sharing for byte reallocation, not for extra loops**. The logical depth stays at 11 layers, but the banked attention/MLP weights are mirrored across symmetric layers. That keeps the compute profile close to the current banked stack while shrinking the unique heavy tensors that dominate the artifact.

## Prior records and experiments that influenced this candidate

This candidate is based on the latest banked training stack from:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

The following records most directly informed the design:

- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: best current stack, plus banked weights and LeakyReLU(0.5)^2.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: evidence that small export/averaging improvements still matter at the margin.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: strong zero/low-parameter architectural refinements on the 11-layer stack.
- `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`: explicit proof that better compression can be traded for more useful capacity, and that larger `BigramHash` tables can help.
- `track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`: cautionary negative result showing that naive layer recurrence can lose badly if it costs too many steps.

## External research that informed the candidate

Three papers were most relevant here:

- **Multi-Token Prediction** (`arXiv:2404.19737`): shows that predicting multiple future tokens from a shared trunk can improve sample efficiency, which is exactly the bottleneck in this 10-minute regime.
- **ALBERT** (`arXiv:1909.11942`): classic evidence that cross-layer parameter sharing can cut parameters substantially without requiring a totally different training stack.
- **Looped Transformers** (`arXiv:2409.15647`): useful as a reminder that shared/reused transformer computation can learn meaningful iterative structure, but this repo's own negative result suggests using sharing primarily for artifact efficiency instead of extra compute depth.

## What changed versus the chosen base implementation

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes the following changes:

- **Mirror-shared parameter banks**
  - Added `SHARED_BANK_MODE` with default `mirror`.
  - Logical layers still run at depth 11, but heavy banked weights are shared across symmetric layers via a bank-layer map.
  - Per-layer control tensors (`attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, skip weights, etc.) remain unique.

- **Artifact-preserving export for shared banks**
  - The export path now unbanks only the **unique bank tensors**, so sharing remains real in the serialized artifact instead of being accidentally duplicated during quantization.

- **Reinvested lexical memory**
  - `BIGRAM_VOCAB_SIZE` default increased to `8192`.
  - `VE_DIM` default increased to `160`.
  - `VE_LAYERS` default expanded to `8,9,10`.

- **More selective precision allocation**
  - Added `INT6_CATS`, defaulting to `mlp`, so MLP banks stay aggressively compressed while attention weights fall back to int8 export.
  - Added `FP16_EMBED_EXPORT=1` by default so lexical embedding tables can stay in fp16 when exporting.

- **Light MTP by default during training**
  - Enabled `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15` by default.
  - The auxiliary heads are already excluded from export, so this is a training-only efficiency bet rather than a permanent artifact cost.

- **Flash-attention fallback for local sanity checks**
  - If `flash_attn_interface` is unavailable, the attention path falls back to `torch.nn.functional.scaled_dot_product_attention`.
  - This is mostly to make light local validation easier; the intended leaderboard path is still the Hopper/FlashAttention stack.

## How to run or evaluate it

From inside this candidate directory:

```bash
RUN_ID=mirror_share_lexmem \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate bakes in its main defaults, but the following knobs are intended to be easy to sweep:

```bash
SHARED_BANK_MODE=mirror \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
BIGRAM_VOCAB_SIZE=8192 \
VE_DIM=160 \
VE_LAYERS=8,9,10 \
INT6_CATS=mlp \
FP16_EMBED_EXPORT=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to compare against the non-shared ablation while keeping the rest of the stack intact, set:

```bash
SHARED_BANK_MODE=none
```

## Validation

I ran the lightweight checks that fit this repository and environment:

- `python -m compileall train_gpt.py train_gpt_mlx.py data`
  - **Passed**.
- `python -m compileall candidates/202603311530_mirror-share-lexmem/train_gpt.py`
  - **Passed**.
- Attempted a CPU import/forward smoke test by importing this candidate module and instantiating a tiny `GPT` on random inputs.
  - **Blocked by environment**: this runner does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`), and the cached dataset/tokenizer artifacts are also absent locally.
  - I therefore did not attempt a heavier dependency install just for smoke validation.

## Main expected risks and tradeoffs

- **Mirror sharing may overconstrain the trunk.** Keeping per-layer control tensors unique should help, but the shared heavy matrices may still remove too much layer-specific specialization.
- **MTP may spend compute in the wrong place.** The auxiliary heads are export-free, but if the extra loss slows step time too much, it could erase the sample-efficiency benefit.
- **Lexical-memory expansion could be the wrong reallocation target.** The repo has strong evidence that `BigramHash` and related tables help, but the best bucket count under this specific shared-bank regime is still unknown.
- **Precision allocation needs real artifact measurements.** This design intentionally spends saved bytes on fp16 lexical tables and int8 attention, but the best final balance may differ once full 8xH100 logs and serialized sizes are available.
- **No GPU run was performed in this environment.** This candidate is meant to be a strong, research-grounded next iteration, but it still needs real wallclock/size evaluation on the intended hardware.
