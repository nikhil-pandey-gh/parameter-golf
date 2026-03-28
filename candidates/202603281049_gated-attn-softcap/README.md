# Gated Attention + Attention Softcap on the LeakyReLU² TTT Stack

## Hypothesis

The current leaderboard is already strong on depth, quantization, EMA/SWA, and evaluation tricks, but it still relies on aggressively quantizing attention-heavy weights down to int6. My hypothesis is that **attention outlier control** is the cleanest next lever: enable the dormant per-head attention gate in the current best stack and add **Gemma-style attention-logit soft-capping** so the model learns less extreme attention scores before GPTQ-lite export.

If this works as in the compact-model literature, it should reduce quantization sensitivity in the attention path while preserving the rest of the winning stack unchanged.

## Why this is promising for this repository

Repository history suggests three important things:

1. The strongest recent gains come from stacking many small wins on the 11-layer/XSA/partial-RoPE/EMA/GPTQ-lite family rather than from wholesale architecture swaps.
2. This challenge is unusually sensitive to post-training quantization quality; several records improved mostly by reducing export damage rather than by changing floating-point loss.
3. Full layer recurrence was already a dead end here under fixed wallclock, so the next idea should avoid extra serial depth and focus on **quality-per-parameter and quantization robustness** instead.

This candidate follows that evidence closely: it keeps the current best end-to-end stack, then only changes the attention path in a way that is directly motivated by prior quantization pain.

## Prior records and candidates that influenced this

There was no existing `candidates/` directory when this candidate was created.

The main repository influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - current best overall stack (`1.1194` mean) and the direct implementation base for this candidate.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest pre-TTT export-aware stack and the clearest evidence that better post-training quantization still matters.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - established partial RoPE + layerwise scaling as durable improvements on the same 11-layer family.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful negative result showing layer recurrence/depth reuse hurt under fixed wallclock, which is why this candidate does not spend extra serial compute.

## External research that informed this

- **Gemma 2 technical report** (`arXiv:2408.00118`): Gemma 2 explicitly soft-caps logits in *each attention layer* and the final layer, using softcap `50.0` for attention logits and `30.0` for final logits. That is a very close architectural match to this repo, which already uses final-logit softcapping but not attention-logit softcapping.
- **Bondarenko et al., "Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing"** (`arXiv:2306.12929`): this paper directly links transformer outliers to attention behavior and proposes **clipped softmax** and **gated attention** as simple fixes that improve quantization-friendliness while maintaining or improving floating-point quality.

Together, these papers suggest that attention outlier control is especially relevant for a tiny model that must survive a harsh int6 roundtrip.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added a new hyperparameter, `ATTN_LOGIT_SOFTCAP` (default `50.0`).
2. Applied soft-capping to **attention logits**, not just the final LM logits.
3. Turned `GATED_ATTENTION` on by default for this candidate (`1` instead of `0`).
4. Added a dense causal-attention fallback path for non-CUDA/local smoke scenarios so the module can still be imported and sanity-checked without FlashAttention installed.

Everything else stays intentionally close to the top record: LeakyReLU(0.5)^2 MLPs, parameter banking + Parallel Muon, XSA on late layers, partial RoPE, VE, EMA/SWA, GPTQ-lite export, and the legal score-first TTT path are all preserved.

## How to run or evaluate it

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
GATED_ATTENTION=1 ATTN_LOGIT_SOFTCAP=50 LOGIT_SOFTCAP=30 \
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

For a tiny local import/forward smoke once PyTorch is available, you can import the module and instantiate a very small `GPT(...)` directly; the dense attention fallback is intended for that path, not for benchmark-speed training.

## Validation commands and outcomes

Validation run in this workflow:

- `python -m compileall candidates/202603281049_gated-attn-softcap/train_gpt.py`
  - **Passed**.

Attempted smoke validation:

- Tried importing the candidate module and running a tiny CPU forward pass.
  - **Blocked by environment**: the workflow Python environment here does not have `torch` installed, so a real runtime smoke test was not feasible.

## Main expected risks or tradeoffs

- **Over-regularization risk**: attention softcapping may blunt sharp retrieval/copying behavior if `50.0` is too low for this tiny model.
- **Interaction risk with TTT**: the post-training adaptation path may want sharper attention than the pre-TTT model, so the net effect on post-TTT BPB is uncertain.
- **Kernel support risk**: CUDA runs require a FlashAttention build whose `flash_attn_func` accepts the `softcap=` keyword. The dense fallback is intentionally limited to non-CUDA/local smoke scenarios and is not intended for leaderboard-speed runs.
