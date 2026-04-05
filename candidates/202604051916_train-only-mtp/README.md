# Train-Only MTP on the LeakyReLU2 + Legal TTT Stack

## Hypothesis

Add a **single auxiliary multi-token prediction (MTP) head** during training, but **strip it from the exported artifact** before quantization and evaluation. The extra head should improve sample efficiency and future-token representations while keeping the shipped model size and evaluation path effectively unchanged.

## Why this is promising here

- The repo's strongest current stack is `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`, and its code already contains dormant MTP support plus logic to exclude `mtp_heads` from export.
- The main leaderboard gains have increasingly come from careful training/eval refinements rather than large architectural rewrites, so a **training-only regularizer with zero artifact cost** is a good fit for the remaining search space.
- Repository review found **no prior `candidates/` directory** and no prior record README that actually enabled MTP in a submitted configuration.

## Prior experiments that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - Best overall score in the repo review (`1.1194` post-TTT mean).
   - Provides the exact base implementation copied here.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - Best non-TTT stack and another reference point showing how tight the non-eval gains have become.
3. `records/track_10min_16mb/2026-03-17_LoRA_TTT/`
   - Useful evidence that evaluation-aware ideas can help, but this candidate intentionally targets **training quality** without adding more eval complexity.

## External research that informed it

- Fabian Gloeckle et al., **Better & Faster Large Language Models via Multi-token Prediction** — <https://arxiv.org/abs/2404.19737>
  - The core evidence for using future-token auxiliary heads as a sample-efficiency booster on top of a shared trunk.
- Tianle Cai et al., **Medusa** — <https://arxiv.org/abs/2401.10774>
  - Relevant supporting precedent for lightweight extra decoding heads on top of a shared backbone.
- Mitchell Stern et al., **Blockwise Parallel Decoding for Deep Autoregressive Models** — <https://arxiv.org/abs/1811.03115>
  - Earlier evidence that predicting multiple future tokens with auxiliary heads is a meaningful autoregressive direction.

I also considered more invasive ideas from recent compact-model literature, including selective parameter sharing (`ALBERT`, `Universal Transformer`, intra-layer recurrence) and stronger post-training rotations (`QuaRot`, `SpinQuant`), but those would require broader architectural or quantization changes than this repository currently needs for a clean next ablation.

## What changed versus the chosen base implementation

Relative to `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`:

1. `MTP_NUM_HEADS` now defaults to `1` instead of `0`.
2. `MTP_LOSS_WEIGHT` now defaults to `0.15` instead of `0.2` to keep the auxiliary objective conservative.
3. The MTP heads are now actually added to the head optimizer, fixing the previously dormant path so the auxiliary loss can train the auxiliary heads.
4. The existing export path that drops `mtp_heads` is kept intact, so the final quantized artifact and evaluation model remain the base model without auxiliary heads.

Everything else stays intentionally close to the March 23 record so this candidate isolates the MTP effect.

## How to run

From the repository root:

```bash
cd candidates/202604051916_train-only-mtp
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For an ablation against the copied base behavior, set `MTP_NUM_HEADS=0`.

## How to evaluate

The script keeps the base evaluation pipeline:

- EMA/SWA averaging
- export without `mtp_heads`
- int6 round-trip eval
- stride-64 sliding-window eval
- optional legal score-first TTT

The important check for this candidate is whether the MTP-enabled training run improves the final exported model's BPB without paying a persistent artifact-size penalty.

## Main risks and tradeoffs

- Even one auxiliary head adds some training-time FLOPs, so any quality gain has to beat the lost wall-clock steps.
- The true bottleneck may now be quantization or TTT rather than pre-export modeling quality, in which case MTP may move the pre-quant metric more than the final score.
- The repo had dormant MTP code but no proved winning setting yet, so this is still a first serious ablation rather than a known-good tweak.

## Validation

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604051916_train-only-mtp/train_gpt.py` - **passed**
- A CPU start-up smoke test was **not feasible** in this environment because the copied record stack hard-requires CUDA plus the FlashAttention bindings it uses at runtime.
