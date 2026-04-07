# Scheduled single-head MTP on the Legal-TTT record base

## Hypothesis

A **single auxiliary multi-token-prediction head** should improve sample efficiency during the fixed 600s training budget by giving each hidden state one extra future-token target. Fading that auxiliary loss during warmdown should keep the final checkpoint aligned with the repository's real objective: **next-token BPB after quantization, sliding-window eval, and legal score-first TTT**.

## Why this is promising here

- The current best line already squeezed large gains out of evaluation, quantization, activation, and optimizer/layout changes; a **training-only auxiliary objective** is one of the strongest remaining low-infrastructure levers.
- This repo's strongest record family already contains dormant MTP plumbing and strips `mtp_heads` from export, so the idea fits the codebase without changing the final inference path or artifact size.
- Unlike the non-record **layer recurrence** experiment, which lost badly because it reduced useful training steps, this candidate keeps the same deployed architecture and spends extra capacity only during training.

## Prior experiments that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` (`val_bpb: 1.1194`), which is the current best record and already combines LeakyReLU^2, Legal TTT, Parameter Banking, Parallel Muon, EMA, XSA, partial RoPE, VE, and tight export logic.
- **MTP hook provenance:** `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`, whose script already had MTP modules plus export stripping, but the recorded runs still logged `mtp_num_heads:0`.
- **Recent stack refinements:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which showed the value of staying close to the winning 11-layer EMA/XSA family and improving training/export details instead of replacing the architecture.
- **Dead-end contrast:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`, whose layer-recurrence attempt regressed badly, reinforcing that the next idea should avoid slower or more invasive depth-reuse schemes.

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

- Fabian Gloeckle et al., **“Better & Faster Large Language Models via Multi-token Prediction”** — arXiv:2404.19737. Core motivation: auxiliary future-token heads improve sample efficiency and representation quality during pretraining.
- DeepSeek-AI et al., **“DeepSeek-V3 Technical Report”** — arXiv:2412.19437. Practical evidence that MTP remained attractive in a modern strong-pretraining stack.
- Anastasios Gerontopoulos et al., **“Multi-Token Prediction Needs Registers”** — arXiv:2505.10518. Reinforced the idea that the extra prediction mechanism should stay lightweight.
- Somesh Mehra et al., **“On multi-token prediction for efficient LLM inference”** — arXiv:2502.09419. Important caution: MTP is not automatically helpful, especially post hoc, so this candidate keeps it **from scratch** and anneals it late rather than bolting it on after training.

The strongest alternative from the research pass was **activation-aware PTQ** (AWQ / SmoothQuant / OmniQuant style), but I deferred that to a future candidate because it needs a broader calibration/export rewrite than this MTP-first pass.

## What changed vs the chosen base

1. **Enabled actual MTP training by default** with `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.15`.
2. Added a **compile-safe runtime tensor buffer** for the live MTP loss weight, so the training loop can adjust it without relying on Python attributes that may be constant-folded by `torch.compile`.
3. Added **scheduled fade-out** via `MTP_FADE_SCALE=0.3`: once the LR scale drops below 0.3 in warmdown, the auxiliary MTP weight linearly decays toward zero.
4. Added a **real optimizer path for `mtp_heads`**. The copied record script still instantiated MTP modules and excluded them from export, but its active optimizer split did not step those parameters.
5. Added **training-log visibility** for the effective MTP weight.

The exported artifact still excludes `mtp_heads`, so the candidate preserves the deployed architecture used for quantized eval and Legal TTT.

## How to run

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 MTP_FADE_SCALE=0.3 \
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

## Validation

| Command | Outcome |
| --- | --- |
| `python -m compileall candidates/202604070830_scheduled-mtp/train_gpt.py` | Passed |
| Minimal CPU-only smoke test | Not feasible in this environment: this trainer assumes CUDA execution, FlashAttention 3, real FineWeb shards, and a real SentencePiece tokenizer. |

## Main risks and tradeoffs

- The extra head increases training FLOPs and may reduce the total number of 600s steps.
- If MTP is too strong at this model size, it may help pretraining loss while hurting the final next-token/quantized/TTT metric.
- The new fade schedule anneals the **loss weight**, not the compiled forward graph itself, so late training still pays the extra head's compute.
- Because no prior repo run actually used nonzero MTP, the main uncertainty is empirical: this is a high-upside but not yet repo-validated direction.
