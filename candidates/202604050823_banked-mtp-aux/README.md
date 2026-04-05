# Banked MTP auxiliary heads on the LeakyReLU/XSA stack

## Hypothesis

Adding a small **training-only multi-token prediction (MTP)** objective to the strongest banked 11-layer stack should improve sample efficiency and trunk representations without increasing the final submission artifact size, because the extra MTP heads are excluded from export.

## Why this is promising here

- The repository's best runs already look saturated on the same core recipe: 11 layers, 3x MLP, seq2048, mixed low-bit export, deep-layer XSA, partial RoPE, EMA/SWA, and increasingly careful eval/export tricks.
- Repo evidence argues against a heavier recurrence rewrite under the 10-minute wallclock cap: a non-record recurrence sweep regressed badly once it traded away too many optimizer steps.
- This codebase already carried dormant MTP support in the stronger 11-layer scripts, so MTP is a realistic low-churn direction rather than a broad infrastructure fork.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - strongest overall stack,
   - LeakyReLU(0.5)^2,
   - parameter banking + parallel Muon,
   - legal score-first TTT.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - strongest clean non-TTT stack,
   - GPTQ-lite clip search,
   - same mature 11-layer family with dormant MTP support.
3. `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
   - useful negative result showing naive layer recurrence hurt badly under a fixed wallclock budget.

## External research that informed it

1. Fabian Gloeckle et al., **Better & Faster Large Language Models via Multi-token Prediction** (arXiv:2404.19737)  
   https://arxiv.org/abs/2404.19737
2. Guoliang Zhao et al., **Self-Distillation for Multi-Token Prediction** (arXiv:2603.23911)  
   https://arxiv.org/abs/2603.23911
3. Lorenzo Noci et al., **Thinking into the Future: Latent Lookahead Training for Transformers** (arXiv:2603.20219)  
   https://arxiv.org/abs/2603.20219

The common thread is that future-token auxiliary supervision is still an active and credible way to buy more effective compute without paying permanent artifact bytes.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

- turn on conservative MTP defaults:
  - `MTP_NUM_HEADS=2`
  - `MTP_LOSS_WEIGHT=0.1`
- fix the banked optimizer path so **MTP head weights are actually optimized**;
  the source record comments that train-only small matrices should go through Adam, but the MTP heads were not wired into any optimizer group;
- keep MTP heads **train-time only** by continuing to drop them from the exported state dict before quantization and eval.

Everything else stays intentionally close to the strongest banked/leaky/XSA/partial-RoPE setup so the candidate isolates the MTP hypothesis.

## How to run

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.1 \
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

If you want to compare against the non-MTP banked baseline, rerun with `MTP_NUM_HEADS=0`.

## Expected risks and tradeoffs

- Extra train-time softmax heads may reduce total steps completed in 600s.
- Auxiliary future-token loss can hurt plain next-token quality if the weight is too high.
- The candidate keeps the banked/TTT-heavy code path, so debugging remains more complex than the cleaner non-TTT 2026-03-22 stack.

## Validation

- `python -m compileall candidates/202604050823_banked-mtp-aux/train_gpt.py` -> **passed**
- dependency probe:
  - `torch_available=False`
  - `flash_attn_interface_available=False`
- A CPU-only runtime smoke test was **not feasible** in this environment because the runner lacks the required training dependencies (`torch`, `flash_attn_interface`), and the real training/eval path also hard-requires CUDA.
