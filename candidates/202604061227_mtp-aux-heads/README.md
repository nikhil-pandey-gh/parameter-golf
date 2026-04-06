# Export-Free Multi-Token Prediction on the 11L Frontier

## Hypothesis

The current frontier has already harvested most of the obvious export and evaluation gains, so the next clean win is to improve **training sample efficiency** without paying extra artifact bytes. This candidate adds **multi-token prediction (MTP)** auxiliary heads to the latest 11-layer LeakyReLU² + legal-TTT backbone, with the key property that the auxiliary heads are **trained but excluded from export**, so the 16MB submission budget is unchanged.

## Why this is promising here

- The repository's best runs are now tightly clustered around the same 11-layer compressed backbone, and recent gains are measured in low thousandths of BPB.
- The 10-minute cap makes **learning efficiency per optimizer step** unusually valuable.
- This codebase already had an MTP path, but the auxiliary heads were never actually placed in an optimizer group, so enabling MTP did not yet correspond to a real training experiment.

Primary external research that motivated this candidate:

1. **Fabian Gloeckle et al., _Better & Faster Large Language Models via Multi-Token Prediction_**, arXiv:2404.19737 — argues that predicting multiple future tokens from a shared trunk improves sample efficiency and especially helps induction-style behavior.
2. **DeepSeek-AI, _DeepSeek-V3 Technical Report_**, arXiv:2412.19437 — uses a multi-token prediction objective in a modern production-scale training stack as a performance-improving objective rather than only an inference trick.

## Prior repository work that influenced it

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` provided the strongest current backbone: 11 layers, LeakyReLU(0.5)^2, partial RoPE, XSA, VE, EMA/SWA, GPTQ-lite int6 export, legal TTT, and parameter banking.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` showed that the current frontier is increasingly limited by small export/eval refinements, making a training-side objective upgrade more attractive.
- Earlier records established the core trends this candidate keeps intact: sliding-window eval, partial RoPE, bigram/smear features, and aggressive quantization-aware compression.

## What changed vs the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **MTP enabled by default** with `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15`.
2. Added **`MTP_HEAD_LR`** so the auxiliary heads can be tuned independently from the trunk.
3. **Actually optimize `mtp_heads`** with AdamW and include their gradients in the manual replicated-parameter all-reduce path.
4. Preserve the existing export behavior that **drops `mtp_heads` from `export_sd`**, keeping the artifact budget impact at zero.
5. Make the script runnable from the candidate directory by resolving default dataset/tokenizer paths relative to the repository root instead of the current working directory.

## How to run

From this candidate directory:

```bash
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_HEAD_LR=0.008 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
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

For a training-only ablation, keep the same command but set `TTT_ENABLED=0`.

## How to evaluate

The script keeps the same evaluation/export flow as the base record:

- int6 GPTQ-lite + lzma export
- sliding-window evaluation
- optional legal score-first TTT when `TTT_ENABLED=1`

## Expected risks and tradeoffs

- **Training overhead:** extra vocab projections and cross-entropy terms can reduce steps completed in 600s if the auxiliary loss is too heavy.
- **Tiny-model mismatch:** Gloeckle et al. report stronger gains at larger scales, so the optimal setting here may be only 1-2 heads and a modest loss weight.
- **Interaction with TTT:** if MTP already sharpens local continuation behavior, some of the gain may overlap with legal TTT instead of stacking cleanly.
- **Hyperparameter sensitivity:** the auxiliary heads need enough LR to learn quickly, but too much can over-regularize the trunk through the shared hidden states.

## Validation run in this workflow

Commands run in this workflow:

1. `python -m compileall train_gpt.py`
2. `python - <<'PY' ... importlib.util.find_spec(...) ... PY` to confirm whether `torch`, `sentencepiece`, and `numpy` were available for a local smoke test.

Outcome:

- `compileall` completed successfully.
- A real CPU smoke test was **not feasible on this runner** because the required runtime dependencies were absent (`torch`, `sentencepiece`, and `numpy` were all missing), so this workflow could only perform syntax-level validation here.
