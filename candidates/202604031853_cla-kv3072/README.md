# CLA KV Sharing + Bigram 3072

## Hypothesis

Pairwise Cross-Layer Attention (CLA) should let this repo share K/V projections across adjacent layers with little quality loss, then reuse part of the recovered artifact budget on a larger BigramHash table. On top of the current 11-layer LeakyReLU^2 + GPTQ-lite + legal-TTT stack, that should improve the quality/size tradeoff under the 16MB cap without a broad rewrite.

## Why this is promising for this repository

- The strongest recent records already sit close to the 16MB artifact cap, so getting more value from existing parameters matters more than adding broad new infrastructure.
- This codebase already uses GQA, parameter banking, and bigram features, so CLA is a natural next compression step rather than a foreign architecture.
- Pairwise K/V sharing cuts the banked K/V weights from **2,883,584** to **1,572,864** parameters, saving **1,310,720** weights. Raising `BIGRAM_VOCAB_SIZE` from 2048 to 3072 only spends back **131,072** weights, leaving a large net byte cushion for the quantized artifact.

## Influences from existing records and candidates

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- **Quantization/export baseline:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Partial RoPE + LN scaling stack:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Prior candidates:** none existed in this repository when this candidate was created.

## External research that informed it

- **Cross-Layer Attention** (Nrusimha et al., 2024, arXiv:2405.12981) shows that sharing K/V heads across adjacent layers can reduce KV state another 2x beyond MQA/GQA while keeping accuracy nearly unchanged.
- **MobileLLM / MobileLLM-LS** (Liu et al., 2024, arXiv:2402.14905) argues that sub-billion models benefit disproportionately from architectural sharing tricks, especially deep-thin designs with immediate block-wise reuse.

These papers do not match this repo's exact objective, but together they make a good case that small models can trade some K/V independence for better byte efficiency without a full redesign.

## What changed vs. the chosen base implementation

1. Added `CLA_SHARE_KV` and `CLA_GROUP_SIZE` hyperparameters, defaulting to pairwise K/V sharing (`CLA_SHARE_KV=1`, `CLA_GROUP_SIZE=2`).
2. Replaced the per-layer `kv_bank` with shared K/V bank slots reused across adjacent layers. Q/O and MLP banks remain layer-specific.
3. Added `normalize_shared_kv_grad()` so a reused K/V slot averages the gradients from its participating layers before the Muon step, keeping update scale closer to the non-shared baseline.
4. Updated the unbank/rebank export path so shared K/V slots are quantized exactly once under `cla_kv.*` keys and reconstructed back into the smaller bank shape after dequantization. This preserves the artifact-size win instead of duplicating shared tensors during PTQ.
5. Increased the default `BIGRAM_VOCAB_SIZE` from 2048 to 3072, using a small part of the recovered budget on a feature that already helped in the March 23 ablations.

## How to run or evaluate it

Run this from the repository root so the script's default `./data/...` paths still resolve correctly:

```bash
NUM_LAYERS=11 CLA_SHARE_KV=1 CLA_GROUP_SIZE=2 BIGRAM_VOCAB_SIZE=3072 \
SWA_ENABLED=1 SWA_EVERY=50 ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 torchrun --standalone --nproc_per_node=8 candidates/202604031853_cla-kv3072/train_gpt.py
```

If you want to isolate the architectural effect without evaluation-time adaptation, set `TTT_ENABLED=0`.

## Main risks and tradeoffs

- Adjacent layers may want different K/V subspaces; CLA could blur layer specialization and undercut the deepest layers.
- The win is primarily a **byte-allocation** idea, not a FLOP reduction idea; if the bigger bigram table is not the right place to reinvest the savings, the candidate may regress despite smaller artifacts.
- Shared bank slots make per-block adaptation less granular than the original layout, especially for future TTT or freezing experiments that want exact layer boundaries.

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604031853_cla-kv3072/train_gpt.py
```

Outcomes:

- `python -m compileall ...` **succeeded** for the baseline files and this candidate.
- A real CPU startup smoke check was **not feasible in this runner** because the available Python environment does not have `torch` importable at runtime (`ModuleNotFoundError: No module named 'torch'`), so the candidate could not be instantiated locally here even with a stubbed flash-attention module.
