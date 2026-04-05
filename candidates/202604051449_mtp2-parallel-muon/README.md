# MTP2 + Parallel Muon + Legal TTT

## Hypothesis

Add a **2-head multi-token prediction (MTP)** auxiliary loss to the strongest current 10-minute stack so each training token teaches the trunk about both the next token and the next two future tokens. Because the extra heads are used only during training and are stripped before export, the candidate can chase better sample efficiency without spending any of the 16 MB artifact budget.

## Why this is promising for this repository

The record trend says the best gains now come from stacking **training-time-only improvements** on top of the mature 11-layer compression-aware architecture:

| Evidence from prior records | Implication |
| --- | --- |
| `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` established the 11L + XSA + EMA + int6 path. | The core architecture is already strong; new wins likely come from better optimization, not another wholesale rewrite. |
| `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` and `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved the same stack with mostly zero- or low-parameter tweaks. | Zero-export-cost improvements are especially valuable under the artifact cap. |
| `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` reached **1.1194 post-TTT** while keeping export size at **~15.95 MB**. | The latest record is the right base: it already wins with training-time changes that do not bloat the final artifact. |
| `2026-03-18_FP16Embed_WD3600` and `2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090` report recurrence and slower MLP variants as wall-clock losers. | The next idea should improve **sample efficiency per step** without materially slowing the main trunk. |

There were **no prior `candidates/` directories** when this run started, so this is the first candidate fork in that namespace.

## External research that informed this candidate

1. **Gloeckle et al., "Better & Faster Large Language Models via Multi-Token Prediction" (arXiv:2404.19737)** argues that predicting multiple future tokens with independent heads improves sample efficiency and helps induction-style behavior. That is a strong fit for a 10-minute training budget where every token update has to work harder.
2. **Croci et al., "QuaRot" (arXiv:2404.00456)** and **Liu et al., "SpinQuant" (arXiv:2405.16406)** make rotation-based quantization look promising, but they also point toward more invasive architecture-preserving transforms. For this repository, that made MTP more attractive as the next step because it is simpler to adapt, training-only, and export-free.

## Records that influenced this implementation

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- **Supporting quantization/export baseline:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Earlier zero-parameter architectural gains:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Dead-end guidance:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`

## What changed versus the chosen base

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, this candidate:

1. **Turns on MTP by default** with `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15`.
2. **Fixes the optimizer wiring for MTP heads** in the parameter-banked training path. The 03-23 script logs MTP settings and computes the loss, but its optimizer split never adds `mtp_heads.*.weight` to any optimizer or replicated all-reduce list, so nonzero MTP settings would not actually train the auxiliary heads.
3. **Keeps export budget unchanged** by continuing to strip `mtp_heads` from `final_model.pt` and the quantized artifact before evaluation/export.
4. Aligns a couple of defaults with the 03-23 winning recipe (`BIGRAM_VOCAB_SIZE=1536`, `TTT_FREEZE_BLOCKS=0`) so the candidate script matches the intended base run more closely.

## How to run

Training + export + standard sliding-window eval:

```bash
cd candidates/202604051449_mtp2-parallel-muon

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 \
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

If the local data cache is missing, fetch the published tokenizer + shards from the **repo root**:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

## Validation

These checks were run from the **repo root**:

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604051449_mtp2-parallel-muon/train_gpt.py` - **passed**
- `/tmp/gh-aw/agent/venv/bin/python - <<'PY' ...` CPU smoke test with a stubbed `flash_attn_interface` - **passed** (`cpu_smoke:ok loss=4.7816 delta=2.047852`)

## Main risks and tradeoffs

- **Wall-clock tradeoff:** extra logits for future-token heads may reduce steps under the 600s cap, so the MTP gain has to beat that throughput hit.
- **Scale uncertainty:** the MTP paper is strongest on larger models; this ~27M-parameter stack may benefit less.
- **Quantization interaction:** MTP should improve the trunk before export, but it may also shift weight statistics in ways that help or hurt int6 + lzma compression.
- **Inherited complexity:** this candidate intentionally leaves the rest of the 03-23 stack intact, so the result still depends on the existing TTT, EMA/SWA, and quantization pipeline.
