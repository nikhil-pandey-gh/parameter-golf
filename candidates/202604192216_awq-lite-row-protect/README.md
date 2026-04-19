# AWQ-lite Row Protect

## Hypothesis

The current 11-layer record family is already strong on architecture, optimizer, and eval. The remaining gap is still heavily quantization-shaped, so a small **activation-aware export improvement** should be a better next bet than another minor architecture tweak.

This candidate tests an **AWQ-lite / GPTQ-lite hybrid**: calibrate on a few training batches, use those activation statistics to choose per-row int6 clip percentiles, and spend a tiny fp16 budget on the single worst rows instead of protecting whole tensors.

## Why this is promising for this repository

- The repo history shows repeated gains from better export paths: mixed int6/int8, zstd, and then GPTQ-lite all moved the leaderboard, while the core 11L stack changed much less after that.
- The 4-hour non-record run still landed at **1.2074** after roundtrip export, which is a strong sign that quantization remained a bottleneck even when training got much better.
- The latest clean pre-TTT base (`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`) already proved that a better post-training quantizer can buy real BPB without changing the architecture.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`  
  Main base. This candidate keeps that 11L XSA4 + EMA + Partial RoPE + LN-scale + VE stack and changes the exporter.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`  
  Important reminder that architectural wins there came from Partial RoPE + LN scale, and that compile-sensitive late-QAT wiring was already a concern in this code family.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`  
  Established the 11-layer EMA/XSA base that this candidate still uses.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md`  
  Useful evidence that better training alone does not remove the export bottleneck.

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

- **GPTQ** — Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*  
  https://arxiv.org/abs/2210.17323
- **LLM.int8()** — Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*  
  https://arxiv.org/abs/2208.07339
- **SmoothQuant** — Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*  
  https://arxiv.org/abs/2211.10438
- **AWQ** — Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*  
  https://arxiv.org/abs/2306.00978

This implementation is intentionally **AWQ-lite**, not a full runtime rescaling rewrite. The repo is already optimized around a compact single-file evaluation model, so a calibration-aware row selector is the least invasive way to test the same core idea.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes:

1. Added a short **post-EMA calibration pass** over training shards that records per-input RMS for large `CastedLinear` weights.
2. Replaced the plain GPTQ-lite row search with **activation-weighted per-row clip selection** for int6 matrices.
3. Added a tiny **global protected-row budget**:
   - default `AWQ_KEEP_ROWS=64` total rows,
   - default `AWQ_MAX_KEEP_PER_TENSOR=2`,
   - protected rows are stored in fp16 and written back after dequantization.
4. Left the underlying model stack alone: 11 layers, XSA on last 4 layers, Partial RoPE, LN scale, VE, BigramHash, SmearGate, EMA, SWA, and mixed int6 export.
5. Set `LATE_QAT_THRESHOLD=0` by default so this candidate isolates the export change instead of relying on the compile-sensitive late-QAT path from the same code family.

## How to run / evaluate

Run from this directory:

The script resolves `DATA_PATH` and `TOKENIZER_PATH` from the repository root by default, so it can be launched directly from inside this candidate folder without extra path overrides.

```bash
cd candidates/202604192216_awq-lite-row-protect

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
LATE_QAT_THRESHOLD=0 \
AWQ_CALIBRATION_BATCHES=4 AWQ_CALIBRATION_BATCH_TOKENS=262144 \
AWQ_KEEP_ROWS=64 AWQ_MAX_KEEP_PER_TENSOR=2 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still writes the same main artifacts as the base:

- `final_model.pt`
- `final_model.int6.ptz`
- `logs/<run-id>.txt`

## Main expected risks / tradeoffs

- Calibration may overfit to the sampled training activations and protect the wrong rows.
- The fp16 protected-row reserve can waste bytes if the chosen rows are not actually the dominant post-quantization error source.
- If `AWQ_KEEP_ROWS` is pushed too high, the compressed artifact can lose the benefit it is trying to buy back.
- This is an export-focused improvement; if the main bottleneck is now training dynamics rather than roundtrip quantization, gains may be small.
- A stronger follow-up would be a compile-safe late-QAT path or a fuller SmoothQuant-style rescaling rewrite, but both are more invasive than this candidate.

## Validation

- From the repository root: `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604192216_awq-lite-row-protect/train_gpt.py`  
  Passed in this environment.
- A minimal CPU import/runtime smoke was **not feasible here** because the environment does not currently have the repo's Python dependencies installed (`torch` is absent), and the full script also expects the CUDA/FlashAttention stack used by the existing record implementations.
