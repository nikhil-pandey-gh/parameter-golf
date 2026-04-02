# Activation-Aware GPTQ-lite Int6 Export

## Hypothesis

The strongest non-TTT stack in this repo already does a lightweight GPTQ-lite search over a few clip percentiles for each int6 weight row. That search still scores candidates using plain weight reconstruction MSE. For this challenge, the metric is post-quantized validation BPB, so errors on channels that see larger activations should matter more than errors on rarely-used channels.

This candidate adds a small post-EMA calibration pass over **training tokens only** and uses the observed input second moments to pick the clip percentile that minimizes an **activation-weighted reconstruction proxy** for each int6 weight matrix. The goal is to reduce the quantization gap without touching the proven 11-layer training recipe.

## Why this is promising for this repository

- Repo history shows that **compression-aware tweaks keep paying off**: int6 export, zstd/lzma, EMA/SWA, weight decay, and GPTQ-lite all produced real gains.
- The current best pure training/export record is already very strong at training time, so an **export-only improvement** is one of the cleanest ways to search for additional BPB wins without risking throughput.
- This candidate keeps the exact same artifact format and nearly the same code path as the `2026-03-22` base, which makes it easier to attribute any gain or loss to the quantization change.

## Prior records that influenced this candidate

| Record | Influence |
|---|---|
| `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` | Direct base. Reuses the full 11L stack and extends its GPTQ-lite export path. |
| `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` | Reinforced that small, surgical changes on top of the mature 11L stack can still move the metric. |
| `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` | Established the 11L + XSA + EMA + int6 direction that later records built on. |

## External research that informed it

- **GPTQ** — Frantar et al., 2022/2023: one-shot post-training quantization using reconstruction-aware scoring.  
  https://arxiv.org/abs/2210.17323
- **AWQ** — Tang et al., 2024: activation-aware weight quantization, motivated by the idea that activation statistics identify the channels whose quantization error matters most.  
  https://arxiv.org/abs/2306.00978
- **SmoothQuant** — Xiao et al., 2023/2024: activation statistics can be used offline to make low-bit quantization more accurate.  
  https://arxiv.org/abs/2211.10438

This candidate is intentionally simpler than full AWQ or SmoothQuant. It does **not** introduce channel rescaling or a new runtime format. Instead, it borrows the central idea that activation statistics should influence quantization decisions, then applies that idea to the repo's existing GPTQ-lite clip search.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

1. Added a post-EMA calibration pass that runs `forward_logits` on a small batch of **training** tokens and records per-linear-layer input second moments.
2. Kept the existing 5-candidate percentile search for int6 quantization, but changed the selection criterion from plain weight MSE to an **activation-weighted reconstruction error** proxy.
3. Added small environment knobs:
   - `AWQ_ENABLED` (default `1`)
   - `AWQ_CALIBRATION_TOKENS` (default `131072`)
   - `AWQ_BATCH_SEQS` (default `8`)

Everything else is intentionally left aligned with the `2026-03-22` record: 11 layers, XSA on the last 4 layers, partial RoPE, LN scaling, VE, SmearGate, BigramHash, EMA + tight SWA, late QAT threshold, and the same int6 artifact path.

## How to run

Example launch, matching the base stack with the activation-aware export enabled:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
AWQ_ENABLED=1 AWQ_CALIBRATION_TOKENS=131072 AWQ_BATCH_SEQS=8 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 candidates/202604021748_awq-lite-int6/train_gpt.py
```

Evaluation remains the same as the base script: it exports `final_model.int6.ptz`, performs the roundtrip load, and reports both roundtrip and sliding-window BPB.

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall train_gpt.py train_gpt_mlx.py data` | Passed |
| `python -m compileall candidates/202604021748_awq-lite-int6/train_gpt.py` | Passed |
| `python - <<'PY' ... importlib.util.find_spec(\"torch\") ... PY` | `torch`, `flash_attn_interface`, cached dataset, and tokenizer were all absent on this runner |

A minimal runtime smoke test was therefore **not feasible** here: this repository's CUDA path requires `torch`, `flash_attn_interface`, the cached FineWeb shards, and the SentencePiece tokenizer, and none of those runtime prerequisites were present in the workflow environment.

## Main risks and tradeoffs

- The calibration slice may be too small or too unrepresentative, which could make the activation-weighted score noisier than plain weight MSE.
- The extra calibration pass adds some export/eval time, so the token count may need tuning if this becomes part of a record attempt.
- Full AWQ-style channel rescaling is more expressive than this lightweight proxy; if this candidate helps, the next step is likely a stronger activation-aware transformation rather than more percentile tuning.
