# 11L EMA + GPTQ-lite + AWQ-SpQR Sidecar

## Hypothesis

The strongest remaining gap in this repository is no longer basic training quality; it is the residual loss from packing a strong 11-layer model into an int6 artifact. A small **activation-aware sparse correction sidecar** should recover part of that gap with **zero training-step overhead** by spending a small slice of the remaining artifact budget on the weights that matter most after GPTQ-lite quantization.

## Why this is promising here

- The record progression moved from architecture and optimizer changes toward increasingly careful export logic: fp16 embeddings, mixed int6/int8, EMA/SWA, then GPTQ-lite clip search.
- The best pre-TTT record, `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, already lands at **15.56 MB**, leaving enough headroom to try a sparse correction sidecar without deleting model capacity.
- The non-record 4-hour run under `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/` showed that long training alone was not enough once the post-quantization gap dominated.

## Prior records that informed this candidate

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`: established the durable 11L + MLP3x + XSA4 + EMA + WD=0.04 stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`: added partial RoPE and LN-scale with zero-parameter improvements.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`: best non-TTT base and the direct code parent for this candidate.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`: confirmed the frontier moved into activation/export/eval details, but that stack is byte-tight and adds TTT/banked-optimizer complexity that is unnecessary for this export-focused experiment.

## External research

- **GPTQ** (Frantar et al., arXiv:2210.17323): one-shot weight quantization with second-order motivation; this repository already uses a lightweight GPTQ-style clip search.
- **AWQ** (Tang et al., arXiv:2306.00978): activation-aware protection of salient channels; this candidate borrows the calibration idea, but uses it only to rank correction weights rather than rescale runtime channels.
- **SpQR** (Egiazarian et al., arXiv:2306.03078): sparse-quantized representation that stores especially harmful outliers in higher precision; this candidate adapts that idea into a tiny fp16 correction sidecar on top of int6 export.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate keeps the base architecture and optimizer recipe intact:

- 11 layers, 512d, 8 heads / 4 KV heads
- MLP×3, SmearGate, BigramHash, XSA on last 4 layers
- Partial RoPE (16/64), LN-scale, EMA + tight SWA
- GPTQ-lite per-row int6 export with zstd-22

New logic added on top:

1. **Short AWQ-style calibration pass** on train tokens after EMA, collecting per-input-channel RMS for each `CastedLinear`.
2. **Global sparse sidecar selection** after GPTQ-lite quantization: score each residual entry by `abs(weight_error) * activation_rms`, with a modest bonus for the last two layers.
3. **Budgeted fp16 residual sidecar**: keep only a tiny global fraction (`INT6_SIDECAR_FRAC`, default `0.0015`) of the highest-scoring corrections, stored as sparse `(row, col, value)` triplets.
4. **Dequantization scatter-add**: rebuild the dense tensor from int6 + sparse fp16 corrections before evaluation.

The important constraint is that this changes the **artifact format**, not the training loop throughput.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202604081056_awq-spqr-sidecar
RUN_ID=awq_spqr_sidecar \
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
AWQ_CALIBRATION_SEQS=32 AWQ_CALIBRATION_BATCH_SEQS=8 \
INT6_SIDECAR_ENABLED=1 INT6_SIDECAR_FRAC=0.0015 \
INT6_SIDECAR_LATE_LAYER_BONUS=1.25 INT6_SIDECAR_OVERSELECT=4.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The key new knobs are:

- `AWQ_CALIBRATION_SEQS`: number of train sequences used to gather activation RMS
- `INT6_SIDECAR_FRAC`: global fraction of int6 entries kept as sparse fp16 corrections
- `INT6_SIDECAR_LATE_LAYER_BONUS`: slight priority boost for late-layer corrections
- `INT6_SIDECAR_OVERSELECT`: over-sample candidates per tensor before global top-k selection

## Risks and tradeoffs

- The sparse sidecar may **compress worse than expected**, especially if too many scattered indices survive zstd; if that happens, the first sweep should reduce `INT6_SIDECAR_FRAC`.
- Calibration is cheap, but it still adds **post-training export time**.
- The sidecar may help only marginally if GPTQ-lite already removed most of the damaging error for this stack.
- The existing late-QAT toggle remains inherited from the base implementation; this candidate does **not** try to fix or depend on that path.

## Validation

Commands run in this repository:

```bash
python -m compileall candidates/202604081056_awq-spqr-sidecar/train_gpt.py
```

Outcome:

- Passed.

Attempted CPU-only smoke:

```bash
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("candidate_train_gpt", "candidates/202604081056_awq-spqr-sidecar/train_gpt.py")
...
PY
```

Outcome:

- Not feasible in this runner because the Python environment here does not provide `torch`, so the candidate module cannot be imported for a real runtime smoke without installing non-repo dependencies.
- I therefore kept validation to syntax compilation only, instead of changing the candidate to add a fake CPU-only path that would not reflect the real challenge runtime.
