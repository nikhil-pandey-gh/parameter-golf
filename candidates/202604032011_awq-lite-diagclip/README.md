# AWQ-lite Diagonal Clip Search on the LeakyReLU² + Legal TTT Stack

## Hypothesis

The current record family has already harvested most of the obvious architecture and schedule wins. The next high-probability gain is to reduce the **post-training int6 export error** by making clip selection depend on **real activation statistics**, not only weight reconstruction error. If the final artifact is chosen using training-token activation RMS, the same 11L/LeakyReLU²/TTT stack should quantize a little more faithfully and recover a small but meaningful amount of validation BPB.

## Why this is promising here

- The repo's strongest non-TTT quantization result already came from **GPTQ-lite clip search** (`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`), which improved export quality with **zero training-cost architecture changes**.
- The current best full stack (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`) is already strong before TTT (**~1.1218 bpb pre-TTT**) and gets only a modest final boost from legal TTT (**~0.0025 bpb**), so tightening the export path is one of the cleanest remaining levers.
- Earlier record history shows that **compression-aware choices dominate** once the model reaches 10-11 layers, 3x MLP, XSA, Partial RoPE, and EMA. This candidate stays on that winning frontier instead of reopening a broad architecture search.

## Prior records and candidates that influenced this

- **Base implementation**: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- **Quantization direction**: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- **Architectural stack carried forward**: `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
- **Compression-aware stack origin**: `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md`
- **There were no prior `candidates/` iterations** when this candidate was created.

## External research that informed it

- **AWQ** (Activation-aware Weight Quantization), arXiv:2306.00978 — motivates using activation statistics, not just raw weight magnitude, to decide which channels are most quantization-sensitive.
- **SmoothQuant**, arXiv:2211.10438 — reinforces that activation outliers drive quantization difficulty and that offline calibration can materially improve PTQ.
- **GPTQ**, arXiv:2210.17323 — motivates search-based post-training quantization that optimizes downstream error rather than a naive clipping rule.
- **SpinQuant**, arXiv:2405.16406 — suggests that more aggressive rotation-based quantization improvements exist, but they are substantially more invasive than this repository's current code path.

This candidate intentionally takes the narrowest useful version of those ideas: **activation-aware clip search without adding new training infrastructure or a full equivalent-transform/rotation stack**.

## What changed versus the chosen base

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate adds:

1. **Training-token calibration pass before export**
   - Collects per-input-channel RMS statistics for every quantized attention and MLP matrix.
   - Uses a fresh `DistributedTokenLoader` on the training shards, not the validation split.

2. **Activation-aware int6 row clipping**
   - Replaces weight-only row-percentile selection with a diagonal output-error proxy:
     - candidate clip percentiles are still searched per row,
     - but rows are scored by reconstruction error weighted by the observed input-channel RMS.
   - This is an **AWQ-lite / GPTQ-lite hybrid**: still cheap and PTQ-only, but now calibrated on actual model usage.

3. **AWQ-specific logging/config**
   - `AWQ_ENABLED=1` by default
   - `AWQ_CALIBRATION_BATCHES=8`
   - `AWQ_CALIBRATION_BATCH_SEQS=2`
   - optional `INT6_CLIP_CANDIDATES=...`

Everything else is intentionally left as close as possible to the 2026-03-23 base:

- 11 layers, 512 width, 8 heads / 4 KV heads
- 3x MLP with **LeakyReLU(0.5)^2**
- XSA on the last 4 layers
- Partial RoPE (16 dims), LN scaling, SmearGate, BigramHash, VE
- EMA late-averaging path carried by the chosen base code
- Parallel Muon + legal score-first TTT evaluation

## How to run / evaluate

From the repository root:

```bash
cd candidates/202604032011_awq-lite-diagclip

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
AWQ_ENABLED=1 AWQ_CALIBRATION_BATCHES=8 AWQ_CALIBRATION_BATCH_SEQS=2 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to ablate only the new idea, keep the run fixed and compare:

```bash
AWQ_ENABLED=0
```

## Validation run for this candidate

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604032011_awq-lite-diagclip/train_gpt.py` | Passed |
| Minimal CPU smoke test | Not run: this script hard-requires CUDA and `flash_attn_interface`, and the repository does not provide an existing CPU fallback path for the real training/eval loop |

## Main expected risks / tradeoffs

- **Small-gain regime**: this is targeting a late-stage export improvement, so the upside is probably measured in low-thousandths of BPB rather than a large architectural jump.
- **Diagonal approximation**: the clip search uses channel RMS as a cheap proxy for output error, not full second-order reconstruction as in GPTQ.
- **Extra export time**: calibration adds a short post-training forward pass before serialization.
- **Calibration mismatch**: early training-shard statistics may not perfectly match the final evaluation distribution.
- **Unverified end-to-end quality locally**: only syntax validation was run here; the real question is whether the activation-aware clip selection beats the existing GPTQ-lite export on H100 runs.
