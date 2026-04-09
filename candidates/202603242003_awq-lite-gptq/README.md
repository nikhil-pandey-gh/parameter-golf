## Hypothesis

The current 11-layer SOTA lineage looks more bottlenecked by **post-training quantization quality** than by raw pre-quant model quality. This candidate tests whether an **AWQ-lite selective channel scaling pass** can improve the existing GPTQ-lite int6 export path: collect a few batches of activation RMS statistics after EMA, scale salient input channels for large int6 `mlp`/`attn` weights, then run the same rowwise clip-percentile search on the scaled weights.

## Why this is promising for this repository

Repository evidence points in the same direction:

- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` shows longer training helped pre-quant quality much more than post-quant quality.
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` explicitly frames the game as training for compressibility, not just raw loss.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` shows the current best run still won via a **better export quantizer** (`GPTQ-lite`) rather than another major architecture jump.
- The same repo review also showed no prior run using activation-aware channel salience or AWQ-style scaling.

So the bet here is not "add more model," but "spend a tiny amount of extra logic where the repo is still losing bits."

## Prior records that influenced this candidate

Base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Most relevant local influences:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - strongest base recipe
  - showed `GPTQ-lite` clip search mattered
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - important warning that late-QAT toggles were compile-fragile
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md`
  - emphasized compression-aware training
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md`
  - showed careful mixed low-bit export is where big gains come from
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md`
  - reinforced that selective quantization detail still moves the score

## External research that informed it

- **AWQ** — Activation-aware Weight Quantization for LLM Compression and Acceleration (`arXiv:2306.00978`)
  - key takeaway used here: activation statistics identify salient channels better than weight magnitude alone.
- **GPTQ** — Accurate Post-Training Quantization for Generative Pre-trained Transformers (`arXiv:2210.17323`)
  - motivated keeping the candidate centered on a better post-training weight quantizer instead of a new training stack.
- **SmoothQuant** (`arXiv:2211.10438`)
  - reinforced the idea that equivalent rescaling transformations can make low-bit quantization easier.
- **SpinQuant** (`arXiv:2405.16406`)
  - pointed to the same high-level lesson from newer work: quantization accuracy often improves when you reshape where outliers live before quantizing.
- **Continuous Approximations for Improving QAT of LLMs** (`arXiv:2410.10849`)
  - informed the decision to avoid making this candidate depend on brittle late-QAT behavior and instead target the export quantizer directly.
- **ALBERT** (`arXiv:1909.11942`)
  - considered as part of the broader search space for parameter sharing, but not chosen here because the repository's strongest trend is still "smarter quantization first."

## What changed versus the chosen base

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`:

- Added repo-root-relative defaults for `DATA_PATH` and `TOKENIZER_PATH` so the script can be run directly from this candidate directory.
- Added dormant `awq_input_scale` / `awq_runtime_enabled` buffers to each `CastedLinear`.
- Added a small **post-EMA calibration pass** that runs a few train batches through the uncompiled base model, records per-input-channel RMS for large int6 candidate weights, and reduces those stats across ranks before computing scales.
- Added `inject_awq_buffers(...)` so only selected large `mlp` / `attn` weights receive nontrivial AWQ buffers in the quantized export state.
- Modified `quantize_int6_per_row(...)` so GPTQ-lite clip search operates on `weight / awq_input_scale` when a calibrated scale is present.
- Left the rest of the winning 11L recipe intact: EMA, XSA4, partial RoPE, LN scaling, shared value embedding, SmearGate, BigramHash, zstd export, etc.
- Set `LATE_QAT_THRESHOLD=0.0` by default in this candidate because earlier repo evidence showed the compile-time toggle path was fragile; this keeps the experiment focused on the new quantization idea.

## How to run or evaluate it

From this candidate directory:

```bash
RUN_ID=awq_lite_gptq \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful AWQ-specific knobs:

```bash
AWQ_ENABLED=1
AWQ_CALIBRATION_BATCHES=4
AWQ_CALIBRATION_BATCH_TOKENS=131072
AWQ_SCALE_EXP=0.5
AWQ_MAX_SCALE=4.0
```

To recover the base export behavior, disable the new path:

```bash
AWQ_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- The calibration pass consumes extra train-data compute after EMA and needs measurement against the 600s wallclock budget before this could be promoted beyond candidate status.
- The AWQ-lite heuristic here is intentionally lightweight; it uses RMS-based channel salience instead of AWQ's full search procedure, so gains may be modest or need tuning.
- Persisting per-linear input scales adds some artifact bytes, though it should be much smaller than changing model shape.
- Because the repo's late-QAT path has been fragile under `torch.compile`, this candidate avoids depending on that path by default.

## Validation

Commands run locally in this container:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603242003_awq-lite-gptq/train_gpt.py
git --no-pager diff --check
```

Outcomes:

- All listed compile checks passed.
- `git diff --check` passed.
- A real runtime smoke test was **not feasible** in this container because the local Python environment does not have `torch`, `sentencepiece`, or `flash_attn_interface` installed, so the training script cannot be meaningfully imported or launched here.
