# 202603251501_act-gptq-lite

## Hypothesis

The strongest non-TTT stack in this repo is already close to saturated on architecture and training knobs, but it still depends on a fairly simple post-training clip search for int6 export. Replacing that weight-only clip objective with an activation-aware, row-wise GPTQ-lite objective should reduce the round-trip quantization error of the int6 weights without changing the trained model or artifact format, improving post-quant validation BPB under the same 16 MB budget.

## Why this is promising for this repository

Repository history keeps pointing to quantization as the remaining bottleneck:

- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/` reached `1.1749` pre-quant BPB but only `1.2074` after quantization, showing that extra training alone does not fix export loss.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` improved the best non-TTT record by another `-0.0013` BPB using a slightly smarter clip search and EMA, even after the major architectural wins had already landed.
- The winning stacks consistently keep tied embeddings gentler than the rest of the network and rely on export-aware decisions, which suggests the next cheap gain is more quantization sensitivity, not another large architectural rewrite.

This candidate therefore keeps the proven 11-layer base intact and only upgrades the export path.

## Prior records that influenced this candidate

The main implementation base is:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

The following records shaped the decision:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` confirmed the current 11L XSA + partial-RoPE stack is a strong architecture base.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` showed there is still quality left in the stack, but the best incremental gains there came from evaluation-time TTT rather than a simpler export change.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` documented layer recurrence as a bad fixed-compute tradeoff, which pushed this candidate away from broader architecture experiments.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/` reinforced that quantization, not just model quality, is the limiter.

## External research that informed it

This candidate is a lightweight synthesis of several PTQ lines of work:

- **GPTQ** (`arXiv:2210.17323`) argues for data-aware, second-order post-training quantization instead of raw weight reconstruction.
- **AWQ** (`arXiv:2306.00978`) argues that activation statistics identify the channels that matter most for preserving model behavior.
- **SmoothQuant** (`arXiv:2211.10438`) shows that activation outliers are tightly coupled to quantization difficulty.
- **Activation Sensitivity as a Unifying Principle for PTQ** (`arXiv:2601.11663`) frames AWQ-style activation weighting and GPTQ-style data awareness as approximations to the same underlying sensitivity signal.
- I also looked at newer alternatives like **SpinQuant** (`arXiv:2405.16406`) and **Forgetting Transformer** (`arXiv:2503.02130`), but they require broader architectural or rotation machinery than this repository currently favors for quick candidate iteration.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes one focused change:

1. During normal training, it saves a tiny number of already-seen training batches for later export calibration.
2. After EMA is applied, it runs a short **post-EMA calibration pass** over those saved batches and records **per-layer input second moments** for each quantized `CastedLinear` weight matrix.
3. The int6 export path switches from a matrix-level weight-MSE clip choice to an **activation-aware, row-wise clip search**:
   - still testing a tiny fixed candidate set of clip percentiles,
   - but now selecting the best candidate **per output row**,
   - and scoring candidates with an **activation-weighted reconstruction error** rather than plain weight MSE.
4. The runtime model, training loop, artifact format, and evaluation path remain otherwise unchanged.

In other words: same trained model family, smarter export objective.

## New knobs

The candidate adds a few export-only environment variables:

- `ACT_GPTQ_ENABLED=1`
- `ACT_GPTQ_CALIBRATION_STEPS=8`
- `ACT_GPTQ_CLIP_PERCENTILES=0.9990,0.9995,0.9999,0.99999,1.0`

The defaults are tuned to stay lightweight. Unlike the base record, the default dataset and tokenizer paths are resolved relative to the repository root from `__file__`, so the candidate can be launched from inside its own directory without overriding `DATA_PATH` or `TOKENIZER_PATH`.

## How to run / evaluate

From the repository root:

```bash
SEED=1337 \
ACT_GPTQ_ENABLED=1 \
ACT_GPTQ_CALIBRATION_STEPS=8 \
torchrun --standalone --nproc_per_node=8 \
  candidates/202603251501_act-gptq-lite/train_gpt.py
```

Or from inside the candidate directory:

```bash
cd candidates/202603251501_act-gptq-lite
SEED=1337 \
ACT_GPTQ_ENABLED=1 \
ACT_GPTQ_CALIBRATION_STEPS=8 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want an ablation against the base export, rerun with `ACT_GPTQ_ENABLED=0`.

## Expected risks / tradeoffs

- The gain may be small if the existing GPTQ-lite clip candidates already capture most of the quantization error structure.
- Calibration adds a bit of post-training time, so it must stay lightweight to remain attractive.
- The activation-weighted objective may overfit slightly to the saved calibration sample if that sample is too narrow.
- This does not address tied-embedding quantization directly; if that remains the bottleneck, a follow-up candidate may need embedding-specific sensitivity handling.

## Validation

Commands run for this candidate:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603251501_act-gptq-lite/train_gpt.py
python -m compileall candidates/202603251501_act-gptq-lite/train_gpt.py
```

Outcomes:

- Both syntax-only `compileall` checks passed.
- I attempted a CPU-only smoke test by importing the candidate with a stubbed FlashAttention path, but that runner's available Python interpreters do not have `torch` or `sentencepiece` installed, so an import/forward smoke run was not feasible without adding infrastructure that does not already exist in this repo.
