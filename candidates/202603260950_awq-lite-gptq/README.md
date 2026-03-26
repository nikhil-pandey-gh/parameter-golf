# AWQ-lite / activation-aware GPTQ-lite export candidate

## Hypothesis

The strongest remaining low-cost improvement path in this repository is likely **better post-training quantization**, not yet another training-time architectural stack-up. The recent record lineage already converged on an 11-layer, 3x-MLP, EMA, partial-RoPE, XSA, sliding-window recipe, but it still keeps winning by shaving down export loss. This candidate tests an **activation-aware GPTQ-lite** variant: instead of picking int6 row clipping purely from weight reconstruction error, it uses a short calibration pass to weight quantization error by each layer's observed input-channel RMS.

## Why it is promising for this repository

Two patterns from the repo stand out:

- Quantization error has repeatedly been large enough to dominate small architecture changes.
- Zero- or low-training-cost export improvements have continued to move the frontier, including fp16 embedding passthrough and GPTQ-lite percentile search.

That makes an AWQ-inspired, calibration-only export improvement a particularly good fit for the Parameter Golf constraints. It adds a few short forward passes before export, but it does **not** add training FLOPs during the 10-minute run.

## Prior records that influenced this candidate

Primary base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Key repository evidence:

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` showed that export sensitivity, especially around embeddings, can overwhelm small training gains.
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` argued that training-for-quantization can matter as much as raw model quality.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` showed that a lightweight percentile-search GPTQ-lite pass was already worth another measurable improvement.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` confirmed that the frontier is now won by stacking small, orthogonal improvements; activation-aware export is intended as another such orthogonal gain.

## External research that informed it

Primary sources:

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** — <https://arxiv.org/abs/2306.00978>
- **SmoothQuant** — <https://arxiv.org/abs/2211.10438>
- **Scaling Law for Quantization-Aware Training** — <https://arxiv.org/abs/2505.14302>

The candidate does **not** implement full AWQ channel rescaling. Instead, it takes the most repository-compatible lesson from AWQ and related work: activation statistics identify which weight errors matter most. Here that idea is applied as a minimal modification to the repo's existing GPTQ-lite percentile search.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

- added `AWQ_*` hyperparameters for a small export-time calibration pass,
- collected per-module input RMS statistics from a handful of train batches using forward pre-hooks on `CastedLinear` modules,
- changed int6 percentile search to minimize **activation-weighted** reconstruction error instead of plain weight MSE,
- kept the rest of the training stack intentionally unchanged so the quantization change is isolated.

This is best thought of as **AWQ-lite / activation-aware GPTQ-lite**, not a full AWQ reimplementation.

## How to run or evaluate it

From this candidate directory:

```bash
RUN_ID=awq_lite_candidate \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=600 \
AWQ_ENABLED=1 \
AWQ_CALIBRATION_STEPS=8 \
AWQ_CALIBRATION_BATCH_TOKENS=131072 \
python train_gpt.py
```

Suggested 8-GPU training invocation on the challenge hardware:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

- `AWQ_ENABLED=0` disables the new export path and reverts to the base GPTQ-lite behavior.
- `AWQ_CALIBRATION_STEPS` controls how much calibration data is used.
- `AWQ_CALIBRATION_BATCH_TOKENS` should provide an integer number of sequences per rank.
- `AWQ_CLIP_CANDIDATES` can widen or narrow the percentile search.

## Validation

Commands run locally in this workflow:

- `python -m compileall train_gpt.py train_gpt_mlx.py data` — passed before implementation
- `python -m compileall candidates/202603260950_awq-lite-gptq/train_gpt.py` — passed
- `python -m py_compile candidates/202603260950_awq-lite-gptq/train_gpt.py` — passed

CPU smoke test status:

- A best-effort module import check was attempted, but the workflow environment does not have the repository's Python dependencies installed (`ModuleNotFoundError: No module named 'numpy'` was the first failure), and the trainer is GPU-first beyond that. A real runtime smoke test was therefore not feasible here.

## Main expected risks or tradeoffs

- Activation-weighted clipping may overfit to the short calibration sample if the sample is too small.
- Because this is not full AWQ channel rescaling, gains may be smaller than the full method's published improvements.
- Some layers may prefer plain MSE clipping; the activation-aware objective is most likely to help the most activation-sensitive projections rather than every matrix uniformly.
- Export-time calibration adds a small amount of wall-clock after training, even though it does not consume training budget.
