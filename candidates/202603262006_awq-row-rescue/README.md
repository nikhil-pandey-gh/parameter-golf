# AWQ-lite Salient Row Rescue

## Hypothesis

The recent record stack suggests this repo is now more constrained by **post-training quantization error** than by raw training loss. The hypothesis here is that a small fraction of output rows dominate that quantization gap, so an **activation-aware row rescue** pass can improve the int6 roundtrip artifact more efficiently than changing the core architecture again.

In practice, this fork keeps the strong 11-layer `2026-03-22` backbone, then uses a short calibration pass over training tokens to estimate per-module output RMS and rescues only the most salient rows into fp16 while leaving the bulk of each matrix in GPTQ-lite-style int6.

## Why this looks promising here

The repository history points pretty clearly at quantization as the remaining bottleneck:

- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` trained much longer but still finished with a large export gap.
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md` and `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` both improved by treating sensitive tensors more gently.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` got another gain from better clip selection alone, which is exactly the kind of surface this candidate extends.

That made quantization-aware export a better fit than re-trying long-context-only ideas, pure LR sweeps, or naive recurrence.

## Influential prior records and candidates

At workflow start there were no pre-existing `candidates/` experiments to reuse.

The main local influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - clean 11-layer base with EMA, GPTQ-lite clip search, XSA, partial RoPE, LN scaling, VE, and strong export plumbing.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - provided the low-risk `LeakyReLU(0.5)^2` MLP activation change.
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`
  - reinforced that protecting the most quantization-sensitive weights is often worth more than uniform lower precision.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful mainly as a warning: naive recurrence hurt badly there, so this fork avoids architectural churn.

## External research that informed this candidate

- **AWQ** — Activation-aware Weight Quantization for LLM Compression and Acceleration, arXiv:2306.00978  
  Main takeaway used here: activation statistics are often a better guide to importance than weight magnitude alone.
- **LLM.int8()** — 8-bit Matrix Multiplication for Transformers at Scale, arXiv:2208.07339  
  Main takeaway used here: isolate the small set of outlier features instead of treating every channel the same.
- **SpQR** — A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression, arXiv:2306.03078  
  Main takeaway used here: explicitly preserving a tiny set of high-error weights/rows can substantially reduce quantization damage.
- **SmoothQuant** — Accurate and Efficient Post-Training Quantization for Large Language Models, arXiv:2211.10438  
  Main takeaway used here: calibration-time activation statistics can be used to move quantization difficulty away from the most fragile parts of the model.

This candidate does **not** implement full AWQ or SmoothQuant equivalence transforms. It adapts the central idea into a much smaller repo-friendly export change.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Main changes in this candidate:

1. **LeakyReLU(0.5)^2 MLP**
   - Replaces `relu(x)^2` with `leaky_relu(x, 0.5)^2`, matching the strongest recent low-risk MLP tweak from the current top record.

2. **Calibration-time salience collection**
   - After EMA is applied, the script runs a short calibration pass on training shards.
   - Forward hooks on each `CastedLinear` collect per-output-channel RMS.
   - This keeps the calibration entirely on training data rather than validation data.

3. **Salient-row rescue during GPTQ-lite int6 export**
   - The existing per-row percentile search is retained.
   - For each 2D int6-quantized tensor, the exporter scores rows by quantization MSE, optionally reweighted by calibration RMS.
   - Only the top rows (controlled by `SALIENT_ROW_FRACTION`, `SALIENT_ROW_MIN`, `SALIENT_ROW_MAX`) are stored exactly in fp16.
   - Their corresponding int6 rows are zeroed before serialization so the bulk payload stays compressible.
   - If the compressed artifact grows too large, the exporter automatically retries with smaller rescue budgets and finally falls back to plain GPTQ-lite int6 before raising if the artifact is still over the configured size limit.

4. **Roundtrip reconstruction restores rescued rows**
   - During dequantization, the rescued fp16 rows overwrite the quantized approximation before evaluation.

## How to run

From this candidate directory:

```bash
cd candidates/202603262006_awq-row-rescue

DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
SALIENT_CALIBRATION_STEPS=8 \
SALIENT_CALIBRATION_BATCH_TOKENS=131072 \
SALIENT_ROW_FRACTION=0.0025 \
SALIENT_ROW_MIN=0 \
SALIENT_ROW_MAX=4 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

- `SALIENT_CALIBRATION_STEPS`: more steps give stabler salience estimates but add export time.
- `SALIENT_ROW_FRACTION`: total rescued rows per matrix.
- `SALIENT_ROW_MAX`: hard cap on rescued rows per matrix to control artifact size.
- `SALIENT_SCORE_POWER`: how strongly calibration RMS reweights row MSE.
- `SUBMISSION_SIZE_LIMIT_BYTES`: decimal artifact cap; the exporter tries smaller rescue budgets before failing if the artifact is still too large.

## Evaluation notes

The script keeps the base record’s normal validation flow:

- post-EMA diagnostic eval,
- int6 roundtrip eval,
- sliding-window eval at `EVAL_STRIDE`,
- optional stride-64 comparison.

The only extra stage is the short calibration pass over training tokens before export.

## Validation run in this workflow

Commands:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603262006_awq-row-rescue/train_gpt.py
python -m compileall candidates/202603262006_awq-row-rescue/train_gpt.py
python - <<'PY'
import importlib.util
print(importlib.util.find_spec("torch"))
PY
python3 - <<'PY'
import importlib.util
print(importlib.util.find_spec("torch"))
PY
```

Outcomes:

- `compileall` succeeded for the root scripts, `data/`, and this candidate script.
- A true local runtime smoke test was **not feasible in this container** because `torch` was not installed for either `python` or `python3`, even though `requirements.txt` lists it. That means I could validate syntax, but not execute the model path locally in this environment.

## Main expected risks and tradeoffs

- **Artifact budget risk**: if the rescue budget is tuned too aggressively, the fp16 residual rows can erase the byte savings from int6 export. This fork now includes an automatic fallback loop, but large manual overrides can still shrink the margin.
- **Calibration mismatch risk**: output RMS on a few training batches may not perfectly identify the rows that matter most for final BPB.
- **Extra export time**: the calibration pass adds a small amount of work after training.
- **Incremental, not structural**: this is a targeted compression experiment, not a full new architecture, so the upside may be smaller than a successful major architectural leap.

## Why this is still meaningfully new

The existing records already cover:

- mixed int6/int8 export,
- late QAT,
- GPTQ-lite clip search,
- EMA/SWA smoothing,
- XSA, partial RoPE, and TTT.

What they do **not** appear to cover is an **activation-aware sparse high-precision residual inside the int6 exporter**. That is the explicit twist this candidate is trying to test.
