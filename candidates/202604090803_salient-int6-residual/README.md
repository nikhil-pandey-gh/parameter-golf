# 202604090803_salient-int6-residual

## Hypothesis

The current 11-layer record stack is already strong enough that another large architectural rewrite is more likely to burn time than win. A better next bet is to reduce the **post-quantization gap**: keep the best 03-22 training backbone, preserve the tiny set of activation-salient columns that matter most for int6 error, and quantize the rest with the existing GPTQ-lite per-row clip search.

## Why this is promising for this repository

- The repo history shows that once sliding-window eval landed, many of the biggest gains came from **better compression/export**, not from changing the tokenizer or adding much more model complexity.
- The non-record 4-hour run still ended up bottlenecked by quantization, which suggests there is still real headroom in the export path.
- This idea is cheap in code and bytes: it adds a short calibration pass plus a tiny fp16 residual payload instead of a whole second model path.

## Prior records and candidates that informed this

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Activation borrowed from the current SOTA stack:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Structural backbone carried forward:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Negative-result guidance:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` showed simple recurrence was not a good fixed-wallclock trade here, so this candidate stays in the proven 11-layer lane.

There were no existing prior experiments under `candidates/` when this candidate was created.

## External research that informed it

1. **AWQ** — *Activation-aware Weight Quantization for LLM Compression and Acceleration* (`arXiv:2306.00978`)  
   https://arxiv.org/abs/2306.00978
2. **SmoothQuant** — *Accurate and Efficient Post-Training Quantization for Large Language Models* (`arXiv:2211.10438`)  
   https://arxiv.org/abs/2211.10438
3. **pQuant** — *Towards Effective Low-Bit Language Models via Decoupled Linear Quantization-Aware Training* (`arXiv:2602.22592`)  
   https://arxiv.org/abs/2602.22592

The direct takeaway used here is:

- use **activation statistics** to identify the sensitive parts of each linear layer,
- keep only a **small high-precision residual branch** for those sensitive parts,
- leave the bulk of the matrix in the repo's existing GPTQ-lite int6 path.

I also considered recent stronger departures such as latent-attention bottlenecks and shared-depth recurrent blocks, but they were much higher-risk for a single self-contained candidate script.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes three targeted changes:

1. **LeakyReLU(0.5)^2 MLP activation**
   - Replaces `relu(x)^2` with `leaky_relu(x, 0.5)^2`, following the top 03-23 record's strongest low-code training tweak.

2. **Activation-aware calibration pass before export**
   - Runs a short no-grad pass over training shards to accumulate per-input-channel RMS activations for each `CastedLinear`.

3. **Salient-column residual int6 export**
   - For each int6-eligible 2D weight matrix, pick a tiny number of activation-salient columns.
   - Store those columns in fp16 as a residual payload.
   - Zero them in the low-bit path and quantize the remaining matrix with the same GPTQ-lite per-row percentile search as the base.
   - Reconstruct by dequantizing the base matrix and restoring the preserved columns from the fp16 residual payload.

I intentionally leave **late QAT disabled by default** in this candidate so the result isolates the export-side idea instead of depending on a compile-sensitive toggle path.

## How to run or evaluate it

From the repository root:

```bash
RUN_ID=salient_int6_residual \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SALIENT_RATIO=0.004 \
SALIENT_CALIB_STEPS=8 \
SALIENT_MAX_CHANNELS=8 \
torchrun --standalone --nproc_per_node=8 \
  candidates/202604090803_salient-int6-residual/train_gpt.py
```

Useful knobs:

- `SALIENT_RATIO` controls how many columns per eligible matrix are preserved.
- `SALIENT_CALIB_STEPS` controls the cost/quality tradeoff of activation collection.
- `SALIENT_MIN_CHANNELS` and `SALIENT_MAX_CHANNELS` cap per-matrix residual size.
- `LEAKY_RELU_SLOPE` can be swept if `0.5` is too aggressive for this stack.

## Validation

Commands run for this candidate:

```bash
python -m compileall candidates/202604090803_salient-int6-residual/train_gpt.py
```

Outcome:

- `compileall` succeeded.

Attempted lightweight smoke validation:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('candidates/202604090803_salient-int6-residual/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

Outcome:

- blocked by the runner environment missing repository Python dependencies (`numpy`, `torch`, `sentencepiece`).
- a true CPU-only start test is not feasible here anyway because this script also hard-requires CUDA plus `flash_attn_interface`.

## Main risks and tradeoffs

- **Artifact-size risk:** preserving too many columns can erase the int6 size win.
- **Calibration overfit risk:** the activation statistics come from a tiny train-shard sample; the chosen columns may not be globally optimal.
- **Marginal-return risk:** GPTQ-lite is already strong, so the residual branch may buy only a small improvement.
- **Post-train overhead:** calibration adds export-time work, even though it does not change training throughput.

## Expected next experiments

1. Sweep `SALIENT_RATIO` in the `0.002-0.010` range and compare size vs roundtrip/sliding BPB.
2. Restrict the residual branch to late layers only if payload grows too quickly.
3. Combine this export path with a compile-safe late-QAT implementation if the residual branch reduces enough error to justify extra training-time complexity.
