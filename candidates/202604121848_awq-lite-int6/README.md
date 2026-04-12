# AWQ-lite Int6 on the current best stack

## Hypothesis

The current best script already squeezes a lot out of training and evaluation, so the highest-ROI next step is to make the **export quantizer activation-aware** instead of choosing int6 clip ranges from weight reconstruction error alone.

This candidate keeps the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` training/eval stack intact and replaces its weight-only GPTQ-lite export heuristic with a short **AWQ-lite calibration pass** that measures channelwise activation energy on train batches and uses those statistics to weight the int6 clip search.

## Why this is promising for this repository

Repository evidence points to export quality being one of the main remaining bottlenecks:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved the training-only stack with a better int6 clip search.
- `2026-03-18_FP16Embed_WD3600` showed that export precision choices alone can materially move post-quant BPB.
- `track_non_record_16mb/2026-03-18_Quasi10Bfrom50B...` shows that even much better-trained weights still lose a lot after export.

That makes a calibration-driven export refinement more attractive than another throughput-costing training change.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` — strongest current overall stack and direct code base.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` — closest prior art because it already improved post-training int6 clipping.
3. `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/` — useful evidence that better training alone does not remove the export bottleneck.

## External research

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (Lin et al., arXiv:2306.00978) motivates using activation statistics, not just weights, to identify salient channels during weight-only quantization.
- **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs** (Ashkboos et al., arXiv:2404.00456) argues that quantization quality is heavily constrained by outlier structure and shows even 6/8-bit regimes can benefit from outlier-aware export.
- **SpinQuant: LLM quantization with learned rotations** (Liu et al., arXiv:2405.16406) reinforces that better outlier handling materially reduces quantization error beyond naive clipping.

This candidate intentionally implements the smallest version that fits the repository: **activation-weighted clip search**, not full rotation learning.

## What changed vs the base implementation

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate adds:

1. **AWQ-lite calibration controls**
   - `AWQ_LITE_ENABLED=1`
   - `AWQ_CALIBRATION_STEPS=8`
   - `AWQ_CALIBRATION_BATCH_TOKENS=131072`
   - `AWQ_IMPORTANCE_POWER=0.5`
   - `AWQ_IMPORTANCE_CLIP=4.0`
2. A short post-training calibration pass over train batches that collects per-channel second moments for:
   - attention Q/K/V inputs,
   - attention output projection inputs,
   - MLP up-projection inputs,
   - MLP down-projection inputs.
3. An **activation-weighted int6 percentile search** for banked attention and MLP weights:
   - existing GPTQ-lite tries several per-row clip percentiles,
   - this candidate scores each candidate with a diagonal activation-weighted reconstruction loss,
   - channels with larger observed activation energy matter more during clip selection.

Everything else is intentionally left as close to the current best script as possible.

## How to run

From the repository root:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
RUN_ID=awq_lite_int6 \
torchrun --standalone --nproc_per_node=8 candidates/202604121848_awq-lite-int6/train_gpt.py
```

This base script keeps the parent record's EMA path always on internally with decay `0.997`, so there is no separate `EMA_ENABLED` flag to set.

The AWQ-lite path is enabled by default. To disable it for an A/B comparison:

```bash
AWQ_LITE_ENABLED=0 torchrun --standalone --nproc_per_node=8 candidates/202604121848_awq-lite-int6/train_gpt.py
```

## Evaluation notes

- The candidate still exports a standalone compressed int6 artifact.
- The calibration pass uses **training** batches only and runs after training, before export.
- The legal score-first TTT path is unchanged.

## Main risks and tradeoffs

- The gain may be small because the baseline is already at int6 with GPTQ-lite.
- This is only an **AWQ-lite** diagonal weighting scheme, not full AWQ/OmniQuant/SpinQuant-style equivalent transforms or learned rotations.
- Calibration adds a small amount of export-time work.
- Importance estimates come from a short training-data sample, so poor calibration coverage could blunt the gain.

## Validation

Commands run:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604121848_awq-lite-int6/train_gpt.py
```

Outcome: **passed**.

Attempted CPU smoke check:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('candidates/202604121848_awq-lite-int6/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.Hyperparameters.awq_lite_enabled)
PY
```

Outcome: **not feasible in this environment** because the local Python runtime is missing repository dependencies (`numpy` was missing at import time), and the full script also expects the CUDA/FlashAttention runtime used by the existing record scripts.
