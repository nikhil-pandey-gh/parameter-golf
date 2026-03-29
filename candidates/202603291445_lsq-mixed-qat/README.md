# LSQ-Style Late Mixed-Bit QAT on the EMA + GPTQ-lite Stack

## Hypothesis

The strongest non-TTT record in this repository already gets within striking distance of the current SOTA, and the remaining gap is increasingly dominated by how gracefully the model survives export rather than by raw float32 validation loss alone. The hypothesis here is that **real, learned late-stage QAT** can shrink the post-export loss more reliably than the repository's current "fake quant from row max" path, while also letting the model tolerate a more aggressive **mixed-bit export**:

- **MLP weights** train toward **int5** export,
- **attention weights** train toward **int6** export,
- **embeddings / small tensors** keep the existing safer int8 / fp16 treatment.

## Why this is promising for this repository

Local evidence points in the same direction:

- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` explicitly argues that export quality, not just float quality, is a first-order bottleneck.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` showed that modest quantization-only improvements like GPTQ-lite clip search and slightly earlier late-QAT can still buy meaningful gains.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` demonstrates that even much better-trained models can give back a lot of progress during round-trip compression.

By contrast, several other directions look less attractive for a "next candidate":

- weight-sharing / recurrence has already looked negative under the fixed 10-minute wall-clock budget,
- SwiGLU looked quality-positive but throughput-negative,
- many optimizer and bucket-size sweeps appear close to saturation.

## Prior records that influenced this candidate

- **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best pure training/export stack in the repo without leaning on legal TTT.
  - Already has EMA, GPTQ-lite export, partial RoPE, XSA, VE, BigramHash, and the 11L / 3x MLP shape.

- **Supporting evidence:**
  - `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`
  - `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`

Those runs collectively suggest that the remaining cheap headroom is more likely to come from **compression-aware training and export fidelity** than from another shallow hyperparameter sweep.

## External research that informed it

- **LSQ: Learned Step Size Quantization**  
  Esser et al., 2019  
  <https://arxiv.org/abs/1902.08153>

- **PACT: Parameterized Clipping Activation for Quantized Neural Networks**  
  Choi et al., 2018  
  <https://arxiv.org/abs/1805.06085>

- **GPTQ**  
  Frantar et al., 2022  
  <https://arxiv.org/abs/2210.17323>

- **AWQ**  
  Lin et al., 2023  
  <https://arxiv.org/abs/2306.00978>

- **SmoothQuant**  
  Xiao et al., 2022  
  <https://arxiv.org/abs/2211.10438>

The common lesson from those papers is that post-training quantization alone is often leaving performance on the table, and that **learned scales / clipping or activation-aware equalization** can preserve much more of the full-precision model. This candidate picks the most repo-compatible version of that lesson: **late, row-wise learned step sizes**, folded into the existing training/export pipeline.

## What changed versus the chosen base implementation

This candidate starts from the `2026-03-22` EMA + GPTQ-lite record and makes the following targeted changes in `train_gpt.py`:

1. **Replaces the old global late-QAT switch with per-module learned QAT state.**
   - `CastedLinear` now supports optional row-wise learned quantization scales.
   - Late QAT is enabled per module instead of through a class-global flag.

2. **Uses LSQ-style row-wise fake quantization for large block weights.**
   - Attention projections are assigned **6-bit** fake quantization.
   - MLP `fc` and `proj` weights are assigned **5-bit** fake quantization.
   - The learned scales are present from the start for DDP consistency, but they are only *used* once late QAT turns on.
   - The learned scales are training-only and are excluded from export.

3. **Initializes late-QAT scales from GPTQ-lite-style percentile search.**
   - This keeps the candidate anchored to the repo's strongest existing export trick instead of discarding it.

4. **Exports the trained model with the learned mixed-bit scales when available.**
   - Learned-QAT weights use the same positive / clamped row-scale convention during export that they used during training.
   - EMA state for those scales is seeded when late QAT actually turns on, so export is averaged over the active QAT window rather than over dummy pre-QAT values.
   - Untouched tensors fall back to the existing quantization logic.

5. **Keeps the rest of the proven stack intact.**
   - 11 layers, 512 width, XSA, partial RoPE, VE, BigramHash, EMA, sliding eval, and the strong optimizer defaults remain in place.

## How to run / evaluate

From the candidate directory:

```bash
cd candidates/202603291445_lsq-mixed-qat

SEED=1337 \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
WARMDOWN_ITERS=3500 \
LATE_QAT_THRESHOLD=0.15 \
QAT_BITS_ATTN=6 \
QAT_BITS_MLP=5 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Expected behavior:

- training starts in full precision,
- once LR scale drops below `LATE_QAT_THRESHOLD`, block attention/MLP weights switch into learned fake quantization,
- export uses mixed-bit round-trip quantization (`int5` MLP / `int6` attention / safer fallback elsewhere),
- the final artifact round-trip metrics are logged under `final_artifact_roundtrip_exact`.

## Validation

Commands run in this environment:

```bash
cd /home/runner/work/parameter-golf/parameter-golf
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603291445_lsq-mixed-qat/train_gpt.py
```

Outcome:

- **Passed**. Python syntax compilation succeeded for the repository baselines, `data/` utilities, and the new candidate script.

Environment checks:

```bash
python - <<'PY'
from pathlib import Path
import importlib.util

root = Path('/home/runner/work/parameter-golf/parameter-golf')
print(importlib.util.find_spec('torch'))
print((root / 'data/tokenizers/fineweb_1024_bpe.model').exists())
print((root / 'data/datasets/fineweb10B_sp1024').exists())
PY
```

Outcome:

- `torch` is not importable in this workflow environment.
- The expected tokenizer model is not present locally.
- The expected FineWeb shard directory is not present locally.

Because this candidate script is also **CUDA-only at runtime**, a real startup smoke test was **not feasible here** without the missing runtime dependencies and dataset artifacts.

## Main expected risks / tradeoffs

- **Throughput risk:** learned fake quantization adds extra math in forward passes after the late-QAT threshold.
- **Scale instability:** row-wise learned scales can become noisy if the late-QAT window is too short.
- **Over-aggressive MLP compression:** `int5` MLP export may help size but could hurt quality if the learned scales do not converge cleanly.
- **`torch.compile` interaction risk:** this candidate avoids the previous class-global gate style, but late graph changes are still the part of the stack that deserves the most scrutiny in a real GPU run.

## Suggested next experiments if this helps

1. Sweep `LATE_QAT_THRESHOLD` across `0.10`, `0.15`, `0.20`, `0.25`.
2. Compare `QAT_BITS_MLP=5` vs `6`.
3. Keep learned late QAT, but restore **LeakyReLU(0.5)^2** from the current best record.
4. If mixed-bit QAT helps, try extending learned scales to selected non-block projections that dominate export error.
