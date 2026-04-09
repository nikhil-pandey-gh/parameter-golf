# 202604090311 Hadamard GPTQ-lite

## Hypothesis

Blockwise Walsh-Hadamard rotation before GPTQ-lite int6 export will flatten weight outliers in attention and MLP matrices, reduce post-quantization reconstruction error, and improve roundtrip BPB without adding training-time parameters or materially changing the training stack.

## Why this is promising here

The repository trend is clear: once sliding-window eval landed, most of the remaining gains came from making larger 10L/11L models survive low-bit export better. Recent records explicitly improved BPB through fp16 embedding handling, mixed int5/int6 export, GPTQ-lite clip search, EMA, and other quantization-friendly training choices. This candidate targets that same bottleneck directly instead of introducing a new architecture family.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the direct code base for this candidate because it already has the strongest clean GPTQ-lite export path in-tree and enough artifact headroom to test an export-only change safely.
2. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the current best local result and confirms the same 11-layer family is still the right neighborhood, but its parameter banking, TTT path, and tighter size budget make it a worse place to isolate a new quantizer.
3. `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`, `2026-03-19_WarmdownQuantization/`, and `2026-03-20_10L_Int5MLP_MuonWD04_SWA50/` all reinforced the same lesson: quantization/export quality is often the limiting factor, not pre-export fit.

## External research that informed it

1. **QuaRot** shows fixed orthogonal rotations can remove outliers and make 6-bit/8-bit export nearly lossless in transformers: <https://arxiv.org/abs/2404.00456>
2. **SpinQuant** shows learned rotations outperform naive PTQ and cut the full-precision gap substantially, reinforcing that rotation-aware quantization is a high-leverage direction: <https://arxiv.org/abs/2405.16406>
3. **FlatQuant** shows flatter transformed distributions continue to matter even after earlier scaling/rotation methods: <https://arxiv.org/abs/2410.09426>
4. **PolarQuant** is especially relevant here because its 2026 result reports that Walsh-Hadamard rotation alone accounts for most of the gain in its weight-only pipeline, which makes a lightweight fixed-rotation variant attractive for this repo: <https://arxiv.org/abs/2603.29078>

## What changed versus the chosen base implementation

Relative to the 2026-03-22 GPTQ-lite record:

1. Added `HADAMARD_QUANT`, `HADAMARD_BLOCK_SIZE`, and `HADAMARD_MIN_DIM` knobs.
2. Added a blockwise normalized Walsh-Hadamard transform on the last dimension of eligible 2D attention/MLP weights before GPTQ-lite int6 quantization.
3. Stored transform metadata in the export and inverted the same transform after dequantization, so runtime model semantics stay in the original basis.
4. Left the model architecture, optimizer stack, EMA/SWA behavior, and evaluation path otherwise unchanged to isolate the export change.

This is intentionally closer to a **QuaRot/PolarQuant-lite export path** than a full SpinQuant/FlatQuant implementation; there is no calibration set, learned rotation, or extra training loop.

## How to run or evaluate it

Typical 8-GPU run:

```bash
torchrun --standalone --nproc_per_node=8 \
  candidates/202604090311_hadamard-gptq-lite/train_gpt.py
```

Key candidate-specific knobs:

```bash
HADAMARD_QUANT=1
HADAMARD_BLOCK_SIZE=256
HADAMARD_MIN_DIM=256
```

Disabling the new export path for ablation:

```bash
HADAMARD_QUANT=0 torchrun --standalone --nproc_per_node=8 \
  candidates/202604090311_hadamard-gptq-lite/train_gpt.py
```

## Validation

Commands run for this candidate:

```bash
python -m compileall candidates/202604090311_hadamard-gptq-lite/train_gpt.py
python - <<'PY'
import importlib.util
for name in ("torch", "flash_attn_interface", "sentencepiece", "numpy"):
    print(f"{name}_spec:{bool(importlib.util.find_spec(name))}")
PY
```

- `compileall`: **passed**
- runtime dependency probe: `torch=False`, `flash_attn_interface=False`, `sentencepiece=False`, `numpy=False`
- CPU-only smoke run: **not feasible in this environment** because the script imports those runtime dependencies and then requires CUDA during execution

## Main expected risks and tradeoffs

1. Fixed Hadamard rotation is much cheaper than SpinQuant/FlatQuant, but also weaker; gains may be small if the repo's GPTQ-lite percentile search already captures most of the easy win.
2. Better reconstruction error does not guarantee a smaller compressed artifact; transformed int6 codes may trade BPB for worse zstd compressibility.
3. `HADAMARD_BLOCK_SIZE=256` is a heuristic. Smaller blocks may preserve more local structure while larger blocks may flatten outliers better.
4. This candidate only changes export/dequantization. If the best path ultimately needs rotation-aware QAT or learned affine transforms, this should be treated as the minimal first step, not the endpoint.
