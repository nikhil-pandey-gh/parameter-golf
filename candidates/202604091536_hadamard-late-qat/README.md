# Hadamard-Rotated GPTQ-lite + Real Late QAT

## Hypothesis

The strongest clean base in this repo (`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`) already has most of the winning architecture stack, but the repo history still shows a meaningful post-quantization bottleneck. This candidate targets that bottleneck directly:

1. make the existing late-QAT path actually activate under `torch.compile`, and
2. apply a deterministic block-Hadamard rotation before GPTQ-lite int6 export so row-wise quantization sees flatter, less outlier-heavy matrices.

The bet is that this combination closes more of the remaining int6 roundtrip gap without taking on the extra evaluation/runtime complexity of legal TTT or the engineering complexity of the banked 2026-03-23 stack.

## Why it is promising for this repository

- The records repeatedly show that compression-aware changes matter as much as modeling changes here.
- `2026-03-19_WarmdownQuantization` explicitly frames quantization damage as a dominant bottleneck.
- The 4-hour non-record run reached strong pre-quant quality but still lost badly after export, which is more evidence that better training/export coupling is still under-explored.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` later documented that its Late QAT path was dead code under `torch.compile`, so there is a concrete repo-specific reason to revisit that direction.

There were no prior runs under `candidates/` at the time this candidate was created, so the relevant prior art was entirely in `records/`.

## Prior record influence

- **Chosen base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best non-TTT / non-banked implementation base.
  - Already includes 11L, XSA4, partial RoPE, LN scale, VE, EMA, and GPTQ-lite clip search.
- **`2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`**
  - Important because it explicitly noted that late QAT never activated under `torch.compile`.
- **`2026-03-19_WarmdownQuantization`**
  - Useful framing: training for compressibility, not only raw pre-quant quality.
- **`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`**
  - Shows the current frontier, but I intentionally stayed on the cleaner 03-22 base so the new candidate isolates quantization-path changes.

## External research that informed this candidate

- **QuaRot** — rotation-based quantization can remove outlier structure before low-bit export with no extra learned parameters at inference time: <https://arxiv.org/abs/2404.00456>
- **SpinQuant** — learned rotations outperform naive/random rotations, which suggests there is still real headroom in the repo's currently unrotated GPTQ-lite export path even when we keep the implementation minimal and deterministic: <https://arxiv.org/abs/2405.16406>
- **Training with Quantization Noise for Extreme Model Compression** — supports the idea that exposing the model to quantization noise during training can improve compact-model quality under aggressive compression: <https://arxiv.org/abs/2004.07320>

## What changed vs. the chosen base

The default model architecture and optimizer settings are intentionally unchanged. The edits are focused on the compression path:

1. **Real late QAT activation**
   - The base file toggled a Python/class-level QAT flag after compilation, which the repo has already seen get constant-folded away.
   - This candidate recompiles the training graph once when late QAT turns on, so the fake-int6 branch is actually present in the compiled graph.

2. **Block-Hadamard rotation before int6 GPTQ-lite**
   - Eligible 2D int6 matrices are rotated on both axes with a fixed block-Hadamard transform before the percentile clip search.
   - Minimal rotation metadata is saved and the inverse rotation is applied after dequantization.
   - Defaults: `ROTATE_QUANT_ENABLED=1`, `ROTATE_BLOCK_SIZE=128`.

3. **Self-contained candidate execution**
   - Default dataset/tokenizer paths now resolve relative to the repository root, so this script can be launched from inside the candidate directory.
   - `flash_attn_interface` import now falls back to PyTorch SDPA so the file can at least import on non-challenge environments.

## How to run

From the repository root:

```bash
cd candidates/202604091536_hadamard-late-qat
SEED=1337 \
ROTATE_QUANT_ENABLED=1 \
ROTATE_BLOCK_SIZE=128 \
LATE_QAT_THRESHOLD=0.15 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults assume the cached challenge data lives in:

- `../../data/datasets/fineweb10B_sp1024/`
- `../../data/tokenizers/fineweb_1024_bpe.model`

Override `DATA_PATH` / `TOKENIZER_PATH` if your checkout differs.

## Validation

Commands run for this candidate:

```bash
python -m compileall candidates/202604091536_hadamard-late-qat/train_gpt.py
```

Outcome:

- **Passed** (`exit code 0`).

Attempted CPU smoke check:

```bash
python - <<'PY'
import importlib.util
import pathlib
spec = importlib.util.spec_from_file_location(
    "candidate_train_gpt",
    pathlib.Path("candidates/202604091536_hadamard-late-qat/train_gpt.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PY
```

Outcome:

- **Blocked in this workflow environment** because `torch` is not installed (`ModuleNotFoundError: No module named 'torch'`).
- The candidate file now avoids a hard dependency on `flash_attn_interface` at import time, so once `torch` is available the next cheap smoke should be a small CPU `GPT(...)` forward plus a toy quantize/dequantize roundtrip.

## Main risks and tradeoffs

- **Late-QAT recompile cost:** enabling QAT now triggers a one-time recompilation near the end of training; that should be cheaper than paying fake-quant overhead for the whole run, but it still costs wall-clock.
- **Rotation helps MSE more directly than compressed bytes:** the Hadamard step should improve quantization fidelity, but it may or may not improve the final `int6 + zstd/lzma` artifact size.
- **DDP / compile interaction risk:** recompiling once mid-run is the cleanest way to avoid the dead-code issue, but it is still a systems-level risk that needs a real multi-GPU run.
- **No architecture change:** this candidate is deliberately narrow. If the remaining gap is now mostly model-side rather than export-side, the gain could be small.
