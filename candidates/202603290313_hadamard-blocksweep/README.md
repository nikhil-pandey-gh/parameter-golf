# Block-Hadamard GPTQ-lite Export + LeakyReLU²

## Hypothesis

A deterministic block-Hadamard rotation applied only inside the existing GPTQ-lite-style int6 export path should reduce per-row quantization error on the strongest non-TTT 11-layer stack, improving roundtrip and sliding-window validation BPB without adding learned parameters. I also port the repo-proven `LeakyReLU(0.5)^2` MLP activation because it delivered a measurable gain on the newer leaderboard-topping stack with almost no implementation cost.

## Why this is promising for this repository

The repo's best results already come from squeezing more quality out of the final compressed artifact rather than from wholesale architecture changes. The record progression moved from XSA + EMA to partial RoPE + LN scale, then to GPTQ-lite clip search and longer warmdown, with each step shaving a few more basis points off the quantized eval score.

This candidate stays in that regime:
- it starts from the strongest clean non-TTT base,
- keeps the training architecture intact,
- attacks the same export-time quantization bottleneck that recent records improved,
- and does so without adding persistent model parameters that would pressure the 16MB artifact budget.

## Prior records and candidates that influenced it

There were **no prior `candidates/` directories checked out locally** when this candidate was created, but public workflow history already included related rotation-aware candidate iterations.

The main record influences were:
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest non-TTT base,
  - already integrates the winning 11-layer/XSA/partial-RoPE/EMA/GPTQ-lite stack,
  - has more artifact headroom than the newest TTT-heavy record.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - supplied the `LeakyReLU(0.5)^2` MLP activation tweak,
  - showed that cheap activation changes can still matter on top of a mature stack.

The most relevant prior public candidate iterations were:
- issue `#229`
  - proposed a QuaRot-lite raw-vs-rotated MSE comparison on the same 2026-03-22 base plus LeakyReLU².
- issue `#35`
  - proposed a broader block-rotation export candidate with sign-search, embedding rotation, repo-root-relative paths, and smoke/fallback plumbing.

This candidate's explicit twist versus those earlier ideas is to keep the export path narrower and artifact-budget-aware:
- it searches **block size** (`0,128,256`) rather than adding sign-search machinery,
- it limits the new logic to the **existing int6 attention/MLP exporter** instead of broadening policy across every export category,
- and it keeps the runtime-facing script minimal while still fixing the practical script-path bug needed to run from inside the candidate directory.

## External research that informed it

Two papers most directly motivated the export change:
- **QuaRot** — Croci et al., 2024, <https://arxiv.org/abs/2404.00456>
  - shows that fixed rotations can remove outliers and make low-bit Transformer quantization easier.
- **SpinQuant** — Liu et al., 2024/2025, <https://arxiv.org/abs/2405.16406>
  - shows that rotation choices materially affect post-training quantization quality.

This candidate implements the smallest repo-friendly version of that idea: search over a few deterministic block-Hadamard rotations during int6 export, choose the lowest reconstruction-error option per tensor, and invert the chosen rotation after dequantization.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes:
- Replaced ReLU² in the MLP with **`LeakyReLU(0.5)^2`**.
- Made the default dataset/tokenizer paths **repo-root-relative from `__file__`**, so the script can be launched from inside the candidate directory.
- Added `INT6_ROTATE_BLOCK_SIZES` (default: `0,128,256`).
- Added a deterministic **block-Hadamard transform** helper.
- Extended `quantize_int6_per_row()` to try no rotation vs supported Hadamard block sizes and keep the lowest roundtrip reconstruction error.
- Stored minimal per-tensor rotation metadata in quantization metadata.
- Inverted the selected rotation during `dequantize_mixed_int6()` so evaluation still uses weights in their original orientation.
- Logged per-block-size rotation counts during export for quick inspection.

Notably unchanged:
- training loop,
- optimizer stack,
- EMA/SWA schedule,
- XSA/partial-RoPE/LN-scale/value-embedding setup,
- tokenizer/eval protocol.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603290313_hadamard-blocksweep

RUN_ID=quarot_int6_export \
INT6_ROTATE_BLOCK_SIZES=0,128,256 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:
- `INT6_ROTATE_BLOCK_SIZES=0` disables the new rotation search and falls back to the original export path.
- `INT6_ROTATE_BLOCK_SIZES=0,128` or `0,256` narrows the search if export time becomes a concern.

The main logs to compare are the final quantized metrics:
- `final_int6_roundtrip_exact`
- `final_int6_sliding_window_exact`
- `final_int6_sliding_window_s64_exact`

## Expected risks and tradeoffs

- The search objective is **per-tensor reconstruction MSE**, which is only a proxy for downstream BPB.
- The extra code bytes may matter on the tightest artifact budgets, even though the method adds no learned parameters.
- Export takes slightly longer because each eligible tensor may try multiple rotation candidates.
- The improvement may be strongest on the non-TTT base and may need retuning before stacking with heavier eval-time methods.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202603290313_hadamard-blocksweep/train_gpt.py
python - <<'PY'
import importlib.util
mods = ['torch', 'sentencepiece', 'flash_attn_interface']
for name in mods:
    print(f'{name}={importlib.util.find_spec(name) is not None}')
PY
```

Outcomes:
- `python -m compileall ...` **succeeded**.
- A real CPU runtime smoke test of `train_gpt.py` was **not feasible in this environment** because the runner lacks `torch`, `sentencepiece`, and `flash_attn_interface`, and the script is designed for CUDA/FlashAttention execution.
- Because of that environment limitation, this candidate has syntax validation only here; the first full functional test should be on a CUDA box with the normal challenge dependencies installed.
- A focused code review on the new candidate directory completed **cleanly** with no significant issues reported.
