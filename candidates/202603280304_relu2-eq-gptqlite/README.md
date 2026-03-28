# ReLU² MLP Equalization + GPTQ-lite

## Hypothesis

The strongest no-TTT stack in this repository already trains a good 11-layer model; the remaining gap is increasingly an export problem rather than a pure optimization problem. In this codebase, each `relu²` MLP has an exact hidden-channel symmetry:

- scale `mlp.fc` output rows by `sqrt(s)`
- scale `mlp.proj` input columns by `1 / s`

That leaves the full-precision function unchanged while changing how easy each hidden channel is to quantize. The candidate uses a few post-training calibration batches to estimate which `relu²` channels matter most, searches a small exponent grid, then applies the best per-block equalization before the existing int6 GPTQ-lite export.

## Why this is promising here

- The repo's unlimited-compute non-record baseline reached **1.1749 pre-quant** but only **1.2074 post-quant**, which is a strong sign that export quality is still a major bottleneck.
- The best no-TTT record, `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, improved by using better post-training quantization alone, so another export-side refinement is a natural next step.
- In the current 11L / 3x-MLP regime, the MLP matrices dominate the byte budget. If we can reduce their quantization error without adding parameters, the improvement should be more leverage-efficient than another broad hyperparameter sweep.
- Repository review found no prior `candidates/` directory and no prior experiment implementing activation-aware equalization or rotation-style quantization.

## Prior repository work that informed this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - Best no-TTT base in the repo.
  - Provides the 11L XSA/EMA/partial-RoPE/LN-scale/VE/GPTQ-lite stack.
- **Quantization frontier signal:** `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3`
  - Large pre-quant vs post-quant gap motivated focusing on export quality.
- **Architectural context:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - Showed partial RoPE + LN scale mattered.
- **Current overall frontier:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - Reinforced that the stack is mature enough that small export/eval refinements can still move the needle.

## External research that informed the idea

- **SmoothQuant** (Xiao et al., 2023): training-free equivalent scaling to move quantization difficulty across channels.  
  <https://arxiv.org/abs/2211.10438>
- **AWQ** (Lin et al., 2024): activation-aware protection of salient channels for weight-only low-bit quantization.  
  <https://arxiv.org/abs/2306.00978>
- **QuaRot** (Croci et al., 2024): exact-function rotations that remove outliers and improve low-bit PTQ.  
  <https://arxiv.org/abs/2404.00456>
- **SpinQuant** (Liu et al., 2025): learned rotations for better low-bit quantization than random rotations.  
  <https://arxiv.org/abs/2405.16406>

The longer-term high-upside direction from that literature is probably full rotation-aided quantization. This candidate implements a smaller, repo-friendly first step: use the exact scaling symmetry already present in the repository's `relu²` MLPs instead of introducing a broader residual/attention rotation framework all at once.

## What changed vs the chosen base

- Added a post-training calibration pass that runs a few training batches after EMA has been applied.
- Added per-block `relu²` hidden-channel statistics collection for `mlp.fc`.
- Added a small alpha search over activation-aware hidden-channel rescaling.
- Before mixed int6 quantization, each MLP pair is rewritten as:
  - `fc' = fc * sqrt(s)`
  - `proj' = proj / s`
- Kept the rest of the training architecture, EMA flow, evaluation path, and GPTQ-lite-style int6 export unchanged.
- Left the inherited late-QAT configuration unchanged so the only intended experiment toggle is the new export equalization pass.

## How to run

From this directory:

```bash
RUN_ID=relu2_eq_gptqlite \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The equalization pass is enabled by default. Useful knobs:

```bash
EQ_ENABLED=1
EQ_CALIBRATION_STEPS=8
EQ_BATCH_TOKENS=131072
EQ_MAX_SCALE=4.0
EQ_ALPHAS=0.0,0.25,0.5,0.75,1.0
```

If you want to disable just the new equalization logic while keeping the copied base defaults:

```bash
EQ_ENABLED=0
```

## Expected risks / tradeoffs

- The weighted reconstruction objective is only a proxy for final validation BPB.
- Equalization uses a small number of post-training calibration batches, so the chosen scales may be somewhat batch-sensitive.
- This only targets MLP quantization, not attention outliers or residual-stream outliers. If it helps, the next experiment should extend the same idea to attention or move to a QuaRot/SpinQuant-style transformation.
- Export adds a bit of post-training calibration overhead, so the approach is mainly valuable if the BPB gain beats that complexity.

## Validation

Executed in this workflow:

```bash
python -m compileall candidates/202603280304_relu2-eq-gptqlite/train_gpt.py
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603280304_relu2-eq-gptqlite
```

Outcome: **both commands passed**.

Attempted runtime smoke check:

```bash
PYTHONPATH=/tmp/gh-aw/agent/flash_stub python - <<'PY'
# import the candidate with a temporary FlashAttention stub and run a tiny CPU forward pass
PY
```

Outcome: **not feasible in this workflow environment** because the available Python interpreter does not have `torch` installed, so no local runtime import or CPU forward pass could be completed without unavailable dependency installation.
