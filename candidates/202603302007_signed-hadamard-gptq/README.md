# Candidate: Signed-Hadamard GPTQ-lite + LeakyReLU^2

## Hypothesis

The current repo is already close to the point where small compression improvements matter more than big architecture swings. This candidate tests whether **function-preserving per-matrix rotations before GPTQ-lite quantization** can cut int6/int8 round-trip error enough to improve final `val_bpb` under the 16MB artifact cap, while also importing the low-cost **LeakyReLU(0.5)^2** activation win from the current SOTA branch.

Concretely, the candidate keeps the strong 11-layer EMA / Partial-RoPE / XSA / VE / late-QAT stack from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, but changes export-time quantization to auto-select between:

- no rotation,
- plain normalized Hadamard rotation, and
- deterministic signed-Hadamard rotation,

choosing the variant with the lowest reconstruction MSE for each large matrix.

Because the rotation is inverted after dequantization, the change is **function-preserving apart from quantization error** and adds **zero training-time compute**.

## Why this is promising for this repository

Repository history strongly favors compression-aware improvements over broad architectural rewrites:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` gained on the order of `-0.0013 BPB` from better export and averaging choices alone.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` showed that a one-line `LeakyReLU(0.5)^2` MLP change was worth roughly `-0.003 BPB` on a closely related stack.
- The non-record 4-hour run underperformed after quantization, which is strong evidence that **post-training compression quality**, not just pre-quant training loss, is the bottleneck.

This repo is an unusually good fit for Hadamard-style transforms because the dominant hidden size is `512`, a power of two. That makes Walsh-Hadamard mixing cheap and exact, with no new infrastructure and no calibration data.

I intentionally forked the `2026-03-22` record instead of the heavier `2026-03-23` TTT / Parallel-Muon stack because the `03-22` branch is the cleanest training/export base and has materially more artifact headroom for a new serialization experiment.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen base implementation.
  - Supplies the strongest clean GPTQ-lite / EMA / warmdown / late-QAT stack without TTT or parameter banking complexity.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Provides the `LeakyReLU(0.5)^2` activation change.
  - Confirms that small, zero-parameter MLP tweaks can still move the metric near the frontier.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Upstream source for Partial RoPE and layer-wise normalization scaling already inherited by the `03-22` base.

## External research that informed it

- **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456))
  - Shows that function-preserving rotations can remove outliers and make low-bit quantization dramatically easier, including essentially lossless 6-bit / 8-bit cases.
- **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406))
  - Highlights that some rotations are much better than others, motivating an auto-selection step rather than a single fixed transform.
- **OptRot** ([arXiv:2512.24124](https://arxiv.org/abs/2512.24124))
  - Argues that data-free rotations targeting weight outliers can beat both vanilla Hadamard and heavier alternatives for weight quantization.
- **WUSH** ([arXiv:2512.00956](https://arxiv.org/abs/2512.00956))
  - Reinforces the broader theme that transform-based quantization remains a strong direction even relative to modern RTN/GPTQ-style baselines.

This candidate is not a full implementation of those papers. It is a **repo-native adaptation**: deterministic, data-free, serialization-only, and deliberately tiny in scope.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **MLP activation change**
   - `relu(x)^2` -> `leaky_relu(x, 0.5)^2`
   - Pulled directly from the current best record lineage.

2. **Rotation-aware GPTQ-lite export**
   - Added a normalized fast Walsh-Hadamard transform for power-of-two axes.
   - Added deterministic tensor-specific sign masks derived from the tensor name via a framework-independent PRNG.
   - During export, each eligible large matrix now compares:
     - baseline GPTQ-lite quantization,
     - Hadamard-rotated GPTQ-lite quantization,
     - signed-Hadamard GPTQ-lite quantization.
   - The best option by reconstruction MSE is stored.
   - During reload, the inverse transform is applied before `load_state_dict`.

3. **Export diagnostics**
   - The script now logs how many matrices were considered and how many selected each rotation flavor.

## How to run / evaluate

From the repository root:

```bash
cd candidates/202603302007_signed-hadamard-gptq

NUM_LAYERS=11 \
XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate script resolves its default `DATA_PATH` and `TOKENIZER_PATH` from the repository root, so it can be launched directly from this candidate directory without extra path overrides.

Optional ablation knobs introduced by this candidate:

- `ROTATION_MIN_DIM=128` (default)
  - Minimum power-of-two axis length considered for rotation search.

## Validation

I ran the following lightweight validations in this workflow environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603302007_signed-hadamard-gptq/train_gpt.py
```

Outcome:

- **Passed** for the root scripts, `data/`, and the new candidate script.

I also attempted a CPU-only serialization smoke test for the new rotation-aware quantizer by stubbing `flash_attn_interface` and importing the candidate module directly, but this runner does not have `torch` installed. That means a real import-time smoke test is **not feasible in this environment** without changing the repository environment itself.

Attempted command shape:

```bash
python - <<'PY'
# stub flash_attn_interface, import candidate train_gpt.py, and exercise
# mixed_quantize_int6() / dequantize_mixed_int6() on synthetic tensors
PY
```

Outcome:

- **Blocked by environment**: `ModuleNotFoundError: No module named 'torch'`

## Main expected risks / tradeoffs

- The extra rotation search may improve quantization error but slightly worsen compressed payload size on some runs.
- Deterministic signed-Hadamard is only a lightweight proxy for learned or data-aware transforms like SpinQuant / OptRot+.
- Some matrices may prefer no rotation at all; that is why the implementation keeps a no-rotation baseline in the search.
- If this works on the cleaner `03-22` stack, the natural next step is to port the export path onto the `03-23` TTT / Parallel-Muon branch and verify that the artifact still stays under 16MB.
