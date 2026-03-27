# Candidate: AWQ-lite Block Scales

## Hypothesis

The 11-layer EMA + GPTQ-lite stack is already close to saturating the easy training wins in this repository, so the next meaningful gain is more likely to come from **reducing the post-training quantization gap** than from another large architecture rewrite.

This candidate adds an **AWQ/SmoothQuant-style equivalent channel rescaling pass** to the block MLP and attention linears before int6 export. The idea is to shrink hard-to-quantize weight columns, move the inverse scale into cheap runtime input multipliers, and let the existing GPTQ-lite per-row clip search quantize a flatter set of weights.

## Why this is promising for this repository

Repository history points at the same bottleneck repeatedly:

- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` explicitly argues that post-training quantization damage is larger than many training-side gains.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` and `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` show the current winning non-TTT stack is already mostly settled on 11L + XSA4 + EMA + Partial RoPE + LN scale, with smaller improvements now coming from quantization/export details.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` improves the leaderboard further, but much of that lift comes from evaluation-time TTT rather than a cleaner base artifact.

So this candidate keeps the strong 2026-03-22 non-TTT base and pushes on a seam that the repo has not tried yet: **calibration-based equivalent reparameterization for weight-only quantization**.

## Prior experiments that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Other influential records:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` for the Partial RoPE + LN scale stack.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` for the 11L XSA + EMA template.
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/` for the central insight that quantization quality is often the limiting factor.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` as evidence that the remaining gains are small and compositional, so a precise export-time trick is a more attractive next move than another broad rewrite.

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

This candidate is directly motivated by equivalent-transform quantization papers:

- **SmoothQuant** (Xiao et al., 2023, `arXiv:2211.10438`) shows that channel-wise scaling can migrate difficulty between activations and weights without changing the underlying function.
- **AWQ** (Tang et al., 2024, `arXiv:2306.00978`) shows that activation-aware channel scaling is especially effective for low-bit weight-only quantization.
- **QuaRot** (Croci et al., 2024, `arXiv:2404.00456`) and **SpinQuant** (Liu et al., 2025, `arXiv:2405.16406`) reinforce the broader lesson that equivalent reparameterizations which flatten outliers can materially improve PTQ quality.
- **Primer** (So et al., 2021, `arXiv:2109.08668`) remains relevant as background for the repo's ReLU-squared lineage, though this candidate does not change the activation stack.

This implementation intentionally takes the **smallest practical subset** of those ideas for this codebase: block-local channel scaling, calibrated from a short training-token pass, with no new optimizer, no new kernels, and no new artifact-heavy learned rotation matrices.

## What changed versus the chosen base implementation

Relative to `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. Added persistent per-input-channel `input_scale` buffers to `CastedLinear`.
2. Added an **AWQ-lite block scaling pass** that:
   - calibrates activation RMS on a short training-token pass,
   - computes a per-column scale from activation RMS and weight RMS,
   - divides the stored block weights by that scale,
   - multiplies the runtime inputs by the same scale so the full-precision function stays unchanged.
3. Applied this only to `blocks.*` attention/MLP linears, keeping the change narrow and artifact overhead tiny.
4. Kept the existing **GPTQ-lite per-row clip search** intact, so the new scaling pass composes with the strongest prior export path instead of replacing it.
5. Set `LATE_QAT_THRESHOLD=0.0` by default. Prior records already noted that compile-time late-QAT paths can be fragile/no-op; this candidate focuses on the export-time quantization idea instead of relying on that path.
6. Updated the default dataset/tokenizer paths so the script can be run **from inside this candidate directory** without needing `DATA_PATH`/`TOKENIZER_PATH` overrides.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603271849_awq-lite-blockscales
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults mirror the 2026-03-22 11-layer stack, with the AWQ-lite pass enabled by default. The main new knobs are:

- `AWQ_ENABLED=1`
- `AWQ_ALPHA=0.5`
- `AWQ_MAX_SCALE=4.0`
- `AWQ_CALIB_TOKENS=262144`
- `LATE_QAT_THRESHOLD=0.0`

EMA remains baked into the script exactly as in the base implementation.

If you want the explicit fully-specified command closest to the intended run:

```bash
cd candidates/202603271849_awq-lite-blockscales
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
AWQ_ENABLED=1 AWQ_ALPHA=0.5 AWQ_MAX_SCALE=4.0 AWQ_CALIB_TOKENS=262144 \
LATE_QAT_THRESHOLD=0.0 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Commands run on this runner from the **repository root**:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603271849_awq-lite-blockscales/train_gpt.py
```

Outcome:

- **Passed** for the root training scripts, `data/`, and this candidate `train_gpt.py`.

Attempted runtime smoke check:

- I attempted a tiny CPU-only import/forward smoke test using a temporary FlashAttention stub so the candidate module could be imported without a GPU.
- That **could not proceed on this runner** because its local `python` environment is missing `torch`, `numpy`, and `sentencepiece`, so the script cannot be imported here before any model code runs.
- Because `main()` also requires CUDA, no further runtime smoke check was feasible in this environment without installing heavyweight dependencies that are not already present.

## Main expected risks or tradeoffs

- The scaling heuristic is deliberately simple. It may underperform stronger but heavier methods like learned rotations or full AWQ search.
- Per-row GPTQ-lite quantization already handles many outliers, so the extra rescaling might help only marginally or even hurt some matrices if the heuristic is poorly tuned.
- The candidate adds a small calibration pass at export time and a tiny number of extra control tensors (`input_scale`), so there is some artifact-size and evaluation-time overhead.
- This is intentionally a **non-TTT** candidate. It aims to improve the base artifact itself, not compete with the leaderboard's evaluation-time adaptation tricks.

## Suggested next experiments if this works

- Sweep `AWQ_ALPHA` and `AWQ_MAX_SCALE`.
- Try limiting the scaling pass to only the deepest XSA blocks.
- Combine the same idea with the 2026-03-23 LeakyReLU² activation stack.
- Replace the RMS-based heuristic with a stronger activation-aware saliency rule or a lightweight rotation/permutation pass.
