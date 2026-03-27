# Candidate: Sign-Searched Hadamard GPTQ-lite

## Hypothesis

Take the strongest clean pre-TTT stack already in the repo, then reduce the remaining quantization loss with a **deterministic block-Hadamard export rotation** chosen per matrix only when it improves post-quant reconstruction error. The training-side model stays unchanged at inference time because the rotation is inverted after dequantization, so the added complexity lives almost entirely in export math.

I also carry over the extremely cheap **LeakyReLU(0.5)^2** MLP activation that the current top record reports as a strong low-risk gain.

## Why this is promising for this repository

The repo's best 11-layer stacks already look close to saturated on the usual knobs: EMA, XSA, partial RoPE, LN scale, BigramHash, and tighter quantization heuristics. What still matters is the gap between the trained float model and the serialized artifact. Rotation-based PTQ papers report that orthogonal transforms can remove weight outliers and make low-bit quantization meaningfully more faithful without changing the represented model.

That is a good fit for Parameter Golf because:

- it targets the exact bottleneck this repo already optimizes aggressively: post-training compression quality,
- it adds almost no training-time overhead,
- it reuses the repository's existing per-row GPTQ-lite/int6 export path instead of requiring new infrastructure,
- it keeps runtime math identical after load by undoing the rotation before evaluation.

## Prior repository work that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Strongest non-TTT clean stack here: 11L/512d, XSA4, EMA, GPTQ-lite clip search, VE, BigramHash, partial RoPE, LN scale, warmdown 3500.
  - This candidate uses that record as the implementation base.

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Reported that `LeakyReLU(0.5)^2` gave a meaningful gain for almost no code complexity.
  - I ported that activation into the simpler 2026-03-22-style code path.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Confirms the value of partial RoPE + LN scale and also documents that one late-QAT path was accidentally constant-folded away.
  - This further pushes the candidate toward safer post-training export improvements rather than another large QAT branch.

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
  - Notes that full or late QAT had poor speed/benefit tradeoffs in earlier stacks.

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - Documents recurrence as a poor fit for fixed-wallclock throughput, another reason to prefer export-side improvements here.

## External research that informed it

- **QuaRot** — Croci et al., 2024. `https://arxiv.org/abs/2404.00456`
  - Shows that orthogonal rotations can remove outliers and substantially improve quantization with no change to the represented transformer.

- **SpinQuant** — Liu et al., 2024/2025. `https://arxiv.org/abs/2405.16406`
  - Shows that some rotations are much better than others and motivates searching over a small family of rotations rather than using only one fixed transform.

- **QuIP#** — Tseng et al., 2024. `https://arxiv.org/abs/2402.04396`
  - Highlights randomized Hadamard transforms as a fast and theoretically grounded way to make weights more incoherent before PTQ.

- Background references on export-time scaling/quantization tradeoffs:
  - **SmoothQuant** — Xiao et al., 2022/2024. `https://arxiv.org/abs/2211.10438`
  - **AWQ** — Lin et al., 2023/2024. `https://arxiv.org/abs/2306.00978`

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. Added **deterministic sign-searched block-Hadamard rotations** during export quantization.
   - For each eligible 2D weight matrix, the script compares:
     - standard quantization,
     - plain block-Hadamard rotation,
     - several sign-flipped block-Hadamard variants.
   - It keeps the rotated representation only when it reduces reconstruction MSE after quantization.
   - On load, the chosen rotation is inverted before evaluation, so the executed model remains in the original basis.

2. Applied the same rotation-aware logic to both:
   - mixed int6 export for attention/MLP matrices,
   - int8 export for other large matrices.

3. Swapped the MLP activation from `relu^2` to **`LeakyReLU(0.5)^2`**.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202603270037_hadamard-gptqlite

RUN_ID=hadamard_gptqlite \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
python3 -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

Optional export knobs:

```bash
QUANT_ROTATION_ENABLED=1
QUANT_ROTATION_MAX_BLOCK=512
QUANT_ROTATION_MIN_BLOCK=64
QUANT_ROTATION_TRIALS=4
```

Notes:

- EMA remains enabled in the copied 2026-03-22-style base path and still uses its built-in `0.997` decay.
- Late fake-quantization is triggered automatically when the LR scale falls below `LATE_QAT_THRESHOLD`; there is no separate `LATE_QAT=1` switch in this script.

## Main expected risks / tradeoffs

- The benefit may be modest if the strongest matrices are already close to saturation under GPTQ-lite clip search.
- Export time will increase because several rotation candidates are evaluated per eligible matrix.
- Some matrices may prefer no rotation at all; that is why the implementation keeps the unrotated baseline when it wins.
- This is still a heuristic approximation of the stronger learned-rotation literature (e.g. SpinQuant), not a full learned-rotation pipeline.

## Validation

Commands run in this environment:

- `python3 -m compileall candidates/202603270037_hadamard-gptqlite/train_gpt.py`
- `python3 - <<'PY' ... importlib.util.find_spec(...) ... PY` to check whether the local Python environment could import `torch`, `flash_attn_interface`, `sentencepiece`, and `numpy` for a smoke run.

Outcome:

- `compileall` succeeded.
- A runtime smoke test was **not feasible in this container** because `python3` here cannot import `torch`, `flash_attn_interface`, `sentencepiece`, or `numpy`, so the candidate could not be exercised honestly beyond syntax compilation.

CPU-only runtime smoke test:

- Not feasible in this environment for the reason above. The first real smoke run should happen in a Python environment that matches the repository's CUDA/FlashAttention training stack, ideally with the smallest dataset and single-process settings before any 8×H100 run.
