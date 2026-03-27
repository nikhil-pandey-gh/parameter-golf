# SpinQuant-lite GPTQ on the current TTT record stack

## Hypothesis

The current frontier in this repository is already quantization-limited: several record notes explicitly show that post-training compression quality, not just more training, is the main bottleneck. The hypothesis here is that **deterministic block-Hadamard rotations applied only when they reduce reconstruction error** can further smooth outlier-heavy attention/MLP weight matrices before the existing GPTQ-lite int6 export, improving the final round-trip artifact with **zero training-time cost** and **no runtime cost after load**.

In short: keep the strongest known training stack, but make the export smarter.

## Why this is promising for this repository

Repo history points in the same direction:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` is the current best overall stack, and it still depends on GPTQ-lite int6 export plus legal TTT.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` shows that a better post-training clip search alone was worth another measurable gain.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` shows that longer training can still lose after quantization, which is a strong sign that the export path is the right place to keep pushing.
- Earlier records such as `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` and `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md` also repeatedly benefited from making quantization exceptions and mixed-precision export decisions more carefully.

This candidate follows that evidence: it does not spend more of the 10-minute training budget, and instead tries to recover bits-per-byte during the artifact step.

## Prior records that influenced this candidate

The implementation starts from the strongest current stack in:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

It also directly borrows the export intuition from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` for GPTQ-lite clip search,
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` for quantization-aware artifact thinking,
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/` for the general pattern of using smarter export to buy better effective capacity.

## External research that informed it

This idea is a deliberately lightweight adaptation of recent rotation-based PTQ work:

- **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs** (`https://arxiv.org/abs/2404.00456`) argues that orthogonal rotations can remove hidden-state/weight outliers without changing the model function, making low-bit quantization easier.
- **SpinQuant: LLM quantization with learned rotations** (`https://arxiv.org/abs/2405.16406`) shows that better-chosen rotations can reduce quantization error even further than fixed rotations.
- **ButterflyQuant: Ultra-low-bit LLM Quantization through Learnable Orthogonal Butterfly Transforms** (search result: `https://arxiv.org/search/?query=%22ButterflyQuant%3A+Ultra-low-bit+LLM+Quantization+through+Learnable+Orthogonal+Butterfly+Transforms%22&searchtype=all&source=header`) reinforces the idea that a one-size-fits-all rotation is suboptimal and motivates per-matrix selection.

This candidate does **not** try to reproduce the full learned-rotation pipelines from those papers. Instead, it takes the smallest repo-friendly slice of the idea:

1. use a fixed normalized Hadamard transform in small groups,
2. apply it only to matrices already headed for int6 export,
3. compare identity vs rotated reconstruction error,
4. keep the lower-error option per matrix.

That keeps the code self-contained and preserves the repo's single-file submission style.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Main ML change:

- Added **per-matrix block-Hadamard rotated GPTQ-lite int6 export** for attention and MLP weights.
- The exporter now tries two candidates for each eligible 2D int6 matrix:
  - the original GPTQ-lite percentile clip search,
  - the same search after a normalized group-wise Hadamard rotation along the input dimension.
- The script stores rotation metadata only when the rotated version wins, and applies the inverse transform during round-trip dequantization.

Support changes:

- Added `ROTATE_INT6` and `ROTATE_GROUP_SIZE` environment flags.
- Added an optional FlashAttention fallback to `scaled_dot_product_attention` so the script is easier to smoke-test outside the exact H100 runtime.
- Added `SMOKE_TEST=1` for a tiny CPU-safe-ish code-path that only exercises the new quantization logic and a small forward pass.
- Made `numpy` and `sentencepiece` optional imports for that smoke path; full training still requires the normal repo dependencies.

These support changes are for validation ergonomics and are not the main hypothesis.

## How to run / evaluate

From the candidate directory:

```bash
cd candidates/202603271726_spinquant-lite-gptq
```

Run the same recipe as the current 2026-03-23 record stack, plus the new export flags:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
ROTATE_INT6=1 ROTATE_GROUP_SIZE=128 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For an export ablation, compare against:

```bash
ROTATE_INT6=0
```

## Validation run in this workflow

Commands run:

```bash
python -m compileall candidates/202603271726_spinquant-lite-gptq/train_gpt.py
```

Outcome:

- **Passed**: the candidate script compiled successfully.

Attempted smoke test:

```bash
cd candidates/202603271726_spinquant-lite-gptq && SMOKE_TEST=1 python train_gpt.py
```

Outcome:

- **Could not run in this workflow runner** because both `/usr/bin/python` and `/usr/bin/python3` were missing the repository's required Python packages, including `torch`, `numpy`, and `sentencepiece`.
- The script now avoids requiring `numpy`/`sentencepiece` for `SMOKE_TEST=1`, but `torch` is still required, so the smoke path remains blocked in this environment.

## Main expected risks / tradeoffs

- The gain may be small if GPTQ-lite percentile search already removes most of the same outliers.
- Fixed Hadamard groups are much simpler than learned rotations, so this may capture only part of the SpinQuant/QuaRot upside.
- Export time increases because each eligible matrix now compares identity vs rotated quantization.
- The extra code bytes may slightly eat into artifact headroom, so the final byte count should be checked on a real training run.
- Because the selection is based on reconstruction MSE rather than direct validation loss, some matrices may look better numerically without helping downstream bpb.

## Suggested next experiments

If this helps, the most natural follow-ups are:

1. try `ROTATE_GROUP_SIZE=64` vs `128`,
2. restrict rotation to only MLP or only attention weights,
3. combine rotation with the repo's existing fp16 exceptions for particularly sensitive tensors,
4. try the same exporter on the no-TTT `2026-03-22` GPTQ-lite stack to isolate the export effect before TTT.
