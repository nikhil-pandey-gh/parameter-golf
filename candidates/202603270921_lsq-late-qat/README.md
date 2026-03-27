# LSQ-style late QAT on the GPTQ-lite 11L stack

## Hypothesis

The strongest near-term gain in this repository is to make late-stage quantization-aware training actually influence the main int6-bound weights, then give that late QAT a slightly richer quantizer than the existing row-max fake-quant path.

This candidate forks the strong `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack and replaces its dormant compile-folded late-QAT toggle with a compile-safe, LSQ-inspired late-QAT blend on every `CastedLinear` attention/MLP projection. The expectation is a smaller pre-export quantization gap at almost no artifact-cost increase.

## Why this is promising for this repository

Repository evidence points at quantization error as the main remaining bottleneck:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly says the intended late-QAT path never activated because `torch.compile` constant-folded the class-level QAT flag.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` shows GPTQ-lite clip search, EMA, and a slightly earlier QAT threshold stacked into a further gain.
- The best records repeatedly improve by shrinking the quantization gap rather than changing the core 11-layer shape.

So the missing lever is not “add another big subsystem”, but “make the already-motivated late QAT real, and make its learned step size slightly smarter than row-max only”.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`

Additional influences:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` for the late-QAT bug diagnosis.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` for the stable 11L/XSA/EMA recipe that the later record extends.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` as evidence that later improvements are still coming from quantization- and evaluation-aware refinements rather than wholesale architecture changes.

## External research that informed it

- Steven K. Esser et al., _Learned Step Size Quantization_, ICLR 2020. <https://arxiv.org/abs/1902.08153>
- Elias Frantar et al., _GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers_, ICLR 2023. <https://arxiv.org/abs/2210.17323>
- Mengzhao Chen et al., _Scaling Law for Quantization-Aware Training_, 2025. <https://arxiv.org/abs/2505.14302>

How they map to this candidate:

- LSQ motivates learning quantizer step sizes instead of freezing them to a simple heuristic.
- GPTQ motivates keeping the strong post-training quantizer already validated by this repo and improving the weights it receives.
- The QAT scaling-law paper strengthens the repository's own observation that weight quantization error remains important late in training, especially under aggressive low-bit regimes.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. `CastedLinear` now owns a per-output-channel `lsq_log_scale` parameter.
2. Fake quantization is no longer a dead class-boolean branch. Instead, every `CastedLinear` computes an LSQ-inspired int6 proxy and mixes it in with a runtime `qat_blend` scalar.
3. Late QAT is now a **ramp** controlled by `LATE_QAT_THRESHOLD` and `QAT_RAMP_SPAN`, rather than a one-time boolean flip that can be constant-folded away.
4. The new late-QAT control path is compile-safe because it uses arithmetic blending instead of swapping Python control flow after the graph has already been traced.
5. Default `DATA_PATH` and `TOKENIZER_PATH` resolve from the repository root, so the script can be launched from this candidate directory without patching paths.
6. A PyTorch SDPA fallback is included for environments that do not have `flash_attn_interface`, which makes local debugging less brittle.
7. `torch.compile` is now gated through `USE_TORCH_COMPILE`, so a debugging run can disable it without editing code.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603270921_lsq-late-qat
RUN_ID=lsq-late-qat \
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
LATE_QAT_THRESHOLD=0.15 QAT_RAMP_SPAN=0.15 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
USE_TORCH_COMPILE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

EMA remains enabled at the same fixed `0.997` decay used by the base record.

Useful debug toggles:

- `USE_TORCH_COMPILE=0` to force eager execution.
- `QAT_ENABLED=1` to start with full QAT blend immediately.
- `QAT_RAMP_SPAN=0.05` to make the late-QAT transition sharper.

## Main expected risks or tradeoffs

- The LSQ-style path adds fake-quant math to every training forward pass, even when the blend is still zero. The arithmetic overhead may reduce step throughput.
- Learned scales are small, but they do add a little optimizer state and artifact size.
- This candidate intentionally changes only the quantization path, so if the repo's current best gains mostly come from evaluation tricks like legal TTT, the improvement ceiling here may still be modest.
- The PyTorch-compile interaction is better-behaved than the previous dead-branch design, but the exact throughput/graph behavior still needs a real GPU run.

## Validation

Commands run in this environment:

```bash
python3 -m compileall candidates/202603270921_lsq-late-qat/train_gpt.py
python3 -m compileall train_gpt.py train_gpt_mlx.py data
python3 - <<'PY'
mods = ['torch', 'sentencepiece']
for mod in mods:
    try:
        __import__(mod)
        print(f'{mod}:ok')
    except Exception as exc:
        print(f'{mod}:missing:{exc.__class__.__name__}:{exc}')
PY
```

Observed outcomes:

- `python3 -m compileall candidates/202603270921_lsq-late-qat/train_gpt.py` succeeded.
- `python3 -m compileall train_gpt.py train_gpt_mlx.py data` succeeded.
- A real import/run smoke test was **not feasible in this container** because runtime dependencies are missing here: `torch` and `sentencepiece` are not installed, so the script cannot be executed end-to-end in this environment.
