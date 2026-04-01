# Candidate: LeakyReLU^2 + smooth late-QAT ramp

## Hypothesis

The strongest clean non-TTT stack in this repo already has very strong architecture and export settings, but its late QAT path was effectively inert in earlier partial-RoPE/XSA records because `torch.compile(..., fullgraph=True)` could specialize on a Python-side boolean. This candidate keeps that 11-layer GPTQ-lite/EMA/XSA/partial-RoPE backbone, ports in the proven `LeakyReLU(0.5)^2` MLP activation from the current best overall record, and replaces the brittle late-QAT toggle with a smooth tensor-backed ramp that should actually inject low-bit weight noise during the final training phase.

## Why this is promising for this repository

This challenge is dominated by the post-training quantization gap, not just fp16 validation loss. Recent repo history repeatedly shows that compression-aware tweaks such as fp16 embedding passthrough, int6 export, GPTQ-lite clip search, EMA, SWA, and evaluation-aware changes have produced the biggest improvements.

Recent external work points in the same direction:

- **SiLQ: Simple Large Language Model Quantization-Aware Training** ([arXiv:2507.16933](https://arxiv.org/abs/2507.16933)) reports that a simple end-to-end QAT recipe can outperform more elaborate quantization methods with less than 0.1% extra training budget.
- **EfficientQAT** ([arXiv:2407.11062](https://arxiv.org/abs/2407.11062)) argues that making the real low-bit path active during optimization matters, especially when quantization parameters interact with the rest of the network.
- **Scaling Law for Quantization-Aware Training** ([arXiv:2505.14302](https://arxiv.org/abs/2505.14302)) finds that weight quantization error becomes increasingly important as training-token count grows, which is directly relevant to the repo's very high-token 600s training setup.
- **DAQ** ([arXiv:2410.12187](https://arxiv.org/abs/2410.12187)) highlights that low-bit weight-only quantization is sensitive to outlier handling and dynamic range alignment, which motivates using a gradual ramp instead of an abrupt fake-quant switch.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen base implementation. It is the strongest clean non-TTT stack: 11L, XSA on the last 4 layers, EMA, GPTQ-lite int6 export, warmdown 3500, partial RoPE, LN scale, SmearGate, BigramHash, VE.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Supplies the `LeakyReLU(0.5)^2` activation that the top record attributes as a meaningful gain.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Important negative result / implementation lesson: the README explicitly notes that the late-QAT path did not actually activate under `torch.compile`, so a working replacement is worthwhile.

There were no prior `candidates/` directories in the repo when this candidate was created.

## What changed vs. the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes two targeted changes:

1. **LeakyReLU(0.5)^2 MLP activation**
   - Base: `relu(x)^2`
   - Candidate: `leaky_relu(x, negative_slope=0.5)^2`
   - Motivation: this already helped the repo's current best record, and it is cheap in both code size and runtime.

2. **Smooth tensor-backed late QAT ramp**
   - Base: class-level boolean `_qat_enabled` toggled once late in training.
   - Candidate: every `CastedLinear` carries a tiny tensor buffer `qat_mix`, and the training loop updates it smoothly from `0.0 -> 1.0` once `scale < LATE_QAT_THRESHOLD`.
   - Motivation: this avoids relying on a Python boolean that `torch.compile` can constant-fold, while keeping the implementation minimal and accelerator-friendly.

Concretely, when the LR/warmdown scale falls below `LATE_QAT_THRESHOLD`, the script now linearly ramps the fake-quant blend factor instead of flipping it from off to on in one step.

## How to run

Run from this candidate directory on the same 8xH100-style setup used by the records:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- This candidate resolves the default dataset and tokenizer paths relative to the repository root, so running from this candidate directory works without extra `DATA_PATH` / `TOKENIZER_PATH` overrides.
- `QAT_ENABLED=1` still forces full fake-quant training from the start.
- With the default `QAT_ENABLED=0`, the smooth ramp is controlled by `LATE_QAT_THRESHOLD`.
- The activation change is hardcoded in this candidate because it is the main hypothesis, not an ablation knob.

## Evaluation expectations

Expected upside:

- better alignment between training-time weights and the final int6 export path,
- a smaller post-training quantization gap,
- a chance that the LeakyReLU^2 gain and working late-QAT gain stack constructively.

Expected tradeoffs / risks:

- once the ramp starts, cached fake-quant weights are refreshed once per optimizer step, so step time may still tick up slightly during the late phase;
- the training-time fake quantizer still uses simple per-row absmax scaling, while export uses GPTQ-lite clip search, so the train/export quantizers are not perfectly matched;
- LeakyReLU^2 and smooth late-QAT may not combine as cleanly as they did independently;
- because the top clean stack was already strong, gains may be small even if the hypothesis is directionally right.

## Validation run for this candidate

Commands run locally in this workflow:

```bash
python -m compileall candidates/202604010735_leaky-rampqat/train_gpt.py
```

Outcome:

- `python -m compileall` **succeeded**.

CPU smoke test status:

- Not run. This trainer hard-depends on CUDA/FlashAttention (`flash_attn_interface.flash_attn_func`), NCCL-style distributed setup, and the repository's FineWeb/tokenizer assets. The repo does not provide an existing CPU fallback for this script, so I did not add new infrastructure just for smoke validation.
