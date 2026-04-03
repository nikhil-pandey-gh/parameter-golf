# LSQ-lite Endgame on the March 22 XSA/EMA/GPTQ-lite Stack

## Hypothesis

The strongest training-only stack in this repository already does a good job on model quality, but the repo history shows that **weight quantization error is still a first-order bottleneck**. A compile-safe, late-training **learned step-size / learned clip** phase should make the final int6 export easier with almost no extra artifact cost, because the quantizer parameters are used only during training and export selection.

## Why this is promising here

- The repo repeatedly improved by attacking the quantization gap directly: FP16 tied embeddings, mixed int6/int8 export, int6 QAT, and GPTQ-lite clip search all helped.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` notes that its late-QAT branch was effectively dead because `torch.compile` constant-folded the class-level enable flag.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` is the best local **training-only** stack, so it is the cleanest high-signal base for a quantization-focused follow-up.
- The overall record (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`) wins with evaluation-time TTT, but that mixes training and eval innovations. This candidate stays focused on the training/export path.

## Prior records that influenced this candidate

- `2026-03-19_WarmdownQuantization`: framed post-training quantization damage as a dominant bottleneck.
- `2026-03-19_MixedQuant_Int6Int8_SlidingWindow` and `2026-03-19_MLP3x_QAT_Int6_SlidingWindow`: showed that explicit quantization-aware training and mixed low-bit export can materially help.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: exposed the compile-folding failure mode for late QAT.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: chosen base implementation.

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

- **LSQ** (Esser et al., 2020, arXiv:1902.08153): learn quantizer step sizes jointly with model parameters rather than fixing them.
- **EfficientQAT** (Chen et al., 2025 camera-ready, arXiv:2407.11062): highlights a two-stage QAT recipe where end-to-end training of quantization parameters is especially valuable late in the process.
- **Scaling Law for Quantization-Aware Training** (Chen et al., 2025, arXiv:2505.14302): reports that weight quantization error grows more important as training tokens increase, which matches this challenge's long-token / fixed-artifact setting.
- **QuIP#** (Tseng et al., 2024, arXiv:2402.04396): reinforces that better clipping / quantization structure can materially improve low-bit PTQ, even though its full Hadamard/vector-quantized pipeline would be too large a departure for this repo.

## What changed vs. the March 22 base

1. **Compile-safe late QAT enablement**
   - Replaced the class-level `_qat_enabled` toggle with a per-module `qat_mix` buffer so the compiled graph can actually see the switch.
2. **LSQ-lite learned row-wise clip multipliers**
   - Every `CastedLinear` now carries a train-only `qat_row_log_scale` parameter.
   - Fake quantization uses a row-wise scale derived from `amax(row) * exp(qat_row_log_scale)`.
   - Added lightweight LSQ-style gradient scaling on the quantizer parameters.
3. **Dedicated optimizer treatment for quantizer parameters**
   - Quantizer parameters are optimized in their own no-weight-decay AdamW group.
4. **Export uses learned clip proposals**
   - GPTQ-lite percentile search now also evaluates the learned clip candidate for each row.
5. **Artifact stays clean**
   - Training-only `qat_row_log_scale` parameters are excluded from the exported model artifact.
   - Roundtrip eval explicitly allows only those missing keys when reloading the dequantized model.

## How to run

From this candidate directory:

```bash
RUN_ID=lsq_lite_endgame \
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 QAT_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
LATE_QAT_THRESHOLD=0.15 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still performs end-of-run EMA application, GPTQ-lite-style int6 export, roundtrip reloading, and validation scoring.

## Validation run for this candidate

Executed in this repository:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202604031546_lsq-lite-endgame/train_gpt.py
python - <<'PY'
from pathlib import Path
path = Path("candidates/202604031546_lsq-lite-endgame/train_gpt.py")
compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("static_parse_ok", path)
PY
```

Outcomes:

- `compileall` on the existing root scripts and `data/` succeeded.
- `compileall` on this candidate script succeeded.
- A direct Python `compile(...)` parse of the candidate script succeeded.
- A deeper CPU import smoke was **not feasible in this container** because `torch` is not installed here, and the real training path also expects CUDA/runtime deps.

## Main risks and tradeoffs

- The fake-quant path now does more work during training, especially late in warmdown.
- Learned clip multipliers can overfit to the fake-quant surrogate and still fail to beat plain GPTQ-lite export.
- Because the candidate intentionally stays close to the March 22 stack, the upside is likely incremental rather than architectural-step-change upside.
- The no-artifact-cost training-only quantizer parameters are useful only if their learned clip proposal transfers well to the final export search.
