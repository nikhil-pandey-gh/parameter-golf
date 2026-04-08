# LSQ Late QAT on the 11L EMA + GPTQ-lite Stack

## Hypothesis

This candidate targets the repo's most persistent remaining weakness: the gap between strong pre-quantized models and their final low-bit artifacts. The hypothesis is that a **compile-safe, row-wise LSQ-style late QAT ramp** can shape attention/MLP weights toward their eventual int6 export scales without paying permanent artifact bytes, improving final roundtrip `val_bpb` more reliably than another architecture change.

## Why this is promising here

- The best records in this repo keep improving by attacking **quantization-aware training/export**, not by restarting from a radically different backbone.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly reports that its late-QAT branch was dead-code-eliminated by `torch.compile`, so there is unfinished headroom on this axis.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` shows that better quantization still yields real wins even on the already-strong 11-layer stack.
- There were **no prior `candidates/` directories** in the repository when this candidate was created, so this idea is not duplicating an earlier candidate iteration.

## Prior repo work that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` — chosen base implementation.
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` — motivated the compile-safe rewrite after its README noted the late-QAT path never actually activated.
3. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` — reinforced that the current frontier is now built from incremental improvements stacked on top of the 11L quantization-aware recipe rather than broad architectural resets.

## External research that informed it

- **Esser et al., “Learned Step Size Quantization” (arXiv:1902.08153)** — the core idea here: learn quantizer step sizes instead of relying on fixed late fake-quant scales.
- **Croci et al., “QuaRot” (arXiv:2404.00456)** and **Liu et al., “SpinQuant” (arXiv:2405.16406)** — recent evidence that better low-bit behavior often comes from explicitly shaping weight distributions for quantization, which is why this candidate spends its complexity budget on export-aware training instead of a larger architecture fork.

## What changed vs. the chosen base

This directory starts from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py` and makes four surgical changes:

1. **Compile-safe LSQ fake quantization for attention/MLP linears.**
   - Each `CastedLinear` used by attention/MLP blocks gets a learned per-row `qat_scale`.
   - Fake quantization is always traceable by `torch.compile`; activation is controlled by a mutable tensor buffer (`qat_strength`) instead of a Python/class flag.
2. **Late-QAT ramp instead of an abrupt on/off switch.**
   - `LSQ_START_SCALE` begins the ramp during warmdown.
   - `LSQ_FULL_SCALE` reaches full strength near the end of training.
3. **Training-only LSQ parameters are excluded from export.**
   - Learned `qat_scale` tensors influence training and final quantization but are not serialized into the artifact.
4. **Final int6 export reuses the learned LSQ row scales for attention/MLP weights.**
   - Non-LSQ tensors still fall back to the existing GPTQ-lite/int8 export path.

I also changed the default `DATA_PATH` and `TOKENIZER_PATH` resolution so the script can be run directly from this candidate directory without having to override paths when using the repository's standard `data/` layout.

## How to run

From this candidate directory:

```bash
cd candidates/202604081629_lsq-late-qat

SEED=1337 \
QAT_ENABLED=1 \
LSQ_START_SCALE=0.22 \
LSQ_FULL_SCALE=0.08 \
LSQ_CLIP_RANGE=31 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If your data/tokenizer live somewhere else, override `DATA_PATH` and `TOKENIZER_PATH` explicitly. Otherwise the defaults resolve back to the repository root's `data/` directory.

## Expected risks and tradeoffs

- **Step-time risk:** LSQ fake quantization adds extra per-step math, so a win on quantization quality could be partially offset by fewer steps inside the 600s wallclock cap.
- **Schedule sensitivity:** if the LSQ ramp starts too early, the model may over-optimize for the proxy quantized objective before the trunk has converged.
- **Exporter coupling:** this candidate intentionally couples late-QAT scales to the final int6 export path. If that coupling is too rigid, a hybrid “LSQ initialization + GPTQ-lite percentile search” export may work better.

## Validation

Commands run:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604081629_lsq-late-qat/train_gpt.py
```

Outcome:

- **Passed** for the repository baseline files and this candidate's `train_gpt.py`.
- A minimal CPU smoke run was **not feasible** in this environment because the training script still requires CUDA and FlashAttention 3 at runtime, so it would fail before reaching a meaningful training step on CPU-only hardware.
