# LSQ-lite Int6 Late-QAT on the 11L EMA/GPTQ-lite Stack

## Hypothesis

The repo's strongest recent gains have come from making the exported model quantize better, not from large new architectural departures. This candidate replaces the fixed late-QAT path in the 11-layer EMA/GPTQ-lite stack with a **compile-safe, learned-clip int6 fake-quantizer** so the model can spend the last warmdown phase adapting directly to the same clipping rule used at export time.

## Why this is promising here

- Recent records repeatedly improved by reducing the post-training quantization gap: mixed int6/int8 export, fp16/int8-protected embeddings, EMA, GPTQ-lite clip search, and late fake-quantization all helped.
- The repo also has a documented failure mode where late QAT can be optimized away by `torch.compile` when it is toggled with a static class flag. This candidate addresses that directly by **recompiling once when late QAT turns on**.
- External research points the same way: **LSQ** and **PACT** show that learning quantizer scales/clips during training is more effective than using fixed hand-tuned scales, while **GPTQ** and **AWQ** reinforce that export quality is highly sensitive to how clipping/scaling is chosen.

## Influential prior experiments

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best pre-TTT training/export stack to build from
  - showed GPTQ-lite clip search and EMA were still worth ~0.001 BPB
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - documented the `torch.compile` late-QAT dead-code issue explicitly
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the low-risk `LeakyReLU(0.5)^2` MLP activation change
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful negative evidence that naive recurrence/weight sharing is high-risk under a strict wallclock budget, which is why this candidate stays in the compression-aware lane instead

No prior `candidates/` directory existed when this candidate was created.

## External research that informed this candidate

- **LSQ — Learned Step Size Quantization** (Esser et al., 2020): <https://arxiv.org/abs/1902.08153>
- **PACT — Parameterized Clipping Activation** (Choi et al., 2018): <https://arxiv.org/abs/1805.06085>
- **GPTQ** (Frantar et al., 2023): <https://arxiv.org/abs/2210.17323>
- **AWQ — Activation-aware Weight Quantization** (Lin et al., 2024): <https://arxiv.org/abs/2306.00978>

The implementation here is deliberately **LSQ-lite** rather than a full research reproduction: each large `CastedLinear` learns a bounded clip ratio that scales per-row absmax during late QAT, and export reuses that learned ratio instead of doing only fixed percentile search.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes:

1. **LSQ-lite learned clipping**
   - every `CastedLinear` now owns a scalar `lsq_clip_logit`
   - the logit maps to a bounded clip ratio (`LSQ_CLIP_MIN..LSQ_CLIP_MAX`)
   - late fake-quant uses STE through a clipped/scaled int6 quantizer instead of a fixed row-max quantizer
   - fake-quant is applied only to modules that are exported through the learned int6 path, not to train-only heads
2. **Compile-safe late-QAT**
   - when LR scale crosses `LATE_QAT_THRESHOLD`, the script enables QAT and recompiles the training model once so `torch.compile` cannot keep an old no-QAT graph alive
   - on multi-GPU runs, the enable decision is synchronized across ranks before recompilation
3. **Export uses learned clip ratios**
   - int6 export reuses each weight matrix's learned clip ratio when available
   - if no learned ratio exists, the old GPTQ-lite percentile-search path remains as a fallback
4. **LeakyReLU(0.5)^2**
   - swaps the MLP activation from `relu^2` to the top-record `leaky_relu(0.5)^2`
5. **Portable smoke path**
   - if dependencies are installed, `SMOKE_TEST=1` runs a tiny synthetic forward/backward pass without dataset files
   - attention falls back to PyTorch SDPA when `flash_attn_interface` is unavailable, and math SDPA stays enabled in that case

## How to run

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
LATE_QAT_THRESHOLD=0.18 LSQ_CLIP_MIN=0.55 LSQ_CLIP_MAX=1.0 LSQ_CLIP_INIT=0.90 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Evaluation notes

- Standard validation still uses the repository BPB metric.
- Sliding-window evaluation remains available through `EVAL_STRIDE=64`.
- EMA remains part of the base script and is always applied before export/eval.
- Export still writes the raw model plus the compressed mixed-precision artifact and then evaluates the round-tripped weights.

## Risks and tradeoffs

- **One-time recompile cost**: enabling late QAT now forces one extra compile event near warmdown.
- **Scalar clip ratio may be too coarse**: one learned ratio per matrix may underfit row-by-row outlier structure relative to full GPTQ-style search.
- **LeakyReLU interaction is unproven on this exact stack**: it helped the current top record, but it is not yet ablated here with LSQ-lite.
- **Still export-centric**: this candidate does not explore deeper structural ideas like train-only MTP or cross-layer sharing; it intentionally focuses on the most validated improvement axis in this repo.

## Validation run here

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604030756_lsq-lite-int6/train_gpt.py` | **Passed** |
| `SMOKE_TEST=1 SMOKE_BATCH_SIZE=2 SMOKE_SEQ_LEN=16 python candidates/202604030756_lsq-lite-int6/train_gpt.py` | **Not runnable in this environment** because the runner does not have the repo's Python dependencies installed (`numpy` and `torch` were missing). The script now includes a dependency-complete smoke path for a proper Python environment. |
