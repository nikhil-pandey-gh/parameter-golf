# Candidate: Salient Row Sidecar PTQ

## Hypothesis

The strongest non-TTT stack in this repo already does a good job training an 11-layer 512d model, but its final score is still limited by post-training quantization. The remaining error should be concentrated in a small number of hard-to-quantize rows, especially in sensitive 2D tensors like tied embeddings and selected MLP / attention projections. If we keep the existing GPTQ-lite-style mixed int6/int8 export and spend a small global byte budget on int8 residual sidecars for only the worst rows, we should recover more roundtrip quality than a uniform bitwidth increase.

## Why this is promising here

- The root baseline loses meaningful quality at export time (`1.2172 -> 1.2244` bpb), which establishes quantization as a major bottleneck in this repository's scoring path.
- `2026-03-18_FP16Embed_WD3600` showed that embedding / output-head precision is unusually sensitive here, which is exactly the pattern an outlier-aware sidecar can target.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` is already a strong training-only base with GPTQ-lite clip search and still has artifact headroom versus the 16 MB cap, making it a better home for export-time experimentation than the tighter `2026-03-23` TTT record.
- This adds no training-path complexity, so it avoids the compile-time and stability risks that have already bitten late-QAT style changes in prior records.

## Prior repo work that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best non-TTT stack in the repo: 11L, XSA4, partial RoPE, LN scale, EMA, GPTQ-lite int6/int8 export.
- **Quantization sensitivity clue:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - Demonstrated that the tied embedding is much more precision-sensitive than most other tensors.
- **Why not branch from the top TTT record:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Best overall score, but it already uses almost the entire byte budget and adds substantial evaluation complexity; this candidate is intentionally isolating an export-side idea first.

## External research that informed it

- **AWQ**: protecting a very small subset of salient weights can recover most quantization quality without making the whole model higher precision.  
  https://arxiv.org/abs/2306.00978
- **SpQR**: near-lossless compression can come from isolating the hard-to-quantize outliers instead of uniformly increasing precision everywhere.  
  https://arxiv.org/abs/2306.03078
- **SpinQuant**: outliers remain a dominant source of quantization error even in strong modern PTQ pipelines, so targeted error reduction is a plausible lever.  
  https://arxiv.org/abs/2405.16406
- **AQLM**: more expressive compressed representations can outperform naive low-bit quantization, but full codebook machinery is too broad for this repo's next minimal fork.  
  https://arxiv.org/abs/2401.06118

I also considered a more aggressive hybrid ternary-QAT branch inspired by BitNet, but repo history makes QAT-path changes higher risk under a 600-second wallclock and this candidate is meant to test the cheaper export-time variant first.

## What changed vs the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. Keeps the same training architecture and mixed int6/int8 export path.
2. Computes residual error after quantizing each eligible 2D tensor.
3. Ranks the highest-error rows globally by estimated error reduction per byte.
4. Stores a small number of selected residual rows as int8 sidecars under a single global byte budget.
5. Reapplies those sidecars during dequantization before roundtrip evaluation.
6. Logs how many rows / tensors were protected and how much of the sidecar budget was used.

The new knobs are:

- `SALIENT_ROW_SIDECAR_ENABLED=1`
- `SALIENT_ROW_SIDECAR_MAX_BYTES=196608`
- `SALIENT_ROW_SIDECAR_TOPK=4`
- `SALIENT_ROW_SIDECAR_MIN_WIDTH=128`

## How to run / evaluate

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SALIENT_ROW_SIDECAR_ENABLED=1 SALIENT_ROW_SIDECAR_MAX_BYTES=196608 \
SALIENT_ROW_SIDECAR_TOPK=4 SALIENT_ROW_SIDECAR_MIN_WIDTH=128 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The sidecar feature is enabled by default, so the explicit `SALIENT_ROW_SIDECAR_*` env vars are mainly for ablation.

## Validation

Ran in this workflow:

1. `python -m compileall candidates/202604020029_salient-row-sidecar/train_gpt.py` — passed.
2. A CPU import-level smoke test was **not feasible** in this runner because the repo's runtime dependencies (`torch`, `numpy`, `sentencepiece`) and the `flash_attn_interface` module are not installed here, so even helper-only module import would require pulling in the full training stack.

## Main risks / tradeoffs

- The extra rows may not buy enough score to justify the added bytes.
- Compression ratio is data-dependent: the raw sidecar budget is controlled, but final compressed size will still vary.
- Protecting rows independently may miss structured residual error that a low-rank correction could capture better.
- If this works, the next follow-up should compare it against a more aggressive compression-aware training idea (for example a selective ternary-QAT MLP path) rather than stacking more export heuristics blindly.
