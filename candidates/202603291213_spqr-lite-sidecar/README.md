# SPQR-lite sidecar on the 11L EMA + GPTQ-lite base

## Hypothesis

The strongest training-only 11-layer stack in this repo already squeezes most of the easy gains out of architecture and schedule tuning, but it still relies on uniform per-row int6 quantization for the largest attention and MLP weights. A tiny `SpQR-lite` sidecar that stores the worst residual quantization errors in fp16 should recover a disproportionate amount of the roundtrip loss while staying inside the remaining artifact budget. I also port `LeakyReLU(0.5)^2` from the current best in-tree record because it is a cheap, repo-proven way to improve the underlying full-precision checkpoint before compression.

## Why this is promising here

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` is the best strong non-TTT base in the repo, and it already has the winning 11L / XSA4 / Partial-RoPE / LN-scale / VE128 / BigramHash / EMA recipe.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` reports that `LeakyReLU(0.5)^2` was worth about `-0.0021` to `-0.0030` BPB in its ablations, so it is a natural low-risk carry-over to the simpler 11L base.
- Earlier repo notes repeatedly show that quantization, not just training loss, is a central bottleneck. The 4-hour non-record run improved pre-quant quality a lot but still landed at `1.2074` after compression, and the fp16-embedding record was largely about reducing quantization damage.
- This candidate spends the remaining artifact headroom on the handful of weights that are hardest for GPTQ-lite int6 to reconstruct, instead of broadening the whole format or changing training infrastructure.
- If you have seen earlier remote `SpQR-lite` explorations around this repo, the twist here is deliberate: use the cleaner 2026-03-22 donor, fold in the repo-proven `LeakyReLU(0.5)^2` activation, and apply a bounded per-matrix residual sidecar instead of a late-layer-only outlier rule.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen base implementation and training recipe
  - proved that GPTQ-lite clip search + EMA + the 11L XSA4 stack are a strong foundation
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - source of the `LeakyReLU(0.5)^2` MLP change
  - showed that activation tweak still matters on top of a strong modern stack
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - reinforced that small quantization/export changes can be worth more than raw training tweaks
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`
  - highlighted that better checkpoints still need a smarter artifact format to keep their gains

## External research that informed it

- **AWQ** (`arXiv:2306.00978`)
  - argues that a small subset of salient weights dominates low-bit degradation, and that selectively protecting them can materially reduce error
- **SpQR** (`arXiv:2306.03078`)
  - isolates outlier weights and stores them in higher precision while quantizing the bulk of the matrix
- **AQLM** (`arXiv:2401.06118`)
  - further supports the idea that extreme compression benefits from spending extra bits only where they matter most

This candidate does **not** implement full AWQ or SpQR. Instead, it uses a minimal repo-friendly adaptation: after GPTQ-lite per-row int6 quantization, it stores a tiny fp16 residual sidecar for the highest-error weights in each large int6 matrix.

## What changed vs the chosen base

1. **LeakyReLU(0.5)^2 MLP**
   - Replaces `relu^2` with `LeakyReLU(0.5)^2`, following the 2026-03-23 record's best ablation.

2. **SPQR-lite residual sidecar**
   - New env knobs:
     - `SPQR_SIDECAR_ENABLED=1`
     - `SPQR_SIDECAR_FRAC=0.0005`
     - `SPQR_SIDECAR_MAX_COUNT=512`
   - For each large int6 attention/MLP matrix, the exporter:
     - runs the existing GPTQ-lite clip search,
     - computes the flat residual after dequantization,
     - saves the top residual indices plus fp16 residual values,
     - reapplies them during roundtrip load.

3. **FlashAttention fallback**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA so the script is easier to run outside the exact record environment.

4. **Path handling**
   - Default dataset and tokenizer paths are resolved relative to the repository root, so the script can be launched from this candidate directory without rewriting paths.

## How to run

From this candidate directory:

```bash
cd candidates/202603291213_spqr-lite-sidecar

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
WARMDOWN_ITERS=3500 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
SPQR_SIDECAR_ENABLED=1 SPQR_SIDECAR_FRAC=0.0005 SPQR_SIDECAR_MAX_COUNT=512 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to compare against the base export path only, disable the sidecar with `SPQR_SIDECAR_ENABLED=0`.

## Validation

Commands run for this candidate:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603291213_spqr-lite-sidecar/train_gpt.py
```

Outcome:

- both compile checks passed
- no runtime smoke test was executed in this environment because the available Python environment does not currently have `torch` or `sentencepiece` installed, and the repository snapshot here does not include the expected dataset/tokenizer artifacts needed for a safe start-up run

## Main risks and tradeoffs

- The sparse sidecar is a heuristic, not full activation-aware calibration; the top residuals by absolute error may not perfectly line up with BPB-sensitive weights.
- If `SPQR_SIDECAR_FRAC` or `SPQR_SIDECAR_MAX_COUNT` are pushed too high, artifact size can erase the benefit.
- `LeakyReLU(0.5)^2` transferred well in the more complex record, but it still needs confirmation on this simpler 11L training-only base.
- The SDPA fallback improves portability, but the intended fast path is still FlashAttention 3 on CUDA.
