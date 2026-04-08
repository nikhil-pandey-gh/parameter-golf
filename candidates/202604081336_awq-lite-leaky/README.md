# AWQ-lite + LeakyReLU^2

## Hypothesis

The strongest no-TTT stack in this repo already squeezed most of its gains out of architecture and optimization, but quantization is still the main remaining bottleneck. A lightweight, activation-aware int6 export should reduce the post-training quantization gap more effectively than the current weight-only GPTQ-lite clip search, and the latest LeakyReLU(0.5)^2 activation win should transfer cleanly on top of that stack.

## Why this is promising for this repository

- The repo trend is overwhelmingly quantization-first: fp16 embeddings, mixed int6/int8 export, int5/int6 splits, and GPTQ-lite all produced real gains.
- The best pure train/export record here is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, which already has the strongest clean 11-layer architecture without depending on TTT.
- The current best overall record, `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, showed that swapping `relu^2` for `LeakyReLU(0.5)^2` was a meaningful additive gain.
- Even the 4-hour non-record run stayed bottlenecked by post-training quantization, which suggests export quality is still the best place to spend complexity.

## Prior repository influence

This candidate is primarily based on:

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - best no-TTT base,
   - mature 11-layer stack,
   - GPTQ-lite per-row clip search,
   - EMA/SWA + XSA + partial RoPE + LN scale + VE.
2. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - LeakyReLU(0.5)^2 activation improvement,
   - evidence that a one-line MLP activation change can still matter at this frontier.
3. `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
   - the earlier clean stack that established SmearGate + BigramHash + MLP3x as durable motifs.

There were no prior `candidates/` directories in the repo when this candidate was created.

## External research that informed the idea

- **AWQ**: *Activation-aware Weight Quantization for LLM Compression and Acceleration* (Lin et al., MLSys 2024, arXiv:2306.00978). The key takeaway is that activation statistics identify which weight channels matter most, and that using activation information can lower quantization error without retraining.
- **SmoothQuant**: *Accurate and Efficient Post-Training Quantization for Large Language Models* (Xiao et al., ICML 2023, arXiv:2211.10438). This reinforces the repo's own experience that outliers and activation-aware rescaling matter more than naive uniform clipping.
- **GPTQ**: *Accurate Post-Training Quantization for Generative Pre-trained Transformers* (Frantar et al., ICLR 2023, arXiv:2210.17323). This is the closest conceptual ancestor to the existing GPTQ-lite export path in the repo.
- **SpinQuant**: *LLM Quantization with Learned Rotations* (Liu et al., ICLR 2025, arXiv:2405.16406). Rotation-based outlier handling looks strong, but it is a noticeably heavier implementation jump than this repository currently needs.

The implementation here intentionally chooses the smallest useful step from that literature: keep the repo's current GPTQ-lite-style percentile search, but score those clip candidates with activation-weighted error from a tiny calibration pass.

## What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. swaps the MLP activation from `relu^2` to `LeakyReLU(0.5)^2`,
2. adds a short offline AWQ-style calibration pass after EMA is applied and before export,
3. records per-layer input second moments for each `CastedLinear` during that calibration pass,
4. changes the int6 clip-percentile search to minimize activation-weighted reconstruction error when those stats are available,
5. adds env knobs for the new export path:
   - `AWQ_ENABLED`
   - `AWQ_CALIBRATION_BATCHES`
   - `AWQ_CALIBRATION_TOKENS`
   - `LEAKY_RELU_SLOPE`

The architecture, training loop, tokenizer-agnostic BPB eval, and mixed int6/int8 export path are otherwise kept as close to the strong March 22 base as possible.

## How to run

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
AWQ_ENABLED=1 AWQ_CALIBRATION_BATCHES=8 AWQ_CALIBRATION_TOKENS=131072 \
LEAKY_RELU_SLOPE=0.5 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`AWQ_CALIBRATION_TOKENS` is the **total** token budget for the full calibration pass, split across `AWQ_CALIBRATION_BATCHES`.

## How to evaluate the idea

The most important signal is whether the candidate shrinks the gap between:

- the post-EMA full-precision diagnostic BPB, and
- the final `int6` roundtrip / sliding-window BPB.

If the hypothesis is right, the main win should appear in export quality rather than in early training loss.

## Main expected risks and tradeoffs

- **Extra export-time compute**: the calibration pass adds a few inference-only training-batch forwards before quantization.
- **Possible weak calibration signal**: too few batches may be noisy; too many batches may be wasted wallclock.
- **No guaranteed pre-quant improvement**: the AWQ-lite path is aimed at the compressed artifact, not the teacher model.
- **Still lighter than rotation-based PTQ**: this is intentionally less ambitious than SpinQuant/QuaRot-style changes, so the upside may also be smaller.

## Validation

- `python -m compileall candidates/202604081336_awq-lite-leaky/train_gpt.py` — passed in CI.
- CPU-only smoke test — not feasible in this environment:
  - `flash_attn_interface` is not installed here, and this script imports it at module load time,
  - no local FineWeb shard directories are present under `data/`, so there is no safe dataset-backed launch target either.
