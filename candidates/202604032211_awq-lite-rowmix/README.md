# AWQ-lite RowMix + LeakyReLU^2

## Hypothesis

The strongest remaining headroom in this repository is **export-time quantization quality**, not raw full-precision training loss. The current best non-TTT 11-layer stack is already strong in fp/bf16, but the repository history keeps showing that final `val_bpb` is often limited by quantization damage. This candidate tests whether we can recover some of that gap by:

1. keeping the strong 11-layer EMA/XSA/Partial-RoPE stack from the `2026-03-22` record,
2. adopting the cheap **LeakyReLU(0.5)^2** activation win from the `2026-03-23` record, and
3. replacing uniform int6 export with an **activation-aware row-wise int6/int8 mix** that protects only the most valuable rows.

The core idea is AWQ-inspired, but adapted to this repository's constraints: collect a tiny calibration signal from already-seen train batches, score each quantized row by activation-weighted reconstruction error, and only widen the highest-value rows from int6-range to int8-range.

## Why this is promising here

Repository evidence points in the same direction:

- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` shows that even very long training still left a large post-quantization gap.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` and `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` both found that a small set of sensitive tensors dominate quantization quality.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` improved SOTA mostly through better export-time quantization and EMA, not a radical architecture change.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` showed that a one-line activation swap still buys measurable BPB on top of an already strong stack.

So instead of adding more runtime-heavy depth tricks or another large architectural change, this candidate spends complexity on a tighter export path.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- **Activation choice:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- **Quantization bottleneck evidence:**  
  `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md`  
  `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`  
  `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md`

## External research that informed it

1. **AWQ** - [Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)  
   Key takeaway: not all weights are equally important, and protecting only a tiny salient subset can sharply reduce quantization error.
2. **GPTQ** - [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)  
   Key takeaway: export-time weight quantization quality can materially move downstream LM quality even without changing training.
3. **HAWQ-V2** - [Hessian Aware trace-Weighted Quantization of Neural Networks](https://arxiv.org/abs/1911.03852)  
   Key takeaway: mixed precision should be allocated by sensitivity rather than uniformly.
4. **AdaRound** - [Up or Down? Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568)  
   Key takeaway: simple nearest rounding leaves quality on the table; smarter per-weight/per-row choices help.

This candidate does **not** implement those papers literally. It takes the part that best fits this codebase: a very lightweight saliency signal and a very cheap mixed-precision decision during export.

## What changed vs the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **LeakyReLU(0.5)^2 MLP**
   - `relu(x)^2` -> `leaky_relu(x, 0.5)^2`
   - pulled from the later `2026-03-23` record because it is cheap, orthogonal, and already empirically positive in-repo.

2. **AWQ-lite calibration pass**
   - the script caches the last few training microbatches already seen during training,
   - after applying EMA, it runs one no-grad calibration forward on those cached batches,
   - forward-pre hooks collect per-linear input RMS statistics.

3. **RowMix export**
   - for each quantized 2D MLP/attention weight matrix, the exporter tries the existing percentile-search quantizer at both int6-range (`[-31, 31]`) and int8-range (`[-127, 127]`) per row,
   - it scores the per-row gain using activation-weighted reconstruction error,
   - it then spends one **global** `AWQ_PROMOTE_FRAC` promotion budget across the whole model, so only the highest-value rows overall are promoted to the wider int8 range.

4. **Default late QAT disabled**
   - this candidate is intentionally centered on export-time quantization quality, not the existing late-QAT path.

Implementation note: this repo already serializes quantized weights as `int8` tensors plus row scales and relies on `zstd`/`zlib` for compression. That means widening only a few salient rows mostly changes **entropy/compressibility**, not the on-disk tensor format or loader.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202604032211_awq-lite-rowmix

DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_WD=0.04 ADAM_WD=0.04 MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 MLP_NEGATIVE_SLOPE=0.5 \
AWQ_ENABLED=1 AWQ_CALIBRATION_BATCHES=2 AWQ_PROMOTE_FRAC=0.01 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script performs training, EMA application, quantized export, dequantized roundtrip reload, and final validation in one run. The final metrics are the `final_rowmix_roundtrip*` and `final_rowmix_sliding_window*` log lines.

## Main expected risks / tradeoffs

- **Compression risk:** even if promoted rows reconstruct better, they may compress worse and erase the win in artifact bytes.
- **Calibration noise:** the saliency signal comes from a tiny cache of recent train batches, so it may overfit or simply be noisy.
- **Attribution blur:** this candidate mixes one known activation improvement with one new export idea, so any win would still need ablations.
- **Repo-specific packing constraint:** because the current serializer stores quantized tensors as `int8`, this is a practical proxy for mixed precision, not a true packed-bit implementation.

## Validation

Ran:

```bash
python -m compileall candidates/202604032211_awq-lite-rowmix/train_gpt.py
```

Outcome:

- **Passed**.

CPU-only runtime smoke testing was **not feasible** in this environment because this record-family script imports `flash_attn_interface` and hard-requires CUDA during execution; a meaningful start-up check would need the same GPU/FlashAttention stack expected by the existing 11-layer records.
