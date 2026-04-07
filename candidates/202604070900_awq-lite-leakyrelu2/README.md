# AWQ-lite Int6 + LeakyReLU^2

## Hypothesis

The strongest non-TTT stack in this repo is already close to the point where post-training quantization error matters as much as training quality. This candidate keeps that 11-layer XSA/EMA/GPTQ-lite backbone, adds the repo-proven `LeakyReLU(0.5)^2` MLP activation, and replaces weight-only int6 clip search with an **activation-aware clip search** that uses a few training batches to weight quantization error by real input-channel power.

## Why this is promising here

- The recent record tree shows that the challenge has become increasingly **quantization-limited**, not just optimization-limited:
  - `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/` trained much longer but still lost a lot at export.
  - `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` got a measurable win from a better clip search alone.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` shows `LeakyReLU(0.5)^2` is a clean local improvement on top of the same general family of 11-layer stacks.
- AWQ-style calibration is attractive under this repo’s rules because it adds **no artifact bytes** and only a small post-training calibration pass.

## Prior repository experiments that influenced this candidate

- **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best non-TTT result in the repository at implementation time.
  - Already includes the 11L / XSA4 / partial-RoPE / LN-scale / EMA / GPTQ-lite recipe.
- **Activation carry-over:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Supplies the `LeakyReLU(0.5)^2` MLP change, which had a clean positive ablation in that stack.
- **Quantization bottleneck evidence:**  
  - `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`
  - `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`

At implementation time there were **no existing `candidates/` experiment directories** to avoid or extend.

## External research that informed it

- **AWQ — Activation-aware Weight Quantization** (Lin et al., MLSys 2024, arXiv:2306.00978)  
  Motivation: quantization sensitivity depends on the activation distribution, not just the weights.
- **GPTQ** (Frantar et al., ICLR 2023, arXiv:2210.17323)  
  Motivation: better low-bit PTQ comes from using sensitivity-aware error objectives instead of uniform clipping.
- **SpinQuant** (Liu et al., ICLR 2025, arXiv:2405.16406)  
  Motivation: outlier-aware / sensitivity-aware PTQ is still a live frontier for LLM compression.
- **LSQ** (Esser et al., ICLR 2020, arXiv:1902.08153)  
  Motivation: low-bit success often comes from improving the quantizer itself, not only the base model.

This candidate does **not** implement full AWQ rescaling or learned rotations. Instead it takes the smallest adaptation that fits this codebase cleanly: use activation power from calibration batches to choose better per-row int6 clip percentiles during export.

## What changed versus the base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **LeakyReLU(0.5)^2 MLP**
   - Replaces `relu(x)^2` with `leaky_relu(x, 0.5)^2`.
2. **AWQ-lite calibration pass**
   - Collects per-input-channel second moments for every `CastedLinear` using a few post-training batches from the training split.
3. **Activation-aware int6 clip search**
   - Existing GPTQ-lite search minimizes uniform weight-space MSE.
   - This candidate weights row reconstruction error by observed input-channel power, so errors on high-usage channels matter more.
4. **Portability-only extras**
   - FlashAttention 3 import is now guarded with an SDPA fallback so the script can do a CPU smoke run when FA3 is absent.
   - A `CPU_SMOKE_TEST=1` path exercises model init, forward/backward, calibration, quantization, and dequantization without dataset shards or CUDA.

## How to run or evaluate it

### Full training / export

```bash
cd candidates/202604070900_awq-lite-leakyrelu2

RUN_ID=awq_lite_leakyrelu2 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

- `AWQ_CALIBRATION_STEPS` (default `8`)
- `AWQ_CALIBRATION_BATCH_TOKENS` (default `262144`)
- `AWQ_CHANNEL_POWER` (default `1.0`)
- `AWQ_CLIP_CANDIDATES` (default `0.9990,0.9995,0.9999,0.99999,1.0`)

### CPU smoke

```bash
CPU_SMOKE_TEST=1 \
AWQ_CALIBRATION_STEPS=1 \
BIGRAM_VOCAB_SIZE=128 \
BIGRAM_DIM=32 \
VE_DIM=32 \
SMOKE_BATCH_SIZE=1 \
SMOKE_SEQ_LEN=32 \
python train_gpt.py
```

## Main expected risks and tradeoffs

- **Calibration overfit risk:** the activation-weighted clip search uses a tiny calibration slice from the training distribution. If the slice is not representative, it may choose worse clips than the plain GPTQ-lite search.
- **Small-gain regime:** recent winning deltas are often only `1e-3` to `3e-3` BPB, so this may be directionally correct but still hard to distinguish from noise without multi-seed runs.
- **LeakyReLU interaction risk:** `LeakyReLU(0.5)^2` was positive in the latest record stack, but its exact gain on the 03-22 non-TTT base is still unverified.
- **Extra export time:** the AWQ-lite pass is cheap relative to training, but it does add several post-training forward passes before quantization.

## Validation

1. `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604070900_awq-lite-leakyrelu2/train_gpt.py`  
   **Passed**.
2. `python3 -m venv /tmp/gh-aw/agent/pg-venv && /tmp/gh-aw/agent/pg-venv/bin/pip install -r requirements.txt`  
   **Passed**. This was only needed because the workflow runner’s system Python did not have the repo’s declared runtime dependencies installed.
3. ```bash
   CPU_SMOKE_TEST=1 \
   AWQ_CALIBRATION_STEPS=1 \
   BIGRAM_VOCAB_SIZE=128 \
   BIGRAM_DIM=32 \
   VE_DIM=32 \
   VE_LAYERS=9,10 \
   SMOKE_BATCH_SIZE=1 \
   SMOKE_SEQ_LEN=32 \
   /tmp/gh-aw/agent/pg-venv/bin/python candidates/202604070900_awq-lite-leakyrelu2/train_gpt.py
   ```
   **Passed** with:
   `cpu_smoke_test:ok loss:6.9239 roundtrip_loss:6.9297 awq_modules:68`

I did not run a real dataset-backed training step here: the normal training path is still CUDA-only and expects the FineWeb shard layout, so the safe local smoke target was the candidate’s CPU validation path.
