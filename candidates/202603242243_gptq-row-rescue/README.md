# Candidate: GPTQ-lite Row Rescue

## Hypothesis

The current best 11-layer stack is increasingly limited by **post-training quantization quality**, not by raw training loss. The repo history shows repeated wins from fp16 passthrough for sensitive tensors, mixed int6/int8 export, EMA/SWA smoothing, and most recently GPTQ-lite clip search. This candidate pushes that exact frontier one step further by:

1. upgrading the current GPTQ-lite exporter from **matrix-wide clip-percentile choice** to **true row-wise clip-percentile choice**, and
2. spending a small, explicit byte budget on the **worst reconstructed int6 rows** in fp16.

The expected upside is a smaller quantization gap at essentially zero training-time cost and without changing the proven 11L/XSA/Partial-RoPE/EMA architecture.

## Why this is promising for this repository

The repository trend is clear: recent improvements have come more from **better export under the 16MB cap** than from wholesale architecture changes.

- The current best record, `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, already improved the leaderboard mainly through a smarter post-training quantizer plus EMA.
- Earlier records showed that keeping especially sensitive tensors in higher precision can be worth the bytes:
  - `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600` kept the tied embedding in fp16.
  - `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` kept the last layer key projection in fp16.
- The non-record 4-hour baseline under `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` improved pre-quant quality much more than post-quant quality, which is another sign that **quantization error is the bottleneck**.

This candidate follows that evidence rather than trying a speculative new backbone.

## Prior experiments that influenced this candidate

### Main base implementation

`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate keeps the whole winning stack intact:

- 11 layers, 512 dim, 3x MLP
- U-Net skip connections
- XSA on the last 4 layers
- Partial RoPE (16/64)
- LN scaling
- SmearGate + BigramHash + shared Value Embedding
- EMA for export/eval
- int6/int8 mixed export with zstd

### Other influential records

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600`
  - showed that a small mixed-precision carveout can pay for itself if it targets a highly sensitive tensor.
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow`
  - showed that coarse mixed-precision export is already a strong lever in this repo.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`
  - demonstrated that even a single carefully chosen fp16 projection can help quantization.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - reinforced that the best gains now come from small, targeted changes on top of the 11L stack.

There were **no existing `candidates/` directories** to reuse or avoid duplicating.

## External research that informed it

This candidate is grounded in the same family of ideas as several primary-source quantization papers:

- **GPTQ** — Frantar et al., 2022.
  - <https://arxiv.org/abs/2210.17323>
  - Key relevance: row-wise, low-bit, post-training quantization can recover a lot of quality with careful reconstruction.
- **AWQ** — Lin et al., 2023/2024.
  - <https://arxiv.org/abs/2306.00978>
  - Key relevance: not all weights are equally important; protecting a tiny salient subset can disproportionately reduce quantization error.
- **LLM.int8()** — Dettmers et al., 2022.
  - <https://arxiv.org/abs/2208.07339>
  - Key relevance: mixed-precision treatment of outliers can preserve quality without giving up most of the compression win.
- **SpQR** — Egiazarian et al., 2023.
  - <https://arxiv.org/abs/2306.03078>
  - Key relevance: isolate especially bad-to-quantize outliers and store them more precisely.

I intentionally adapted the **salient/outlier preservation idea** in a minimal way that matches this repo’s existing exporter instead of importing a heavy new quantization stack.

## What changed versus the chosen base implementation

Relative to the 2026-03-22 record script, this candidate makes two focused changes plus one quality-of-life fix:

1. **True row-wise clip search for int6 export**
   - The base script searched a small percentile grid and chose a single best clip policy per matrix.
   - This candidate chooses the best clip percentile **per row**, still from the same tiny search grid.
   - That gives more flexibility at export time without changing training.

2. **Byte-budgeted fp16 row rescue**
   - After int6 quantization, the script scores rows by reconstruction error per byte.
   - It then preserves a small global budget of the worst rows in fp16, charging both the rescued row payload and the stored row indices against that budget.
   - Defaults:
     - `ROW_RESCUE_BUDGET_BYTES=280000`
     - `ROW_RESCUE_MAX_PER_TENSOR=48`
     - `ROW_RESCUE_MIN_TENSOR_NUMEL=262144`
   - The rescue rows are written into the compressed artifact and re-applied during dequantization.

3. **Candidate-directory path robustness**
   - Default `DATA_PATH` and `TOKENIZER_PATH` are resolved relative to the repo root, so this script can be run directly from the candidate directory.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603242243_gptq-row-rescue
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable the new mixed-precision rescue path.
ROW_RESCUE_BUDGET_BYTES=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Make the rescue path more conservative.
ROW_RESCUE_BUDGET_BYTES=160000 ROW_RESCUE_MAX_PER_TENSOR=24 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The rest of the defaults intentionally stay aligned with the current best 11L stack.

## Main expected risks and tradeoffs

- **Export-time complexity increases.** The new row-wise search and rescue pass make post-training quantization slower, though training speed is unchanged.
- **The rescue heuristic is activation-blind.** AWQ-style methods use activation statistics; this candidate uses reconstruction error per byte only, which is simpler but less informed.
- **Artifact budgeting is now another tuning surface.** Too much fp16 rescue can eat into the 16MB cap; too little may leave quality on the table.
- **The gain may be modest.** This is a high-probability, low-infrastructure idea, but it is still an incremental improvement on a very tuned baseline.

## Validation

### Commands run

```bash
python -m compileall candidates/202603242243_gptq-row-rescue/train_gpt.py
```

### Outcomes

- `python -m compileall ...` **passed**.
- A fuller CPU import/smoke run was **not feasible in this environment** because the runner does not have the repository’s Python runtime dependencies installed (`torch` and other required packages are missing), and the script’s real forward path also expects CUDA/FlashAttention.

That means this candidate currently has a **syntax-level validation pass**, but it still needs a proper GPU-backed smoke/eval run in a fully provisioned repo environment.
