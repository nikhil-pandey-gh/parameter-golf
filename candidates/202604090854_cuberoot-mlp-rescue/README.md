# Cube-Root MLP Equalization + FP16 Rescue Channels

## Hypothesis

The strongest clean pre-TTT stack in this repo still leaves a meaningful post-quantization gap, so the next high-leverage improvement is smarter export rather than another slower training-time architectural change.

This candidate applies an exact **cube-root channel equalization** to every MLP before export, then keeps a small number of the worst quantized hidden channels in fp16. Because the base MLP uses `relu(x)^2`, scaling a hidden unit by `s` requires scaling the corresponding output column by `1 / s^2`, so the magnitude-balancing transform lands at a cube root instead of the square root used in standard ReLU equalization.

## Why this looks promising here

- The records show that **export quality is a recurring bottleneck**: even very strong runs still report a non-trivial quantization penalty, and a 4-hour non-record baseline still underperformed after quantization despite much better pre-quant quality.
- The 2026-03-22 record already has a strong core stack without TTT and still leaves roughly **0.45 MB** of artifact headroom, which is enough to spend a little budget on selectively protected channels.
- External quantization research points in the same direction: **weight equalization / scale migration** and **salient-channel protection** both consistently improve low-bit PTQ without retraining.

## Main repository influences

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`  
  Chosen as the base implementation because it is the strongest simpler pre-TTT stack and already includes EMA, GPTQ-lite clip search, Partial RoPE, LN scale, XSA, SmearGate, BigramHash, VE, and the 11L/3x-MLP trunk.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`  
  Important evidence that the frontier has moved toward evaluation/export refinements, but it is also much more invasive because of legal TTT and parameter banking.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md`  
  Strong reminder that extra training alone does not solve the artifact bottleneck.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`  
  Reinforces that **targeted higher-precision preservation** can be worth spending bytes on when a tensor is unusually quantization-sensitive.

## External research used

- **SmoothQuant** — Xiao et al., 2023: offline migration of quantization difficulty with equivalent scaling transforms.  
  <https://arxiv.org/abs/2211.10438>
- **AWQ** — Lin et al., 2024 MLSys best paper: protect salient channels using equivalent rescaling instead of broad mixed precision.  
  <https://arxiv.org/abs/2306.00978>
- **SpQR** — Egiazarian et al., 2023: isolate quantization outliers and store them in higher precision.  
  <https://arxiv.org/abs/2306.03078>
- **Data-Free Quantization via Weight Equalization** — Nagel et al., 2019: classic cross-layer equalization baseline.  
  <https://arxiv.org/abs/1906.04721>
- **SpinQuant** — Liu et al., 2025: modern evidence that structured preconditioning before quantization materially helps low-bit LLM export.  
  <https://arxiv.org/abs/2405.16406>

## Why not the externally suggested recurrence/sharing route

The broader literature does make partial sharing and recurrence look tempting for parameter-limited models, but this repository's own history is a strong counter-signal: earlier recurrence experiments explicitly reported that they needed more optimization steps than the 600-second budget allows. Given that evidence, I prioritized a **quantization-path improvement that does not slow training**.

## What changed versus the base implementation

Base file: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Export-time MLP equalization**
   - For every `blocks.*.mlp.fc.weight` / `blocks.*.mlp.proj.weight` pair, compute per-hidden-unit scales from the row/column max magnitudes.
   - Apply the exact function-preserving transform:
     - `fc_row *= s`
     - `proj_col /= s^2`
   - Use a configurable cube-root equalization factor via `EXPORT_MLP_EQ_ALPHA`.

2. **FP16 rescue channels**
   - After equalization, estimate per-hidden-channel relative quantization error under the existing int6 exporter.
   - Keep the top `EXPORT_MLP_RESCUE_CHANNELS` hidden channels per block in fp16 and overwrite those slices after roundtrip dequantization.

3. **CPU-friendly smoke mode**
   - Added `SMOKE_TEST=1` path with a synthetic model that is intentionally large enough to hit the real int6 MLP export path, plus a non-FlashAttention fallback so the candidate can be sanity-checked without GPUs or dataset shards.

4. **FlashAttention fallback**
   - If `flash_attn_interface` is unavailable or the model is on CPU, attention falls back to `torch.nn.functional.scaled_dot_product_attention`.

## How to run

### Training / export on the normal challenge stack

```bash
cd candidates/202604090854_cuberoot-mlp-rescue
RUN_ID=cuberoot_mlp_rescue \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script resolves its default `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so it can be launched directly from this candidate directory without extra path plumbing.

Optional export knobs:

```bash
EXPORT_MLP_EQ_ENABLED=1
EXPORT_MLP_EQ_ALPHA=1.0
EXPORT_MLP_EQ_MAX_SCALE=8.0
EXPORT_MLP_RESCUE_CHANNELS=8
```

### Local smoke test

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Risks / tradeoffs

- The equalization transform is exact in function space, but the **chosen statistics** (row/column maxima plus relative quantization error proxy) are still heuristic.
- Protecting too many channels in fp16 can eat the remaining artifact headroom.
- This candidate only preconditions **MLP weights**, not attention matrices, so the upside may be capped if the remaining quantization gap is attention-dominated.
- The CPU smoke path is intentionally minimal and does not prove anything about Hopper throughput or final BPB.

## Validation

- `/tmp/gh-aw/agent/pg-venv/bin/python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604090854_cuberoot-mlp-rescue/train_gpt.py` — passed.
- `SMOKE_TEST=1 /tmp/gh-aw/agent/pg-venv/bin/python candidates/202604090854_cuberoot-mlp-rescue/train_gpt.py` — passed while exercising the int6 MLP export path.
- `cd candidates/202604090854_cuberoot-mlp-rescue && SMOKE_TEST=1 /tmp/gh-aw/agent/pg-venv/bin/python train_gpt.py` — passed with `smoke_test:ok loss=4.180583 roundtrip_loss=4.180573 rescued_channels=16`, confirming the candidate is runnable from its own directory.
