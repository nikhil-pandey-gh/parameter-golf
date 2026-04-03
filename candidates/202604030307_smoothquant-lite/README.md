# SmoothQuant-lite Export Scaling on the 11L GPTQ-lite Stack

## Hypothesis

The strongest non-TTT training stack in this repo is already heavily optimized for int6 export, but it still treats export quantization mostly as a per-row weight problem. A lightweight **SmoothQuant-lite** pass should help by measuring large activation channels on a short calibration sweep, storing a small per-linear input scale, and quantizing an equivalent rescaled weight matrix instead of the original one. That should reduce the remaining int6 roundtrip gap with little code or artifact overhead.

I also carried forward the now-proven **LeakyReLU(0.5)^2** MLP activation from the current best record, because repository evidence suggests it is the biggest low-risk training delta on top of the 11-layer stack.

## Why this is promising here

- The repo's winning trend is clear: the best records increasingly win by reducing the **post-training quantization penalty**, not by changing the overall model family.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed that better clip selection and EMA still produced measurable gains without changing the core training loop.
- External PTQ work suggests that **outlier-aware equivalent transforms** can further improve low-bit export quality by making weight/activation distributions easier to quantize:
  - SmoothQuant: <https://arxiv.org/abs/2211.10438>
  - AWQ: <https://arxiv.org/abs/2306.00978>
  - QuaRot: <https://arxiv.org/abs/2404.00456>
  - SpinQuant: <https://arxiv.org/abs/2405.16406>
- This candidate uses the easiest version that fits the repo: no learned rotations, no extra training phase, just a short calibration sweep over train batches and a per-linear scaling vector stored in the exported state.

## Prior repo work that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Activation carry-over:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- Other trends that shaped the choice:
  - XSA / EMA / Partial RoPE / LN scale continue to stack well.
  - Parameter banking mainly improved speed, not score.
  - Late QAT has been fragile in this repo, so this candidate focuses on **export-time** improvement instead of another training-time fake-quant variant.

## External research that informed it

1. **SmoothQuant** (<https://arxiv.org/abs/2211.10438>) showed that moving activation difficulty into weights via equivalent scaling can materially improve PTQ.
2. **AWQ** (<https://arxiv.org/abs/2306.00978>) reinforced that activation-aware quantization matters because a small fraction of salient channels often dominate error.
3. **QuaRot** (<https://arxiv.org/abs/2404.00456>) and **SpinQuant** (<https://arxiv.org/abs/2405.16406>) pushed the same theme further with rotation-based outlier removal.
4. I also reviewed newer low-bit papers while choosing the direction, including **Low-Rank Quantization-Aware Training for LLMs** (<https://arxiv.org/abs/2406.06385>) and **SERQ** (<https://arxiv.org/abs/2603.08185>), but this candidate keeps the implementation narrower and closer to the repo's current export pipeline.

## What changed vs the chosen base

Compared with the 2026-03-22 record script, this candidate:

1. switches the MLP from `ReLU^2` to **LeakyReLU(0.5)^2**,
2. adds a **SmoothQuant-lite** calibration pass controlled by:
   - `SQ_ENABLED`
   - `SQ_ALPHA`
   - `SQ_CALIB_STEPS`
   - `SQ_MAX_SCALE`
3. stores a per-`CastedLinear` `sq_scale` buffer plus a serialized `sq_apply` flag,
4. keeps `sq_apply=0` for the live/full-precision model, but sets `sq_apply=1` for transformed int6 roundtrip weights,
5. quantizes `weight * sq_scale` for int6 matrices so the quantized model is equivalent to dividing the linear input by `sq_scale`,
6. adds a **FlashAttention fallback** to PyTorch SDPA so the candidate can be imported and smoke-tested without `flash_attn_interface`,
7. adds `SMOKE_TEST=1` for a tiny CPU-only model instantiation + quantize/dequantize roundtrip.

## How to run or evaluate

### Full GPU training/eval

```bash
RUN_ID=sq_lite_candidate \
SEED=1337 \
SQ_ENABLED=1 SQ_ALPHA=0.8 SQ_CALIB_STEPS=16 SQ_MAX_SCALE=16 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### CPU smoke test

```bash
SMOKE_TEST=1 python train_gpt.py
```

This smoke path uses the SDPA fallback and only checks that the candidate can instantiate, calibrate export scales, quantize/dequantize, reload, and score a tiny synthetic batch without crashing.

## Validation run for this candidate

Commands run in this workflow:

```bash
python -m compileall candidates/202604030307_smoothquant-lite/train_gpt.py
SMOKE_TEST=1 python candidates/202604030307_smoothquant-lite/train_gpt.py
```

Outcome:

- `compileall`: passed
- CPU smoke test: passed with `smoke_test:ok fp_loss:5.2941 q_loss:5.2941 scaled_layers:4`

In this workflow, the smoke path was run from a temporary virtual environment because the runner's system Python did not include the repository dependencies by default.

## Main risks and tradeoffs

- **Runtime overhead:** each scaled linear adds an elementwise divide on the input path at eval time.
- **Calibration sensitivity:** a short calibration pass may over-smooth some channels or underfit others; `SQ_ALPHA` and `SQ_CALIB_STEPS` likely need tuning.
- **Artifact budget:** the `sq_scale` vectors are small, but they are not free.
- **Train/eval mismatch:** the model is still trained without the scaling transform; only export/eval sees it.
- **Not a full AWQ/SpinQuant implementation:** this is the smallest useful equivalent-transform variant, not a learned rotation method.
