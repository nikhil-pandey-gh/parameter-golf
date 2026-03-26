# Candidate: AWQ-lite activation-aware int6 export

## Hypothesis

The strongest non-TTT stack in this repository is already close to the training frontier, but it still loses quality at export time. This candidate tests whether a tiny calibration pass over training-shard activations can improve the existing GPTQ-lite int6 export by choosing row clip thresholds with **activation-weighted** error instead of weight-only MSE.

In short: keep training unchanged, make quantization smarter.

## Why this is promising for this repository

Repository evidence points to quantization/export as a persistent bottleneck:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` showed that GPTQ-lite clip search alone was worth about `-0.0006 BPB` at zero training cost.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` showed that extra training compute improved pre-quant quality much more than post-quant quality, which is another hint that export is still the choke point.
- The current top non-TTT stack already carries most of the obvious modeling wins: 11 layers, XSA, partial RoPE, LN scale, EMA, VE128, SmearGate, and BigramHash. That makes low-cost export improvements especially attractive.

External research suggests this direction is well motivated:

- **AWQ** emphasizes protecting salient channels using activation information during weight quantization: <https://arxiv.org/abs/2306.00978>
- **SmoothQuant** shows that activation-aware rescaling can materially reduce quantization difficulty in transformer layers: <https://arxiv.org/abs/2211.10438>
- The recent transformer-compression survey highlights quantization and efficient architecture/export co-design as the highest-leverage practical tools under deployment constraints: <https://arxiv.org/abs/2402.05964>

This candidate intentionally implements the lightest-weight version that fits this repo cleanly: no new inference graph, no custom kernels, and no retraining-only scaffolding.

## Chosen base implementation

This candidate starts from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

That base was chosen because it is the strongest archived **non-TTT** implementation in this repo and already includes the modeling stack I want to preserve:

- 11 layers / 512d / 4 KV heads
- 3x MLP with relu-squared
- XSA on the final 4 layers
- partial RoPE + LN scale
- EMA + warmdown3500
- VE128, SmearGate, BigramHash
- GPTQ-lite int6 export

Other influential records:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`

There were **no prior entries under `candidates/`** when this candidate was created, so there was no earlier candidate implementation to avoid or extend.

## What changed versus the base

Training is intentionally unchanged. The differences are all in portability and export:

1. **Candidate-local default paths**
   - The script now derives default `DATA_PATH` and `TOKENIZER_PATH` from the repository root so it can be run directly from this candidate directory.

2. **AWQ-lite calibration pass**
   - After applying EMA weights and before export, the script runs a small train-shard calibration pass.
   - Forward pre-hooks collect per-input-channel second moments for each large `CastedLinear` that will actually be quantized.

3. **Activation-aware row clip selection for int6**
   - The existing 5-point GPTQ-lite clip search is kept.
   - Instead of scoring candidates with plain weight reconstruction MSE, rows can now be scored with activation-weighted MSE using the collected input moments.
   - This approximates output error better than weight-only scoring while preserving the repo's existing export format and dequantization path.

4. **AWQ controls**
   - Added:
     - `AWQ_ENABLED` (default `1`)
     - `AWQ_CALIBRATION_BATCHES` (default `8`)
     - `AWQ_CALIBRATION_BATCH_SEQS` (default `8`)

## How to run

From this candidate directory:

```bash
cd candidates/202603262101_awq-lite-int6
RUN_ID=awq_lite_candidate \
AWQ_ENABLED=1 \
AWQ_CALIBRATION_BATCHES=8 \
AWQ_CALIBRATION_BATCH_SEQS=8 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The default data/tokenizer paths resolve back to the repository root. If your dataset or tokenizer live elsewhere, override `DATA_PATH` and `TOKENIZER_PATH` explicitly.

## How to evaluate the idea

The main comparison is against the chosen base implementation:

- Keep the training stack the same.
- Compare `final_int6_roundtrip_exact` and especially `final_int6_sliding_window_s64_exact`.
- Watch whether the activation-aware export narrows the gap between post-EMA quality and post-int6 quality without pushing artifact bytes over budget.

A clean ablation is:

```bash
# Base-style export scoring
AWQ_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Activation-aware export scoring
AWQ_ENABLED=1 AWQ_CALIBRATION_BATCHES=8 AWQ_CALIBRATION_BATCH_SEQS=8 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main risks and tradeoffs

- The gain may be small if the existing GPTQ-lite search already captures most of the available export improvement.
- Activation-aware scoring uses train-shard statistics, so it could pick clip thresholds that help train-distribution activations more than validation-distribution activations.
- The calibration pass adds extra export/eval time, though it should be modest because it is forward-only and low-batch.
- This is still a heuristic approximation of AWQ/SmoothQuant, not a full implementation of either paper.

## Validation

Commands run in this workflow environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603262101_awq-lite-int6/train_gpt.py
python - <<'PY'
try:
    import torch
    print('torch:present', torch.__version__)
except Exception as exc:
    print('torch:missing', type(exc).__name__, exc)
PY
```

Outcomes:

- `python -m compileall ...` succeeded for the root scripts, `data/`, and this candidate script.
- A CPU runtime smoke test was **not feasible in this environment** because `torch` is not installed here, and this script also requires CUDA for execution.
- A focused code review of `candidates/202603262101_awq-lite-int6/` completed with no significant issues found.
