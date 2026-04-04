# Activation-Aware GPTQ-lite on the 11L EMA/XSA stack

## Hypothesis

The best non-TTT recipe in this repo already has a strong 11-layer backbone, but it still leaves score on the table at export time. Replacing its weight-only GPTQ-lite clip search with an activation-aware, per-row clip search should reduce the int6 roundtrip gap at essentially zero training cost.

## Why this is promising for this repository

- The record history shows that **export/compression improvements keep paying off**: fp16 embedding passthrough, mixed int6/int8, GPTQ-lite clip search, and tighter warmdown/EMA all produced repeatable gains.
- The strongest reusable static base is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, which already consolidated the winning 11L/XSA/EMA/partial-RoPE stack and then improved further with a zero-training-cost quantization tweak.
- Recent quantization literature points the same way: for low-bit weight-only export, **activation statistics matter more than raw weight magnitudes** when deciding what to protect.

## Prior records and candidates that influenced this

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` for the 11-layer XSA + EMA core.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` for partial RoPE + layer scaling.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` as the direct implementation base.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` as evidence that the repo frontier is now mostly in final-stage refinements, even though this candidate intentionally stays on the simpler static-export path.
- No prior `candidates/` directory existed when this candidate was created.

## External research that informed it

- **GPTQ** — Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (arXiv:2210.17323). Motivates using second-order / activation-informed error instead of plain weight reconstruction.
- **AWQ** — Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (arXiv:2306.00978). Motivates protecting the weight dimensions that matter most under the observed activation distribution.
- **EfficientQAT** — Chen et al., *EfficientQAT: Efficient Quantization-Aware Training for Large Language Models* (arXiv:2407.11062). Reinforces the idea that the final quantization parameters are a high-leverage place to spend complexity.
- **Scaling Law for Quantization-Aware Training** — Chen et al. (arXiv:2505.14302). Suggests weight quantization error becomes more important as training tokens grow, which matches this repo's strong training stack plus remaining export gap.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate keeps the same model/training recipe, then makes five surgical changes:

1. Adds a post-EMA calibration pass over a small slice of training-stream sequences to collect per-layer input second moments for quantized `CastedLinear` layers.
2. Replaces the base record's global weight-MSE clip search with a **per-row activation-weighted clip search** across the same percentile candidates.
3. Makes FlashAttention optional, uses it only on supported CUDA setups, and falls back to PyTorch SDPA otherwise.
4. Stores the artifact codec in the compressed blob so loading no longer depends on the local machine guessing zstd vs. zlib correctly.
5. Adds `CPU_SMOKE_TEST=1`, which runs a tiny synthetic forward/backward + calibration + quantize/dequantize cycle without dataset or tokenizer access, and also resolves default data/tokenizer paths by searching upward for the repo root.

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202604041548_activation-aware-gptqlite
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
ACTIVATION_AWARE_QUANT=1
ACTIVATION_CALIBRATION_SEQS=96
ACTIVATION_CALIBRATION_BATCH_SEQS=8
ACTIVATION_CLIP_CANDIDATES=0.9990,0.9995,0.9999,0.99999,1.0
```

The script now resolves dataset/tokenizer defaults relative to the repository root, so it can be launched from inside the candidate directory without rewriting `DATA_PATH` or `TOKENIZER_PATH`.

## Validation

- `python -m compileall candidates/202604041548_activation-aware-gptqlite/train_gpt.py`  
  Outcome: success.
- `python3 -m venv /tmp/gh-aw/agent/pg-venv && . /tmp/gh-aw/agent/pg-venv/bin/activate && pip install numpy sentencepiece torch && CPU_SMOKE_TEST=1 python candidates/202604041548_activation-aware-gptqlite/train_gpt.py`  
  Outcome: success, printed `cpu_smoke_test:ok loss:4.8976 calibrated_layers:13`.

Notes:

- The CPU smoke test was run in an isolated temporary venv because the container did not have `torch`, `numpy`, or `sentencepiece` preinstalled.
- A full training smoke run was not attempted here because the real path still expects CUDA plus the repo's dataset shards/tokenizer.

## Main expected risks and tradeoffs

- The calibration slice uses the first training-stream sequences, so it may still be unrepresentative if those tokens are not close to the eventual evaluation distribution.
- The scoring metric is still a diagonal activation-weighted proxy, not full GPTQ blockwise reconstruction, so gains are plausible but not guaranteed.
- Export gets a little slower because it performs one extra calibration pass before quantization.
- The SDPA fallback is for portability and smoke tests; competitive runs should still prefer the FlashAttention path when available.
