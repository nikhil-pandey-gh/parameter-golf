# Candidate: One-Head MTP on the 2026-03-22 Pre-TTT Stack

## Hypothesis

Enable a **single training-only multi-token prediction (MTP) head** on top of the strongest simpler pre-TTT stack (`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`) to improve sample efficiency during the fixed 600-second training budget **without increasing artifact size**, because the auxiliary head is stripped before export.

## Why this is promising for this repository

The record history shows that this repo has already harvested many of the obvious architectural and export wins:

- sliding-window evaluation,
- 11-layer / 3x MLP stacks,
- SmearGate + BigramHash,
- EMA / longer warmdown,
- selective XSA,
- partial RoPE + LN scale,
- GPTQ-lite style clip search,
- and, at the current top, evaluation/system complexity like legal TTT and parallel Muon.

What is *not* represented in prior record READMEs is a training-only auxiliary objective that improves learning efficiency while leaving the final artifact unchanged. That makes MTP a good fit for the current frontier: it spends extra compute during training, not bytes in the exported model.

## Prior repository experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen as the base implementation because it is the strongest cleaner pre-TTT stack and already combines the mature repo recipe: 11L, 3x MLP, XSA4, partial RoPE, LN scale, VE128, SmearGate, BigramHash, EMA, GPTQ-lite, and zstd export.

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Useful as evidence that the very top score currently depends on additional evaluation/systems complexity. This candidate intentionally targets a different axis: better training efficiency before any TTT.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Confirms that partial RoPE + LN scale were real wins and should be preserved.

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - Confirms EMA + XSA on late layers is a stable part of the winning stack.

- No prior `candidates/` directory existed when this candidate was created, so this idea is not repeating an existing candidate implementation.

## External research that informed it

- Fabian Gloeckle et al., **“Better & Faster Large Language Models via Multi-token Prediction”**, arXiv:2404.19737
  - Shows that multi-token prediction can improve sample efficiency as an auxiliary training task while keeping benefits even when the final use case is still standard next-token prediction.

- Guoliang Zhao et al., **“Self-Distillation for Multi-Token Prediction”**, arXiv:2603.23911
  - Highlights that multiple MTP heads can be harder to train jointly and motivates a conservative configuration when preserving main-head quality matters. That is why this candidate uses **one** auxiliary head with a modest loss weight rather than a more aggressive multi-head setup.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Turned on one auxiliary MTP head by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`

2. **Kept export artifact size unchanged in spirit**
   - The script already excluded `mtp_heads` from the exported state dict.
   - This candidate keeps that behavior and leans on it as the core idea: spend training compute, not artifact bytes.

3. **Made the script runnable from the candidate directory by default**
   - `DATA_PATH` and `TOKENIZER_PATH` now default via `Path(__file__).resolve().parents[2]`, so the candidate can be launched from its own folder without manually fixing relative paths.

4. **Added a narrow FlashAttention fallback for validation portability**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA.
   - This is mainly for local smoke validation and does not change the intended GPU fast path when FlashAttention is installed.

5. **Added `SMOKE_TEST=1`**
   - Runs a tiny CPU-only forward/backward plus quantize/dequantize roundtrip.
   - Explicitly verifies that `mtp_heads` are excluded from export and that the int6 path is exercised.

## How to run / evaluate

From the candidate directory:

```bash
cd candidates/202603312347_one-head-mtp
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional overrides if you want to sweep the idea:

```bash
cd candidates/202603312347_one-head-mtp
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script keeps the base stack defaults for the rest of the setup:

- 11 layers, 512 dim, 8 heads / 4 KV heads
- 3x MLP
- SmearGate + BigramHash
- XSA on the last 4 layers
- partial RoPE (`16`)
- LN scaling
- VE128 on layers `9,10`
- EMA + tight SWA
- GPTQ-lite int6 export + zstd
- sliding-window evaluation at stride `64`

## Expected risks / tradeoffs

- **Training compute tradeoff:** even one auxiliary head adds extra logits/loss work. If the throughput hit is larger than the sample-efficiency gain, the net result can be flat or negative.

- **Main-head interference:** MTP can improve representations, but it can also pull optimization away from the exact next-token objective if the loss weight is too high. That is why this candidate starts with a single head and `0.15` weight instead of a larger multi-head setup.

- **Not yet ported to the top TTT stack:** this candidate intentionally starts from the cleaner 2026-03-22 base. If the idea helps, the next experiment should port it onto the 2026-03-23 LeakyReLU² + legal TTT stack.

- **Portability fallback is for validation, not speed:** the SDPA path is only meant to make local smoke tests possible when FlashAttention is unavailable.

## Validation run

Validated locally with lightweight checks:

```bash
python -m compileall candidates/202603312347_one-head-mtp/train_gpt.py
```

Outcome: success.

```bash
cd candidates/202603312347_one-head-mtp
SMOKE_TEST=1 python train_gpt.py
```

Outcome: success.

Observed smoke output:

```text
smoke_test:ok loss:5.3526 logits_shape:(2, 24, 96) quant_tensors:8 flash_attn:False
```

Notes:

- The smoke test was run inside a temporary venv because the runner initially lacked `torch`, `numpy`, and `sentencepiece`.
- The smoke configuration is intentionally tiny, CPU-only, and only meant to confirm that the candidate starts, runs backward, excludes MTP export tensors, and survives an int6 quantize/dequantize roundtrip.
- The final smoke config uses grouped-query attention (`num_heads=8`, `num_kv_heads=4`), so the SDPA fallback is exercised in the same GQA regime as the actual candidate defaults.
