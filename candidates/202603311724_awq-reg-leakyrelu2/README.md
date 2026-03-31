# Activation-Aware Quantization Regularization + LeakyReLU^2

## Hypothesis

The current frontier in this repo suggests the next clean win is still in the **quantization bottleneck**, not a wholesale architecture rewrite. This candidate keeps the strongest non-TTT stack in the repo and adds a **late activation-aware quantization regularizer**: during the tail of training, it periodically measures per-layer activation RMS on a tiny calibration slice, refreshes cached int6 reconstructions with the existing GPTQ-lite export quantizer, and adds a small loss that pushes the largest block weights toward activation-weighted quantization-friendly values.

In parallel, it ports the repo-validated **`LeakyReLU(0.5)^2`** MLP activation, which was the clearest low-complexity win in the latest top record.

## Why this is promising for this repository

Several prior records point to the same conclusion: once the stack reached 11 layers, XSA, partial RoPE, LN scaling, EMA, and better evaluation, the limiting factor increasingly became **how much quality survives low-bit export**.

The clearest local evidence is:

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` showed the tied embedding was unusually quantization-sensitive.
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` explicitly argued that quantization quality was dominating gains.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` improved the best non-TTT stack with a better post-training quantizer.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` noted that the carried late-QAT path was effectively dead in that implementation.

So instead of reviving full fake-quant forward passes, this candidate uses a cheaper training-time signal that is explicitly aligned with the exported int6 artifact.

## Which records influenced it

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Direct borrow:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Additional motivation:

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`

There were no prior `candidates/` folders in this repo snapshot, so this idea is not overlapping an earlier candidate iteration.

## External research that informed it

This candidate is mainly inspired by the idea that **activation statistics should guide weight quantization pressure**, while keeping the implementation light enough for this challenge.

- **AWQ**: *Activation-aware Weight Quantization for LLM Compression and Acceleration* (`arXiv:2306.00978`) argues that activation distribution is the right signal for identifying quantization-sensitive channels.
- **GPTQ**: *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (`arXiv:2210.17323`) motivates targeting the eventual exported quantizer rather than only weight magnitude.
- **QDrop**: *QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization* (`arXiv:2203.05740`) highlights that quantization-aware reconstruction signals can improve low-bit robustness without full retraining.
- **LSQ**: *Learned Step Size Quantization* (`arXiv:1902.08153`) is a reminder that training quality often depends as much on quantizer configuration as on the base model.

This candidate does **not** implement full AWQ or GPTQ. Instead, it borrows the most repo-compatible idea from that literature: use activation statistics to decide where quantization error matters, and regularize toward the actual exported low-bit reconstruction late in training.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. The MLP activation is now **`LeakyReLU(0.5)^2`** instead of `ReLU^2`.
2. Added a **late activation-aware quantization regularizer**.
   - Enabled only once the wallclock-aware LR multiplier drops below `QUANT_REG_START_SCALE`.
   - Periodically runs a tiny uncompiled calibration pass on a small slice from the current training batch.
   - Tracks per-feature RMS for the large attention and MLP matrices.
   - Periodically refreshes cached int6 reconstructions using the same GPTQ-lite export quantizer.
   - Adds a weighted MSE loss from each live weight matrix to its cached quantized reconstruction, with feature weights derived from activation RMS.
3. The candidate disables the carried late-QAT flag by default (`LATE_QAT_THRESHOLD=0.0`), since the prior record documented that path as ineffective in practice.
4. Default dataset/tokenizer paths now resolve from the repo root so the script can be run **from inside this candidate directory**.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603311724_awq-reg-leakyrelu2
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs for this candidate:

```bash
QUANT_REG_ENABLED=1
QUANT_REG_WEIGHT=0.02
QUANT_REG_START_SCALE=0.25
QUANT_REG_REFRESH_EVERY=16
QUANT_REG_CALIBRATION_SEQS=2
QUANT_REG_CALIBRATION_LEN=256
QUANT_REG_EMA_DECAY=0.9
LATE_QAT_THRESHOLD=0.0
```

The script preserves the base stack's defaults for 11 layers, 3x MLP, XSA on the last 4 layers, partial RoPE, LN scaling, VE128, EMA, GPTQ-lite export, and sliding evaluation.

## Main expected risks or tradeoffs

- The regularizer adds **late-phase overhead** from periodic calibration and quantizer refreshes.
- The cached quantized targets are only refreshed every `QUANT_REG_REFRESH_EVERY` steps, so they lag the live weights slightly.
- Activation RMS comes from a very small calibration slice, so the weighting signal may be noisy.
- The idea targets post-export robustness, so it may slightly hurt pre-quant loss if `QUANT_REG_WEIGHT` is set too high.
- This candidate intentionally stays on the non-TTT path, so it does not inherit the evaluation gain from the current best legal TTT record.

## Validation run in this workflow

Commands run:

```bash
python -m compileall candidates/202603311724_awq-reg-leakyrelu2/train_gpt.py
```

Outcome:

- `compileall` passed.

Attempted additional smoke check:

- I tried an import-level smoke test with a stubbed `flash_attn_interface` so I could exercise the new helpers without a GPU run.
- That was **not feasible on this runner** because both `python` and `python3` lacked an importable `torch` package, so there was no safe way to execute even a minimal runtime path locally.
