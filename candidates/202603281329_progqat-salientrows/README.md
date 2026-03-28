# Progressive QAT + Salient-Row Rescue

## Hypothesis

The strongest non-TTT stacks in this repository are now bottlenecked more by **export quality** than by raw pre-quantization loss. A candidate that makes the model more int6-friendly during training and then spends a tiny amount of extra artifact budget on the most quantization-sensitive rows should improve the final roundtrip model more reliably than another broad architectural rewrite.

This candidate therefore starts from the strong `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack and adds two tightly scoped compression-aware changes:

1. **Compile-safe progressive int6 QAT** during the late phase of training.
2. **Activation-aware salient-row int8 rescue** at export time for a very small subset of rows in int6 matrices.

I also fold in the later **LeakyReLU(0.5)^2** MLP activation that proved effective in the current best TTT-backed record.

## Why this is promising for this repository

Repository evidence points in the same direction:

- The unlimited-compute 4-hour baseline substantially improved pre-quant metrics but still landed at only `1.2074` post-quant, which is a strong sign that **quantization/export remains a central bottleneck**.
- Earlier records showed that **QAT and mixed precision** can dramatically shrink the quantization gap, but the later `11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` record explicitly documented that its late-QAT flag was effectively dead under `torch.compile` constant folding.
- The best training-side record before TTT, `11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, already has a strong architecture and strong GPTQ-lite export path, so it is a good place to test a *real* late-stage low-bit training signal rather than adding more unrelated architecture changes.
- Recent records keep gaining from small, surgical improvements such as EMA, GPTQ-lite clip search, partial RoPE, LN scale, and LeakyReLU^2. This candidate follows that pattern instead of introducing a brand-new training stack.

## Which prior records influenced it

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Direct influences:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - motivated keeping the strong 11-layer XSA/partial-RoPE/LN-scale stack
  - highlighted that the old late-QAT mechanism was compile-fragile
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - motivated switching the MLP activation to `LeakyReLU(0.5)^2`
- `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`
  - demonstrated that QAT can help when the artifact is truly int6-constrained
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`
  - reinforced the hypothesis that more training alone is not enough if export quality still dominates

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

This candidate is intentionally a *repository-adapted* version of ideas from recent quantization papers, not a verbatim implementation of any one paper.

- **Learned Step Size Quantization (LSQ)** — <https://arxiv.org/abs/1902.08153>
  - foundational evidence that low-bit quantizers benefit from training-time exposure rather than pure PTQ
- **LLM-QAT: Data-Free Quantization Aware Training for Large Language Models** — <https://arxiv.org/abs/2305.17888>
  - supports the idea that low-bit LLMs improve substantially once QAT is added
- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** — <https://arxiv.org/abs/2306.00978>
  - motivates protecting a small subset of activation-salient weights instead of uniformly spending bits everywhere
- **Low-Rank Quantization-Aware Training for LLMs (LR-QAT)** — <https://arxiv.org/abs/2406.06385>
  - supports using lighter-weight, budget-conscious QAT variants rather than expensive full retraining machinery
- **EfficientQAT** — <https://arxiv.org/abs/2407.11062>
  - further evidence that practical QAT variants can beat PTQ under realistic constraints
- **Channel-Wise Mixed-Precision Quantization (CMPQ)** — <https://arxiv.org/abs/2410.13056>
  - motivates channel/row-level bit allocation rather than fixed per-tensor rules
- **Scaling Law for Quantization-Aware Training** — <https://arxiv.org/abs/2505.14302>
  - especially relevant because it argues that weight quantization error remains important as training grows, matching this repo’s export bottleneck
- **SiLQ: Simple Large Language Model Quantization-Aware Training** — <https://arxiv.org/abs/2507.16933>
  - important recent evidence that a *simple* end-to-end quantization path can be high leverage without large training overhead

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes four meaningful changes:

1. **Progressive, compile-safe QAT mix**
   - replaces the old boolean class-attribute late-QAT switch with a per-module tensor buffer (`qat_mix`)
   - ramps fake int6 weight quantization from `0 -> 1` as the learning-rate scale moves from `QAT_START_SCALE` to `QAT_FULL_SCALE`
   - recompiles the training graph once when QAT actually turns on, so `QAT_ENABLED=0` is a true no-QAT ablation and the early phase does not pay fake-quant compute overhead
   - avoids relying on a Python-side class flag that can be constant-folded away by `torch.compile`

2. **Activation-aware salient-row int8 rescue at export**
   - runs a short post-training calibration pass over training tokens
   - records per-row output RMS for every `CastedLinear`
   - for int6 matrices, keeps the usual GPTQ-lite int6 export for all rows, then overwrites only the most salient rows with int8 reconstructions
   - this is a lightweight approximation of activation-aware/channel-aware mixed precision tailored to the repository’s existing export format

3. **LeakyReLU(0.5)^2 MLP**
   - swaps the base `relu^2` MLP for the later `LeakyReLU(0.5)^2` activation

4. **Candidate-directory usability**
   - the default dataset and tokenizer paths resolve relative to the repository root, so the script can be run directly from inside this candidate directory as required

## Files added

- `candidates/202603281329_progqat-salientrows/train_gpt.py`
- `candidates/202603281329_progqat-salientrows/README.md`

## How to run

From the candidate directory:

```bash
cd candidates/202603281329_progqat-salientrows

SEED=1337 \
QAT_ENABLED=1 \
QAT_START_SCALE=0.30 \
QAT_FULL_SCALE=0.10 \
SALIENT_ROW_FRACTION=0.0125 \
SALIENT_ROW_MIN=1 \
SALIENT_ROW_MAX=8 \
SALIENT_ROW_CALIB_STEPS=8 \
SALIENT_ROW_CALIB_TOKENS=131072 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

- `DATA_PATH=/abs/path/to/data/datasets/fineweb10B_sp1024`
- `TOKENIZER_PATH=/abs/path/to/data/tokenizers/fineweb_1024_bpe.model`
- `QAT_ENABLED=0` to ablate the progressive QAT path
- `SALIENT_ROW_FRACTION=0` to ablate the export-time rescue path

## How to evaluate / ablate

Suggested first ablations:

```bash
# Base candidate
QAT_ENABLED=1 SALIENT_ROW_FRACTION=0.0125 torchrun --standalone --nproc_per_node=8 train_gpt.py

# QAT only
QAT_ENABLED=1 SALIENT_ROW_FRACTION=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Rescue only
QAT_ENABLED=0 SALIENT_ROW_FRACTION=0.0125 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Neither (closest to the chosen base, aside from LeakyReLU^2)
QAT_ENABLED=0 SALIENT_ROW_FRACTION=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks / tradeoffs

- **Training throughput risk**: the progressive fake-quant path still computes quantized weights every forward pass during training; if the step-time hit is too large, the extra low-bit robustness may not repay the lost optimization steps.
- **Artifact-size risk**: rescuing too many rows can eat the remaining byte budget quickly.
- **Calibration approximation risk**: this is not full AWQ/CMPQ; it uses a short output-stat calibration and row overwrite, which may be too crude or target the wrong rows.
- **Interaction risk**: LeakyReLU^2, EMA, GPTQ-lite export, and progressive QAT may not compose as cleanly as they do individually.

## Validation

I ran the following low-cost validation from this workflow environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603281329_progqat-salientrows/train_gpt.py
python -m compileall candidates/202603281329_progqat-salientrows/train_gpt.py
```

Outcome:

- Both compile passes succeeded.
- I could **not** run a real runtime smoke test here because the workflow Python environment does not have `torch` installed, and the candidate script is a CUDA training script intended for the repo’s GPU environment.
- Because of that limitation, the runtime path still needs first execution in a proper repo training environment (the same limitation applies to any import-level smoke test of this script here).
