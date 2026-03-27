# Candidate: Training-Only Multi-Token Prediction on the 11L EMA + GPTQ-lite Stack

## Hypothesis

A small **multi-token prediction (MTP)** auxiliary loss can improve **sample efficiency** in the repository's fixed 600-second regime by teaching the model to predict farther-ahead tokens during training, while keeping the final artifact almost unchanged because the auxiliary heads are stripped before export.

In this candidate, the model trains with **2 auxiliary future-token heads** by default (`MTP_NUM_HEADS=2`, `MTP_LOSS_WEIGHT=0.15`), but still exports the same next-token model family used by prior records.

## Why this is promising for this repository

The strongest existing records have already squeezed a lot out of this challenge through quantization, compression-aware training, EMA/SWA, partial XSA, partial RoPE, better activation choices, and evaluation tricks. What is still comparatively underexplored here is **getting more learning signal per training step** without paying permanent artifact bytes.

That makes MTP attractive for this leaderboard setting:

- the training budget is wall-clock-limited,
- later-record stacks are already close to the 16 MB cap,
- and the candidate can use extra training-only parameters that are **excluded from the exported checkpoint**.

## Prior records and repo evidence that informed this candidate

This candidate is based directly on:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

That record is the cleanest strong **pure training/export** stack in the repo: 11 layers, 3x MLP, SmearGate, BigramHash, XSA, partial RoPE, LN scale, VE, EMA, tight SWA, GPTQ-lite, and warmdown 3500.

Other repo evidence that shaped the choice:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` showed that late-stage gains have come from **targeted additions** on top of the strong 11L stack, not from wholesale rewrites.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` reported **layer recurrence as a negative result**, which pushed this candidate away from ALBERT/Universal-Transformer-style reuse as the next experiment.
- No prior record README appears to actually run with non-zero `MTP_NUM_HEADS`, even though support already exists in several codepaths.

## External research that informed this candidate

- **Better & Faster Large Language Models via Multi-token Prediction** (Gloeckle et al., 2024): https://arxiv.org/abs/2404.19737
- **DeepSeek-V3 Technical Report** (DeepSeek-AI, 2024/2025), which explicitly uses a multi-token prediction training objective: https://arxiv.org/abs/2412.19437

Why these matter here:

- Gloeckle et al. argue that MTP improves sample efficiency by supervising several future positions from the same hidden state.
- DeepSeek-V3 is useful as evidence that MTP is not just a small-paper curiosity; it is also used in a modern frontier training stack.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Enabled MTP by default**
   - `MTP_NUM_HEADS`: `0 -> 2`
   - `MTP_LOSS_WEIGHT`: `0.2 -> 0.15`

2. **Moved MTP heads to the output-head optimizer path**
   - The base code treated MTP heads like general trunk matrices.
   - This candidate trains them with the same optimizer family used for output/logit-style heads (`head_lr` path), which is a cleaner fit for a training-only auxiliary classifier.

3. **Preserved export stripping**
   - `mtp_heads.*` parameters are still removed from `export_sd` before serialization, quantization, and eval-model reconstruction.

## How to run

From this candidate directory, a single-GPU CUDA run should follow the same `torchrun` pattern as the repository baseline. The script resolves its default `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so these commands work unchanged from the candidate folder:

```bash
RUN_ID=mtp_aux_heads \
SEED=1337 \
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=9000 \
VAL_LOSS_EVERY=2000 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

For an 8-GPU run matching the repository's normal CUDA path, use the same environment that prior record folders use, for example:

```bash
RUN_ID=mtp_aux_heads \
SEED=1337 \
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=9000 \
VAL_LOSS_EVERY=2000 \
EVAL_STRIDE=64 \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected risks and tradeoffs

- **Step-time overhead**: auxiliary heads add extra logits and CE losses, so sample-efficiency gains must outweigh any loss in step count.
- **Objective mismatch**: too much MTP weight could improve farther-ahead prediction while slightly hurting the exact next-token objective used for evaluation.
- **Optimizer sensitivity**: head learning rate may matter more once MTP is enabled because those heads are now trained explicitly as output heads.
- **Unverified 8xH100 tuning**: this is a research candidate, not a measured record submission.

## Validation run for this candidate

Commands executed locally in this workflow:

```bash
python -m compileall candidates/202603270516_mtp-aux-heads/train_gpt.py
```

Outcome:

- **Passed**.

Attempted smoke validation:

```bash
python - <<'PY'
import importlib.util
print(importlib.util.find_spec('torch'))
PY
python3 - <<'PY'
import importlib.util
print(importlib.util.find_spec('torch'))
PY
```

Outcome:

- Both interpreters reported `None` for `torch`, even though `requirements.txt` lists it.
- Because the active runner image does not have a local PyTorch runtime installed, a CPU import/forward smoke test was **not feasible in this workflow** without adding new infrastructure.
