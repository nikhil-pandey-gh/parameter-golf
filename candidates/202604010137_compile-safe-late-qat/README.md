# Compile-Safe Late QAT on the 11L EMA + GPTQ-lite Stack

## Hypothesis

The strongest near-term opportunity in this repository is not another broad architecture rewrite, but making **late fake-quantization actually execute** on the strongest pre-TTT stack. Prior records repeatedly show that post-training quantization is the main bottleneck, and the `2026-03-21` README explicitly documents that its late-QAT path was dead-code-eliminated under `torch.compile`. This candidate keeps the strong `2026-03-22` architecture and changes only the late-QAT control flow so the int6 STE path is retraced and activated when warmdown reaches the configured threshold.

## Why this is promising for this repository

Repository evidence points to the same pattern over and over:

- compression-aware training matters a lot more than raw pre-quant loss,
- QAT can help when it is integrated well,
- but compile-fragile toggles can silently erase the intended gain.

This candidate directly targets that bottleneck with a smaller implementation delta than a brand-new architecture, while still differing materially from the existing records: the late-QAT branch is now **compile-safe by construction**.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen as the base because it is the strongest pre-TTT stack and already combines 11 layers, EMA, GPTQ-lite clip search, partial XSA, partial RoPE, BigramHash, and the warmdown schedule that the repo has converged on.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Its README explicitly notes that late QAT never activated because `torch.compile` constant-folded the toggle.
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`
  - Reinforces that quantization behavior, not just float-model quality, dominates the artifact-constrained leaderboard.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Confirms that the best recent gains are now small, additive refinements on top of already strong stacks.

## External research that informed it

- **Jacob et al., “Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference”** (`arXiv:1712.05877`)
  - Motivates fake-quant-in-the-loop training as a direct way to preserve low-bit inference quality.
- **Esser et al., “Learned Step Size Quantization”** (`arXiv:1902.08153`)
  - Reinforces that quantizer-aware training can recover much of the low-precision accuracy gap when the training graph actually includes the quantization path.
- **Frantar et al., “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers”** (`arXiv:2210.17323`)
  - Supports the repo’s existing direction of pushing more quality into the quantizer/export path with minimal runtime cost.

## What changed versus the chosen base implementation

This candidate starts from the `2026-03-22` record script and makes two targeted changes:

1. **Compile-safe late QAT enablement**
   - Replaces the global `CastedLinear._qat_enabled` class toggle with per-module QAT state.
   - When late QAT should activate, the script explicitly recompiles the training model once so `torch.compile` retraces the fake-quant branch instead of keeping the earlier no-QAT trace.

2. **Candidate-directory-safe default paths**
   - Default dataset and tokenizer paths are now resolved relative to the repository root using `__file__`, so the script can be launched from inside this candidate directory without rewriting `DATA_PATH` or `TOKENIZER_PATH`.

Everything else is intentionally kept aligned with the strong `2026-03-22` stack so the experiment isolates the late-QAT fix.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202604010137_compile-safe-late-qat
RUN_ID=compile_safe_late_qat \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

- `LATE_QAT_THRESHOLD=0.15` keeps the intended record-style activation point.
- `QAT_ENABLED=1` forces fake quantization from step 0 for ablations.
- `DATA_PATH` and `TOKENIZER_PATH` can still be overridden explicitly, but the defaults now resolve correctly from this directory.

## Main expected risks and tradeoffs

- The one-time recompilation may cost a small amount of wallclock near warmdown.
- If late QAT meaningfully increases per-step cost, some of the quantization win could be offset by fewer total steps.
- The underlying stack is already heavily tuned, so gains may be small even if the fix is correct.
- This candidate intentionally does **not** add TTT or the top record’s parameter banking, because the goal is to isolate whether a working late-QAT path can improve the strongest simpler stack.

## Validation

Commands run in this workspace:

```bash
python -m compileall \
  train_gpt.py \
  train_gpt_mlx.py \
  data \
  candidates/202604010137_compile-safe-late-qat/train_gpt.py
```

Outcome:

- `compileall`: passed for `train_gpt.py`, `train_gpt_mlx.py`, `data/`, and `candidates/202604010137_compile-safe-late-qat/train_gpt.py`.

A CPU smoke run was **not** feasible in this environment because this script depends on challenge-specific training assets and GPU-oriented runtime pieces such as `flash_attn_interface`, which are not available for a safe no-infrastructure local launch here.
