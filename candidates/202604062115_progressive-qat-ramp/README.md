# Progressive QAT Ramp on the 11L EMA/GPTQ-lite Base

## Hypothesis

The strongest training-only stack in this repo already squeezes a lot out of architecture and export-time quantization, but its fake-quant path is still a brittle late on/off switch. Replacing that switch with a compile-safe progressive QAT ramp should reduce the train-to-int6 roundtrip gap without disturbing the rest of the proven 11-layer recipe.

## Why this is promising here

- Repo history shows that export-aware training matters a lot once the model is already near the 16 MB ceiling.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly notes that a late-QAT flag was dead-code-eliminated by `torch.compile`.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` is the best training-only base in the repo and is already structured around EMA, GPTQ-lite export, and a late quantization regime.
- Recent QAT papers argue that simple end-to-end quantization in the training graph can be effective when it stays close to deployment-time quantization and avoids extra training infrastructure.

## Prior repository influence

This candidate is primarily based on:

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` for the 11L XSA/EMA/GPTQ-lite stack.
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` for the documented late-QAT compile caveat.
3. The broader repo trend that sliding eval, EMA, partial RoPE, XSA, and aggressive export quantization keep compounding, while naive recurrence and large infrastructure jumps are riskier under the 10-minute budget.

There were no prior `candidates/` runs in the repository when this candidate was created.

## External research that informed it

- **SiLQ: Simple Large Language Model Quantization-Aware Training** (arXiv:2507.16933) argues that simple end-to-end QAT with essentially no extra machinery can outperform heavier quantization recipes.
- **EfficientQAT** (arXiv:2407.11062) emphasizes block-aware QAT and better optimization of quantized models rather than relying purely on post-training recovery.
- **Scaling Law for Quantization-Aware Training** (arXiv:2505.14302) highlights that weight quantization error becomes more important as training data grows, which is directly relevant to this challenge’s many-token, small-model regime.

## What changed vs. the base implementation

Compared with the 2026-03-22 record script, this candidate keeps the same 11-layer architecture, EMA, GPTQ-lite mixed int6 export, XSA, partial RoPE, shared value embedding, and evaluation path. The only intended modeling change is the QAT mechanism:

1. `CastedLinear` now always contains a fake-int6 path in the graph during training.
2. Each linear layer gets a non-persistent `qat_mix` buffer.
3. Training ramps `qat_mix` from `0.0` to `1.0` as the LR scale falls from `QAT_RAMP_START` to `QAT_RAMP_END`, instead of toggling a late boolean.
4. The ramp is buffer-driven so the compiled graph does not depend on a mutable Python/class attribute.

## How to run

From the repository root:

```bash
QAT_RAMP_START=0.30 \
QAT_RAMP_END=0.05 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
  candidates/202604062115_progressive-qat-ramp/train_gpt.py
```

The script now resolves its default dataset and tokenizer paths relative to the repository root, so it can also be launched directly from the candidate directory without overriding `DATA_PATH` or `TOKENIZER_PATH`.

Useful inherited knobs from the base script still apply, including `BIGRAM_VOCAB_SIZE`, `XSA_LAST_N`, `EVAL_STRIDE`, `WARMDOWN_ITERS`, and `MAX_WALLCLOCK_SECONDS`.

## Expected risks and tradeoffs

- The fake-quant path now does real work during any step where training is active, so step time may rise modestly versus the abrupt late-switch baseline.
- The ramp is intentionally simple and uses the repo’s existing rowwise int6 approximation, not learned step sizes, so it may still leave some quantization gap on the table.
- Because the rest of the stack is unchanged, any gain is likely to be incremental rather than architectural-scale.

## Validation

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604062115_progressive-qat-ramp/train_gpt.py` — succeeded
- Minimal CPU smoke test — not feasible in this environment. A targeted import-level smoke attempt was blocked because the runner Python environment does not have `torch` installed, and the inherited runtime path still assumes the CUDA/FlashAttention stack from the chosen base implementation.

## Code review

Completed cleanly; the only issue found was the original README path guidance, which has been corrected.
