# LSQ-lite Row-Clip QAT

## Hypothesis

The current 11-layer record line keeps improving when export quantization gets better, but the only recent "late QAT" attempt that should have helped was later shown to be dead code under `torch.compile`. This candidate makes late quantization-aware training real by using a compile-safe, always-traced LSQ-lite surrogate during training and then reusing the learned row clip gains during int6 export.

The bet is simple: for this repository, the next gain is more likely to come from shrinking the train/export mismatch than from another evaluation-only trick.

## Why this is promising here

Three repo trends point in the same direction:

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` improved the stack with better post-training clipping alone.
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly documents that its late-QAT path was constant-folded away by `torch.compile`, so the idea was never actually tested.
3. The top stacks are already strong architecturally; a training-time quantization fix is a narrower and safer change than another large model rewrite.

## Prior repo runs that influenced this candidate

- **Chosen base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- **Bug/motivation source:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
- **Current SOTA context:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`

This candidate keeps the proven 11L XSA + partial RoPE + LN scale + VE + EMA + GPTQ-lite backbone from the 2026-03-22 record and changes only the quantization-training path.

## External research that informed it

- **Learned Step Size Quantization (LSQ)** argues that learning quantizer step sizes directly during training can preserve low-bit accuracy much better than fixed post-hoc scales: https://arxiv.org/abs/1902.08153
- **PACT** shows that learned clipping parameters can materially improve low-bit behavior by tuning the clipping range rather than freezing it in advance: https://arxiv.org/abs/1805.06085

This implementation is not a full paper-faithful LSQ port. It is an LSQ-lite adaptation for this repo: learn row-wise clip gains on top of the existing per-row int6 export path, and only rely on mechanics that fit the current self-contained `train_gpt.py`.

## What changed vs the base implementation

1. Each `CastedLinear` now owns a learned per-row `qat_log_gain`.
2. Training uses a branchless fake-quant surrogate in `forward()` so `torch.compile` cannot dead-code-eliminate the quant path.
3. `qat_mix` ramps from 0 to 1 during warmdown (`QAT_RAMP_START` -> `LATE_QAT_THRESHOLD`) instead of flipping a Python/class flag late in training.
4. Int6 export can reuse the learned row clip gains instead of always running percentile search.
5. New QAT controls are exposed through:
   - `QAT_ENABLED`
   - `QAT_RAMP_START`
   - `LATE_QAT_THRESHOLD`
   - `QAT_MIN_GAIN`

## How to run

From the candidate directory:

```bash
cd candidates/202604011851_lsq-lite-rowclip-qat
RUN_ID=lsq_lite_rowclip \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides for ablations:

```bash
QAT_ENABLED=0
QAT_RAMP_START=0.30
LATE_QAT_THRESHOLD=0.15
QAT_MIN_GAIN=0.25
```

## Main expected risks and tradeoffs

- The fake-quant surrogate is always traced during training, so it may cost some throughput even before the ramp reaches non-zero mix.
- Learned clip gains are a better fit for the existing export path than for the current optimizer grouping; they are intentionally lightweight, but this is still a new control-parameter family.
- The candidate is trying to improve **post-quantized** performance, so pre-quant metrics may not move much even if export quality does.
- This is deliberately narrower than the latest TTT-heavy record; if it works, the natural follow-up is to combine it with the newer activation/eval stack rather than treat it as a competing endpoint.

## Validation

- `python -m compileall candidates/202604011851_lsq-lite-rowclip-qat/train_gpt.py` — **passed**
- Minimal CPU smoke test — **not feasible in this environment**. The available Python environment does not have `torch` installed, and this script also hard-requires CUDA plus `flash_attn_interface`, so a meaningful CPU-only launch could not be performed here without adding new infrastructure.
