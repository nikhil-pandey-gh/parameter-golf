# GPTQ-lite + selective low-rank residual sidecars

## Hypothesis

The current 11-layer GPTQ-lite stack is already strong, but its remaining quantization gap is probably concentrated in a small number of late, error-sensitive matrices. If we keep the existing int6 per-row export and add tiny float16 low-rank residual sidecars only for the highest-value late-layer matrices, we should recover some post-quantization quality without blowing the 16MB artifact budget.

## Why this is promising for this repository

The repository history keeps showing that export quality matters almost as much as training quality:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved the post-quant score with a better clip search, without changing the core model.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` pushed the stack further, but its artifact is already tight enough that any new export trick has to be byte-efficient.
- Earlier mixed-precision records repeatedly showed that preserving a small subset of sensitive weights in higher precision can beat uniform quantization.

This candidate therefore targets the narrowest remaining bottleneck: retain the strong 11-layer recipe, but spend the remaining artifact budget on the few matrices whose quantization residual looks most structured and recoverable.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - Base implementation copied here.
  - Supplies the 11-layer, EMA, partial-RoPE, XSA, BigramHash, VE, late-QAT recipe plus GPTQ-lite export.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - Demonstrates that the frontier is now close enough that small post-training gains can matter.
  - Motivated keeping the change focused and budget-aware instead of redesigning the whole model.
- The broader int6 / mixed-precision record line from `2026-03-19` through `2026-03-21`
  - Reinforced that post-training compression quality is a first-class optimization target in this challenge.

## External research that informed it

- Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
  - Motivates stronger post-training weight quantization rather than retraining-heavy changes.
- Dettmers et al., [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
  - Highlights that a tiny set of outlier dimensions can dominate quantization error, so selective high-precision protection can be much more efficient than uniform precision increases.
- Saxena et al., [ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals](https://arxiv.org/abs/2412.14363)
  - Directly motivates representing the remaining quantization error with a small low-rank high-precision correction instead of storing whole tensors at higher precision.
- Pandey et al., [QMC: Efficient SLM Edge Inference via Outlier-Aware Quantization and Emergent Memories Co-Design](https://arxiv.org/abs/2601.14549)
  - Strengthens the small-model-specific case for outlier-aware quantization under strict memory budgets.

## What changed versus the base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes four targeted changes:

1. It keeps the same training architecture and optimization defaults.
2. It adds a **selective residual sidecar export path**:
   - quantize late-layer attention/MLP matrices exactly as before,
   - measure the remaining residual,
   - fit a tiny rank-`RESIDUAL_RANK` float16 low-rank correction,
   - keep only the top `RESIDUAL_MAX_MATRICES` candidates under `RESIDUAL_MAX_BYTES`.
3. It adds a CPU-safe attention fallback using `torch.nn.functional.scaled_dot_product_attention` when FlashAttention 3 is unavailable.
4. It adds `SMOKE_TEST=1` so the candidate can be validated on CPU without data shards or CUDA.

## How to run or evaluate it

Training/evaluation on the full stack still follows the repository's usual pattern from the candidate directory:

```bash
cd candidates/202603250600_gptq-residual-sidecars
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful sidecar knobs:

```bash
RESIDUAL_RANK=2
RESIDUAL_MAX_MATRICES=12
RESIDUAL_LAST_N_LAYERS=4
RESIDUAL_MAX_BYTES=196608
RESIDUAL_MIN_GAIN_RATIO=0.01
```

CPU-only smoke test:

```bash
cd candidates/202603250600_gptq-residual-sidecars
SMOKE_TEST=1 \
NUM_LAYERS=2 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 \
TRAIN_SEQ_LEN=32 BIGRAM_VOCAB_SIZE=128 BIGRAM_DIM=32 VE_ENABLED=0 XSA_LAST_N=1 \
python train_gpt.py
```

## Validation

Commands run while preparing this candidate:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603250600_gptq-residual-sidecars/train_gpt.py
SMOKE_TEST=1 \
NUM_LAYERS=2 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 \
TRAIN_SEQ_LEN=32 BIGRAM_VOCAB_SIZE=128 BIGRAM_DIM=32 VE_ENABLED=0 XSA_LAST_N=1 \
python candidates/202603250600_gptq-residual-sidecars/train_gpt.py
```

Observed outcome in this workflow environment:

- `python -m compileall train_gpt.py train_gpt_mlx.py data` passed.
- `python -m compileall candidates/202603250600_gptq-residual-sidecars/train_gpt.py` passed.
- The CPU smoke command could not be completed in this container because the runner does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).

The candidate still includes the `SMOKE_TEST=1` path plus the CPU-safe SDPA attention fallback, so the smoke command above should be runnable in a normal repository environment where PyTorch is installed.

## Main risks and tradeoffs

- The residual factors are chosen from quantization error, not direct validation loss, so the best reconstruction-bytes tradeoff might not perfectly match BPB.
- Even tiny sidecars still spend real artifact bytes; if they do not compress as well as expected, the candidate may need tighter `RESIDUAL_MAX_BYTES` or fewer selected matrices.
- Full SVD at export time is slower than plain GPTQ-lite, though still far cheaper than adding new training infrastructure.
- The CPU fallback is meant for validation and portability, not to mimic the exact Hopper/FA3 runtime characteristics of the full training setup.
