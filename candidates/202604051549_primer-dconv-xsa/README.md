# Primer DConv + XSA Late Stack

## Hypothesis

The current best stack already benefits from Primer's squared-activation family through **LeakyReLU(0.5)^2**, but it still leaves Primer's other core change unused: **depthwise causal convolution after Q/K/V projections**. Adding that local mixing only to the deepest attention blocks should improve phrase-level modeling with tiny parameter overhead and without disturbing the proven early-layer recipe.

## Why this is promising for this repository

The strongest Parameter Golf runs keep rediscovering the same two themes:

- local structure matters (`SmearGate`, `BigramHash`, sliding-window eval, TTT),
- tiny-model architecture choices matter more than they do at larger scales (11 layers, GQA, tied embeddings, careful quantization, weight averaging).

This candidate fits both trends. It preserves the March 23 stack and extends its local inductive bias from the embedding layer into the late attention stack with a very small number of extra parameters.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` — chosen base stack: LeakyReLU^2, parameter banking, Parallel Muon, legal score-first TTT.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` — GPTQ-lite, EMA, warmdown3500, partial RoPE/LN-scale/XSA/VE shape.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` — validated partial RoPE + LN scale on the 11-layer stack.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/` — strong evidence that cheap local bias helps this challenge.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` — useful negative result: naive recurrence hurt, so this candidate keeps depth fixed and adds only lightweight local mixing.

## External research that informed it

- **Primer: Searching for Efficient Transformers for Language Modeling** (So et al., NeurIPS 2021, arXiv:2109.08668) attributes most of Primer's gains to two simple modifications: squared ReLU and depthwise convolution after each Q/K/V projection. This repo has already validated the activation side; this candidate ports the missing QKV depthwise component.
- **MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases** (Liu et al., ICML 2024, arXiv:2402.14905) argues that deep/thin layouts, grouped-query attention, and tied embeddings are especially important in sub-billion models. That matches the current leaderboard direction and supports making a surgical architecture tweak rather than adding heavy new infrastructure.
- I also considered recent rotation-based quantization work such as **QuaRot** and **SpinQuant**, but those require much broader quantization infrastructure than this repository currently has.

## What changed vs the chosen base implementation

Base file:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Candidate changes:

1. **Late-layer Primer DConv on Q/K/V**
   - added `DepthwiseCausalConv1d`,
   - applied it after Q, K, and V projections in the last `DCONV_LAST_N` blocks,
   - defaulted to `DCONV_LAST_N=4` and `DCONV_KERNEL_SIZE=3`,
   - initialized the depthwise kernels to identity so training starts from the March 23 behavior and only departs when useful.
2. **FlashAttention fallback**
   - if `flash_attn_interface` is unavailable, attention falls back to `torch.nn.functional.scaled_dot_product_attention`,
   - this keeps the CUDA fast path when available, and provides a fallback path for PyTorch environments where the external FlashAttention module is missing,
   - `main()` still expects CUDA for actual training and evaluation runs.

No files outside this candidate directory were changed.

## How to run or evaluate it

From `candidates/202604051549_primer-dconv-xsa/`:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
DCONV_LAST_N=4 DCONV_KERNEL_SIZE=3 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For an architecture-only ablation, omit `TTT_ENABLED=1`.

## Main expected risks and tradeoffs

- The late depthwise convolutions add sequence-wise compute in the deepest layers, so a small step-time penalty is possible.
- The best placement is uncertain: `last 4` is a conservative default chosen to line up with the XSA-heavy part of the stack.
- Because the new depthwise kernels are kept at high precision rather than quantized, artifact size rises slightly.

## Validation

Commands run in this repository:

```bash
python3 -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604051549_primer-dconv-xsa/train_gpt.py
python3 -c 'import torch; print(torch.__version__)'
```

Outcomes:

- `compileall`: passed for the baseline scripts, `data/`, and this candidate script.
- Minimal CPU smoke run: not feasible in this container because `python3` does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
