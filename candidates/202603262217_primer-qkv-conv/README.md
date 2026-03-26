# Primer-style causal QKV depthwise convolution

## Hypothesis

The strongest records in this repository repeatedly benefit from adding cheap local inductive bias near the token interface: `SmearGate`, `BigramHash`, partial RoPE, and XSA all improved the same 11-layer stack. The hypothesis for this candidate is that a **tiny Primer-style causal depthwise convolution on the Q/K/V projections** can extend that trend from fixed bigrams to short learned n-grams, while staying small enough not to meaningfully hurt the 10-minute training budget.

## Why this is promising for this repository

- The best clean training base here is the 2026-03-22 record, which already stacks EMA, GPTQ-lite, Partial RoPE, LN scaling, XSA, VE, BigramHash, and SmearGate.
- Repo history suggests **small local-bias changes keep paying off**, while heavier recurrent/depth-reuse ideas have been risky under the fixed wall-clock budget.
- This change adds only a few thousand convolution parameters, so it is compatible with the repository's 16 MB artifact cap and should be cheap relative to the existing attention/MLP work.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen as the direct base because it is the strongest non-TTT stack in the repo and already includes the current winning architecture/quantization recipe.
- `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/`
  - Established that lightweight local token mixing (`SmearGate`) and lexical shortcuts (`BigramHash`) are useful here.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - Reinforced that the same family of local-bias features remained helpful in stronger later stacks.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Useful negative control: full layer recurrence was harmful under a fixed time budget, which pushed this candidate toward a much smaller local-mixing intervention instead of deeper recurrent compute.

## External research that informed it

- **Primer: Searching for Efficient Transformers for Language Modeling** (`arXiv:2109.08668`)
  - Primer reports that a depthwise convolution after each Q/K/V projection improves autoregressive language modeling while remaining simple enough to drop into existing Transformer codebases.
- The design here adapts only the smallest, easiest-to-port part of Primer: a **causal depthwise convolutional residual path on Q/K/V**, rather than broader architectural changes.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. Added `QKV_CONV_KERNEL` hyperparameter, defaulting to `3`.
2. Extended `CausalSelfAttention` with zero-initialized causal depthwise `Conv1d` layers on the Q, K, and V projection streams.
3. Applied those convolutions as a residual update before head reshaping and FlashAttention.
4. Kept the rest of the 2026-03-22 stack unchanged so the delta stays easy to reason about.
5. Updated optimizer grouping so the new 3D convolution weights are trained by the AdamW "scalar/non-matrix" path instead of being skipped.

## How to run

From this candidate directory:

```bash
RUN_ID=primer_qkv_conv \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
QKV_CONV_KERNEL=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate keeps the base record's defaults for the rest of the stack: 11 layers, 512 dim, XSA on the last 4 layers, Partial RoPE, LN scale, EMA, GPTQ-lite int6 export, and sliding-window evaluation.

## Main expected risks and tradeoffs

- Even a cheap convolution can reduce throughput if it blocks fusion or adds enough overhead at sequence length 2048.
- The extra local bias may overlap with what `SmearGate` and `BigramHash` already provide, so gains could saturate.
- Because this runner does not have PyTorch installed, I could only do syntax-level validation here, not a real forward/backward smoke run.

## Validation

- `python -m compileall candidates/202603262217_primer-qkv-conv/train_gpt.py`
  - Passed locally in this workflow.
- `python - <<'PY' ; import torch ; PY`
  - Failed in this workflow with `ModuleNotFoundError: No module named 'torch'`, so a CPU smoke test was **not feasible** here.
