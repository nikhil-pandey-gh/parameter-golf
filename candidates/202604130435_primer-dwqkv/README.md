# Candidate: Primer-style depthwise QKV mixing on the 11L EMA + GPTQ-lite base

## Hypothesis

The repo has already validated two strong themes for tiny models under the 16MB cap: **cheap local-context bias helps** (SmearGate, BigramHash, XSA) and **the strongest pure-training stack is the 11-layer EMA + GPTQ-lite recipe**. This candidate tests the missing half of **Primer**: add a tiny **causal depthwise convolution after each Q/K/V projection** so every attention layer gets a stronger short-range inductive bias with only ~34k extra parameters.

## Why this is promising here

- The best non-TTT base in the repo is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`.
- The repo's strongest trends all point toward local structure helping small models: SmearGate, BigramHash, sliding-window eval, XSA, and partial RoPE all stacked well.
- Primer (`arXiv:2109.08668`) reported that **squared ReLU + depthwise Q/K/V convolutions** consistently improve language-model efficiency. This repo already adopted the squared-ReLU side of that result; the depthwise-QKV half was still unexplored.
- Recurrent/depth-reuse ideas were considered during research, but the repo's own negative results on recurrence under a fixed 10-minute wall-clock budget made them less attractive for the next candidate.

## Prior work that influenced this candidate

- **Chosen base:** `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- **Nearby base:** `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
- **Key motifs retained from the record stack:** 11 layers, XSA on the last 4 layers, EMA, GPTQ-lite export search, partial RoPE, LN scaling, SmearGate, BigramHash, shared value embeddings, sliding-window eval
- **No prior `candidates/` directory existed** when this candidate was created

## External research that informed it

1. **Primer: Searching for Efficient Transformers for Language Modeling** (`arXiv:2109.08668`)  
   Main relevance: Primer attributes most of its gains to **squared ReLU** and **depthwise convolution after Q/K/V projections**. This candidate ports that Q/K/V mixing idea into the repo's strongest compact 11-layer stack.
2. **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** (`arXiv:1909.11942`) and **Universal Transformer** (`arXiv:1807.03819`)  
   These were considered as parameter-sharing/recurrent-depth directions during the research pass, but deprioritized here because the repo already contains negative evidence that naive recurrence loses too many optimizer steps under the strict wall-clock budget.

## What changed versus the chosen base implementation

1. Added `PRIMER_KERNEL_SIZE` (default `3`).
2. Inserted a **causal depthwise 1D convolution** after each attention `Q`, `K`, and `V` projection in every transformer block.
3. Kept FlashAttention when available, but added a **PyTorch SDPA fallback** so the script is less brittle in environments without `flash_attn_interface`.
4. Changed the default dataset/tokenizer paths to resolve from the candidate script location, so the trainer can be launched **from inside this candidate directory** without rewriting paths.

Everything else stays intentionally close to the 2026-03-22 base so the new signal is mostly the Primer-style QKV mixing.

## How to run

From the repository root:

```bash
cd candidates/202604130435_primer-dwqkv
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablation:

```bash
cd candidates/202604130435_primer-dwqkv
PRIMER_KERNEL_SIZE=1 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`PRIMER_KERNEL_SIZE=1` makes the new depthwise mixers identity layers, which is the cleanest built-in ablation against the copied 11-layer base.

## Expected risks and tradeoffs

- The extra sequence mixing may improve per-step quality but still lose overall if step time rises too much.
- BigramHash + SmearGate already inject local context, so the Primer-style QKV convolution may be partly redundant.
- The additional code bytes and tiny extra parameters slightly tighten artifact headroom, even though the parameter increase is small.

## Validation

Validated in this workflow with:

```bash
python -m compileall candidates/202604130435_primer-dwqkv/train_gpt.py
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604130435_primer-dwqkv/train_gpt.py
```

Both commands completed successfully.

A minimal CPU smoke run was **not feasible** in this workflow container because:

- the local Python environment does not have `torch` installed, and
- the checked-out repo does not include cached FineWeb shards or the tokenizer files under the default `data/` runtime paths.
