# Hadamard GPTQ-lite + real late QAT + LeakyReLU²

## Hypothesis

The current strong 11-layer int6 stack is still quantization-limited. A **block-Hadamard rotation before GPTQ-lite rowwise int6 quantization** should make weight coordinates more uniform and reduce post-training quantization error at essentially zero training-time cost. Porting in **LeakyReLU(0.5)²** and making **late QAT actually activate** should further narrow the train/eval quantization gap.

## Why this is promising here

The repo’s best 10-minute results already converge on:

- 11L / 512d / GQA / 3x MLP
- EMA + warmdown3500
- partial RoPE + LN scale + VE + XSA
- aggressive int6 export with GPTQ-lite clip search

That means the next clean win is likely to come from a **better int6 export path**, not a wholesale architecture rewrite. This candidate is intentionally aimed at the same bottleneck.

## Prior repo work that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strong clean GPTQ-lite + EMA + warmdown3500 stack
- **Activation port:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - LeakyReLU(0.5)² was the cheapest recent gain in the repo
- **Late-QAT caveat:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - documents that compile-time constant folding made late QAT a no-op in at least one strong record

There were **no prior `candidates/` runs** in the repository when this candidate was created.

## External research

- **SpinQuant: LLM quantization with learned rotations** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406))
  - shows that orthogonal rotations can materially improve low-bit LLM quantization by suppressing outliers
- **PolarQuant: Optimal Gaussian Weight Quantization via Hadamard Rotation for LLM Compression** ([arXiv:2603.29078](https://arxiv.org/abs/2603.29078))
  - especially relevant here: reports that **Hadamard rotation alone accounts for most of the quantization gain**
- **A Survey on Transformer Compression** ([arXiv:2402.05964](https://arxiv.org/abs/2402.05964))
  - useful framing for why lightweight post-training quantization upgrades are attractive under strict artifact/time budgets

## What changed vs. the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate adds:

1. **Block-Hadamard GPTQ-lite**
   - rowwise int6 clip search now quantizes in a fixed block-Hadamard basis
   - default: `HADAMARD_GPTQ=1`, `HADAMARD_BLOCK_SIZE=64`
   - the inverse rotation is applied during dequantization, so runtime math stays unchanged
2. **LeakyReLU(0.5)² MLP**
   - ports the strongest recent cheap activation tweak from the current repo leader
3. **Real late QAT handoff**
   - late QAT is toggled on the actual `CastedLinear` modules
   - when it activates, training switches from compiled mode to eager mode so the fake-quant path cannot be constant-folded away
4. **Candidate-local ergonomics**
   - default dataset/tokenizer paths resolve from the repository root, so running from this candidate directory works
   - logs and exported artifacts are written back into this candidate directory, not the caller’s working directory
   - safe non-FlashAttention fallback for environments without `flash_attn_interface`
   - `SMOKE_TEST=1` CPU path for quick startup/quantization validation

## How to run

From this candidate directory:

```bash
SEED=1337 \
HADAMARD_GPTQ=1 \
HADAMARD_BLOCK_SIZE=64 \
MLP_NEGATIVE_SLOPE=0.5 \
LATE_QAT_THRESHOLD=0.15 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The rest of the defaults follow the 2026-03-22 record stack:

- 11 layers, 512 width, 8 heads / 4 KV heads
- 3x MLP
- batch 786,432, train/eval seq_len 2048
- EMA + warmdown3500
- XSA on last 4 layers
- partial RoPE (16 dims) + LN scale
- VE on layers 9 and 10
- stride-64 sliding evaluation

## Validation

Validated locally with lightweight checks:

1. `python -m compileall candidates/202604082147_hadamard-gptq-lite/train_gpt.py`
   - **Passed**
2. `SMOKE_TEST=1 python candidates/202604082147_hadamard-gptq-lite/train_gpt.py`
   - **Passed** in a temporary virtualenv because the runner initially lacked `torch`
   - output: `smoke_test:ok loss:4.8575 hadamard:1 block:64`

## Expected risks / tradeoffs

- **Rotation block size may matter.** `64` is a safe default, but `128` or `256` may quantize some matrices better.
- **Hadamard can improve error while hurting compressibility.** Better int6 reconstruction is the goal, but zstd size could move slightly in either direction.
- **Late-QAT eager handoff trades speed for correctness.** It fixes the “dead QAT” failure mode, but may cost a small number of final steps.
- **No 8xH100 result yet.** This is a high-conviction candidate built from repo evidence plus recent quantization work, but it still needs a real leaderboard-style run.
