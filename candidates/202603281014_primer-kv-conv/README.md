# Primer K/V Depthwise Conv on the 11L EMA + GPTQ-lite Base

## Hypothesis

The repository already benefits from the `relu^2` half of Primer-style transformer design, but it has not yet tried the other low-cost half: a tiny causal depthwise convolution after the projected attention streams.

This candidate adds a **Primer-style depthwise causal conv on K/V only**, applied to the **lower 6 layers** of the strong `2026-03-22` 11-layer stack. The bet is that a small amount of learned local token mixing inside attention will improve tiny-model language modeling without materially increasing artifact size.

## Why it is promising for this repository

- The best records already stack quantization, EMA/SWA, XSA, Partial RoPE, BigramHash, SmearGate, and shared-value reuse. The next win likely needs a **different inductive bias**, not just another small learning-rate or export tweak.
- Repo evidence says **full recurrence/layer looping was bad**, so the candidate avoids that risk while still adding a lightweight sequence-local mechanism.
- Primer reports that **depthwise conv after Q/K/V** and `relu^2` are the two main gains over vanilla transformers, and this repo already uses `relu^2`.
- The added parameters are tiny: depthwise kernels are `channels x kernel_size`, so they are negligible relative to the existing 11-layer model and should not meaningfully pressure the 16 MB artifact budget.

## Prior records and candidates that influenced this

There were no prior `candidates/` directories when this candidate was created.

The main repository influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Clean, strong pre-TTT 11-layer base with EMA, GPTQ-lite, Partial RoPE, XSA, BigramHash, SmearGate, VE, and tight quantization/export plumbing.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Confirmed that the leading stack is already very mature and that evaluation-time TTT only contributes a modest final gain relative to the strong pre-TTT model.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Useful negative evidence: broad recurrence changes were not attractive, so this candidate stays close to the winning transformer family.

## External research that informed this

- **Primer: Searching for Efficient Transformers for Language Modeling** (`arXiv:2109.08668`)
  - Primer attributes most of its gains to exactly two changes: `relu^2` and a depthwise conv after Q/K/V projections.
  - This repository already uses the `relu^2` half, making the missing depthwise-conv half a natural next experiment.
- **Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization** (`arXiv:2603.21676`)
  - Interesting for depth-reuse ideas, but repo-local evidence against recurrence made it a worse immediate fit.
- **SliderQuant: Accurate Post-Training Quantization for LLMs** (`arXiv:2603.25284`)
  - Reinforced that quantization sensitivity still matters here, but the repo already explores that axis heavily, so this candidate deliberately targets a more novel modeling bias instead.

## What changed versus the chosen base implementation

Base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. Added **Primer-style causal depthwise token mixing** modules with identity initialization.
2. Applied them to **K/V by default** via:
   - `PRIMER_CONV_MODE=kv`
   - `PRIMER_CONV_KERNEL=3`
   - `PRIMER_LAYERS=0,1,2,3,4,5`
3. Left the rest of the Mar 22 stack intact:
   - 11 layers
   - EMA + GPTQ-lite export path
   - XSA on late layers
   - Partial RoPE
   - SmearGate + BigramHash
   - VE on late layers
4. Added a **CPU smoke mode** (`SMOKE_TEST=1`) that:
   - builds a tiny synthetic model with CPU-safe equal Q/KV heads,
   - runs a forward pass,
   - executes the quantize/dequantize roundtrip,
   - avoids dataset/tokenizer setup.
5. Added an explicit **SDPA fallback** when `flash_attn_interface` is unavailable so the smoke path can exercise attention without FlashAttention.

## How to run or evaluate it

From inside this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
PRIMER_CONV_MODE=kv PRIMER_CONV_KERNEL=3 PRIMER_LAYERS=0,1,2,3,4,5 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional smoke check in a dependency-complete environment:

```bash
SMOKE_TEST=1 python train_gpt.py
```

Suggested immediate follow-ups:

- `PRIMER_CONV_MODE=qkv`
- `PRIMER_CONV_KERNEL=5`
- Move Primer conv to only the shallowest 4 layers or all 11 layers
- Trade some BigramHash capacity against the new local mixing if step time stays flat

## Validation

Commands attempted in this workflow:

```bash
python -m compileall candidates/202603281014_primer-kv-conv/train_gpt.py
SMOKE_TEST=1 python candidates/202603281014_primer-kv-conv/train_gpt.py
```

Outcomes:

- `python -m compileall ...` **succeeded**
- `SMOKE_TEST=1 python ...` could not be completed in this workflow container because the local Python environment does **not** have the repository runtime dependencies installed (`torch`, `numpy`, and `sentencepiece` were all missing), and this environment does not provide networked package installation through the shell tool. The candidate script now includes a dedicated smoke path so it can be exercised quickly in a dependency-complete environment.

## Main expected risks or tradeoffs

- Even a tiny depthwise conv adds some per-step overhead; if the added local bias is not strong enough, the net effect could be negative under a strict 10-minute wallclock.
- BigramHash and Primer-style local mixing may overlap; gains may require retuning their balance rather than simply stacking both unchanged.
- Applying the conv only to K/V in lower layers is intentionally conservative; the best placement may differ.
