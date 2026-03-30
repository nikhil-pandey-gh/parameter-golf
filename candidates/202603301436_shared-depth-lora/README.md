# Shared-Depth LoRA Loops

## Hypothesis

Reuse a smaller set of strong 11L-stack transformer blocks across two passes, then recover loop-specific specialization with lightweight attention LoRA adapters. The core bet is that under a 16MB artifact budget, **sharing large attention/MLP weights is a more leverageable byte-saving move than squeezing a conventional stack a little harder**, as long as the second pass is allowed a small amount of dedicated capacity.

This candidate uses **6 unique blocks looped 2x** for an **effective depth of 12**, with loop-specific attention LoRA only on the second pass. It also keeps the strongest recent cheap biases: **LeakyReLU(0.5)^2**, **partial RoPE**, **effective-depth LN scaling**, **deep-only XSA**, **BigramHash + SmearGate**, **EMA**, and **GPTQ-lite-style int6 export**.

## Why it is promising for this repository

Recent wins in this repo show a clear pattern: the best runs stack low-parameter inductive biases and compression-aware tricks on top of a fairly fixed small-model backbone, rather than relying on one giant architectural rewrite. At the same time, the challenge is explicitly parameter- and artifact-limited, so a candidate that directly reduces the number of unique large matrices is well aligned with the objective.

The twist here is that earlier repo exploration suggested **naive recurrence / layer looping was not enough on its own**, so this candidate does **not** use pure weight reuse. Instead, it adds a small second-pass LoRA for attention and computes **LN scale, VE placement, and XSA placement in effective-depth space**, so the second pass behaves like a genuine later-stage refinement pass instead of a perfect copy of the first.

## Prior records and experiments that influenced this candidate

Primary local influences:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest clean non-TTT training stack in this repo snapshot
  - provided the base for EMA, GPTQ-lite-style clipping search, VE, XSA, partial RoPE, LN scale, BigramHash, and SmearGate
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - motivated replacing ReLU^2 with **LeakyReLU(0.5)^2**
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py`
  - contained a dormant looped/shared-depth path with per-loop attention LoRA scaffolding that made this candidate mechanically straightforward to revive
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - cautionary negative result: plain recurrence was not enough there, which is why this candidate adds loop-specific specialization rather than simple repeated blocks

## External research that informed it

- **ALBERT** — [arXiv:1909.11942](https://arxiv.org/abs/1909.11942)
  - motivates cross-layer parameter sharing as a practical way to reduce memory/parameter count while keeping depth
- **Universal Transformer** — [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)
  - motivates recurrent/refinement-style reuse of the same block over multiple passes
- **Exclusive Self Attention (XSA)** — [arXiv:2603.09078](https://arxiv.org/abs/2603.09078)
  - motivates keeping the repo's existing deep-layer XSA bias, here applied in effective-depth space so the second pass can play the role of late layers

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Main changes:

- changed defaults from **11 unique layers** to **6 unique layers, looped 2x**
- added **per-loop attention LoRA** (`Q/K/V/out`) for loops after the first pass
- changed the MLP activation from **ReLU^2** to **LeakyReLU(0.5)^2**
- moved **LN scaling**, **VE layer selection**, and **XSA placement** to **effective-depth indexing** rather than unique-block indexing
- added a small **FlashAttention -> SDPA fallback** so the module can still be imported for lightweight checks when flash-attn is unavailable
- changed default `DATA_PATH` and `TOKENIZER_PATH` to be **repo-relative**, so the script can be run directly from this candidate directory

## How to run / evaluate

From this candidate directory:

```bash
RUN_ID=shared_depth_lora \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Important defaults baked into this candidate:

- `NUM_LAYERS=6`
- `NUM_LOOPS=2`
- `LORA_RANK=8`
- `TRAIN_SEQ_LEN=2048`
- `BIGRAM_VOCAB_SIZE=2048`
- `XSA_LAST_N=4`
- `ROPE_DIMS=16`
- `VE_LAYERS=10,11`

The script still prints the same final metrics as the late 11L stack, including the quantized roundtrip score and sliding-window score. It also keeps the legacy `final_int8_zlib_roundtrip_exact` alias on the final sliding metric for compatibility with existing repo tooling, even though the actual artifact format here is `int6+zstd` (or `int6+zlib` if `zstandard` is unavailable).

## Main expected risks / tradeoffs

- **Repo evidence says naive recurrence can fail.** This candidate specifically tries to address that with second-pass LoRA plus effective-depth scheduling, but it is still the main uncertainty.
- **Step time may increase modestly** versus the 11-layer baseline stack, since effective depth is 12 rather than 11.
- **LoRA weights are small enough to survive export as fp16 passthrough tensors**, but they still consume some artifact bytes and may need rank tuning.
- **Best loop count is uncertain.** `2x` reuse is the safest initial setting; `3x` would save more bytes but is much riskier.
- If this works but underperforms, the obvious next ablations are:
  - `LORA_RANK` in `{0, 4, 8, 16}`
  - `NUM_LAYERS/NUM_LOOPS` in `{5x2, 6x2, 4x3}`
  - `BIGRAM_VOCAB_SIZE` up once export size is measured

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202603301436_shared-depth-lora/train_gpt.py
```

Outcome:

- `python -m compileall` **passed**

Attempted lightweight CPU smoke check:

```bash
python3 - <<'PY'
# import candidate module and run a tiny forward pass
PY
```

Outcome:

- not feasible in this container because the Python environment does **not** have the required runtime packages installed (`torch`, `numpy`, and `sentencepiece` were all missing)
- because of that limitation, I only validated syntax/bytecode generation here; the next practical check is to run the script in the normal repo training environment where dependencies already exist
