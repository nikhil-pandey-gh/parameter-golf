# Late LSQ Int6 on the 11L GPTQ-lite Backbone

## Hypothesis

The current non-TTT frontier in this repo already appears close to saturated on architecture and scheduler tweaks, but it is still **quantization-sensitive**. The hypothesis here is that a **late, row-wise LSQ-style int6 QAT pass** can learn export-friendly per-row scales during warmdown, so the strongest existing 11-layer backbone gives up less quality when it is packed into the 16 MB artifact budget.

In short: keep the proven 11L/XSA/Partial-RoPE/VE/EMA/GPTQ-lite recipe, but replace the repo's static late fake-quant path with a **learned scale path** that is closer to the final exporter.

## Why this looks promising for this repository

Repository review suggests three strong trends:

- the biggest durable gains came from better **artifact-aware quantization** rather than from the baseline 9-layer architecture alone,
- the best stable non-TTT stack is the 11-layer GPTQ-lite / EMA / Partial-RoPE family,
- the previous late-QAT attempt in the 1.1248 record was explicitly a **no-op** because `torch.compile` constant-folded the class flag.

That makes quantization-aware training with a more principled quantizer a high-leverage next move:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` showed that the existing late-QAT path did not actually fire.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed that export-side quantization tuning still mattered on the best non-TTT core stack.
- External research on **LSQ** and **LSQ+** argues that learning quantizer scales directly is often better than holding them fixed.

## Prior records that influenced this candidate

Primary base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Additional evidence that shaped the choice:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` for the late-QAT constant-folding failure.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` for the conclusion that the best remaining low-complexity gains are small, making quantization quality even more important.
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md` for the earlier evidence that quantization/export quality can dominate final BPB.

## External research that informed it

Primary sources:

- **LSQ**: *Learned Step Size Quantization* — https://arxiv.org/abs/1902.08153
- **LSQ+**: *Improving low-bit quantization through learnable offsets and better initialization* — https://arxiv.org/abs/2004.09576

What I borrowed from that literature:

- quantizer scales should be trainable parameters, not fixed clip heuristics,
- scale gradients should be normalized so learned steps stay well-behaved,
- good initialization matters, so this candidate initializes row-wise scales from the current weight magnitudes rather than from an arbitrary constant.

## What changed versus the chosen base implementation

This candidate starts from the 2026-03-22 11L GPTQ-lite backbone and makes the following focused changes:

1. **Row-wise LSQ-style learned scales inside `CastedLinear`**
   - every `CastedLinear` now owns a trainable `lsq_scale` parameter,
   - the forward pass builds an int6 fake-quantized weight with STE + LSQ-style gradient scaling,
   - late activation is controlled through a host-side module flag instead of the prior class-level flag, so eager training only pays the fake-quant cost after activation.

2. **Late LSQ activation during warmdown**
   - defaults enable the learned quantizer only once the LR multiplier falls below `LSQ_START_THRESHOLD=0.18`,
   - the intent is to keep the strong fp/bf16 optimization dynamics early, then bias the final weights toward export-friendly row scales.
   - training compilation is intentionally **disabled** so the late LSQ branch is actually late in runtime cost instead of being traced from step 0.

3. **Exporter reuses learned scales as candidates**
   - the existing GPTQ-lite per-row percentile search is kept,
   - the learned LSQ scale for each matrix row is added as an extra export candidate,
   - export chooses whichever candidate gives the lower reconstruction error.

4. **Safer attention fallback**
   - if `flash_attn_interface` is unavailable, the script falls back to `torch.nn.functional.scaled_dot_product_attention`,
   - this keeps the candidate easier to exercise outside the exact H100 runtime.

5. **Local smoke entry point**
   - `SMOKE_TEST=1 python train_gpt.py` runs a tiny CPU-only model forward/backward/export roundtrip,
   - this is meant for local debugging and static CI checks, not for score reporting.

## How to run / evaluate

From this candidate directory:

```bash
RUN_ID=late_lsq_int6 \
SEED=1337 \
LSQ_ENABLED=1 \
LSQ_START_THRESHOLD=0.18 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_WD=0.04 ADAM_WD=0.04 MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a tiny local structural check (with PyTorch installed):

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation run for this candidate

Commands attempted in this workflow:

```bash
python -m compileall candidates/202603290136_late-lsq-int6/train_gpt.py
SMOKE_TEST=1 python candidates/202603290136_late-lsq-int6/train_gpt.py
```

Outcomes:

- `python -m compileall ...` **passed**.
- `SMOKE_TEST=1 ...` could not be completed in this runner because the environment does not have the repository's training dependency stack installed (`torch` is missing, and the earlier attempt also showed missing optional dataset/tokenizer deps). The script includes the smoke path for environments where the repo dependencies are available.

## Main expected risks / tradeoffs

- **Training overhead:** even late fake quantization still adds forward-pass work, so step time may regress.
- **Eager training tradeoff:** this candidate leaves training compilation off so the late-LSQ branch is only paid for after activation; that preserves the hypothesis being tested, but it may cost some throughput versus the fully compiled record stack.
- **Scale mismatch risk:** LSQ may learn scales that help training loss but do not beat GPTQ-lite at export time; that is why the exporter still keeps GPTQ-lite candidates in the loop.
- **Optimizer-state overhead:** the learned row scales add scalar optimizer state, which is tiny compared with model weights but not free.
- **No empirical score yet in this workflow:** this candidate is code-complete and statically validated, but it has not been benchmarked in a full challenge runtime from this environment.
