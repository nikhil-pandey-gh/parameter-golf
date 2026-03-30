# Adaptive-Clip QAT on the 11L EMA + GPTQ-lite stack

## Hypothesis

The strongest non-TTT path in this repository already squeezes a lot out of depth, XSA, EMA, Partial RoPE, and export-time GPTQ-lite clip search, but it still pays a quantization gap between the EMA checkpoint and the final int6 artifact. This candidate tests whether **learned per-module int6 clip multipliers during late QAT**, combined with **feeding those learned clip suggestions back into export-time GPTQ-lite search**, can reduce that gap without adding meaningful artifact bytes or new training infrastructure.

## Why this is promising here

The record history says the repo keeps winning by spending compression savings on better model capacity and then shaving the export penalty.

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` showed that better clip selection at export (`GPTQ-lite`) and better quantization-awareness during training each still matter on top of the mature 11-layer stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` showed that the repo is already in the regime where small zero-byte changes can move BPB by a few 1e-3.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` showed that a one-line activation change like LeakyReLU(0.5)^2 can still improve the pre-TTT model quality, so it is a sensible training-side addition to keep while testing a new quantization idea.

The external research points in the same direction:

- **LSQ** argues that learning quantizer step sizes is a low-overhead way to improve low-bit training (`Learned Step Size Quantization`, arXiv:1902.08153).
- **TernaryLM** reports stable native low-bit LM training from adaptive layer-wise scaling (`TernaryLM`, arXiv:2602.07374).
- **Robust Training of Neural Networks at Arbitrary Precision and Sparsity** argues that low-bit training benefits from an explicit forward/backward path for quantization noise rather than a purely blind STE (`arXiv:2409.09245`).
- **Squat** specifically emphasizes that QAT remains especially relevant for small language models where full-parameter training is practical (`arXiv:2402.10787`).

I did **not** implement the more invasive parts of those papers, like native ternary training or token-adaptive precision, because those would require a much larger departure from this repository's current code path and are much less likely to fit the 10-minute iteration budget cleanly.

## Which prior records influenced this candidate

This candidate is intentionally a small fork of the strongest clean training stack rather than a brand-new architecture.

- **Primary base**: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Activation inspiration**: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Architectural context**: `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` and `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`

At the time this candidate was created, there was no existing `candidates/` directory to reuse.

## What changed versus the chosen base implementation

Relative to the 2026-03-22 11L EMA + GPTQ-lite record script, this candidate makes four targeted changes:

1. **Adaptive clip QAT in `CastedLinear`**  
   Every quantized linear layer now has a small scalar `qat_clip_logit` parameter. During QAT, that parameter is mapped into a bounded clip multiplier (default range `0.85..1.00`) and used to shrink or preserve the usual per-row int6 clipping range.

2. **Quantization path with an explicit gradient route**  
   Instead of `w + (w_q - w).detach()`, the fake-quantized forward path is written as `w_q + (w - w.detach())`. That keeps the standard straight-through gradient on the weights while still allowing the learned clip parameter to receive gradient through the quantized forward path.

3. **Learned-clip-guided GPTQ-lite export**  
   Export-time int6 quantization still runs the row-wise min-MSE search over the old percentile candidates, but it now also tests the learned clip multiplier from training as an extra candidate.

4. **Low-risk runtime improvements for local iteration**  
   The script now has a non-FlashAttention fallback, CPU-safe optimizer/compile guards, and a synthetic `SMOKE_MODE=1` path so local validation does not require the real FineWeb shards.

I also carried over **LeakyReLU(0.5)^2** in the MLP, because the latest record history suggests it is a strong, low-risk improvement that stacks naturally with this quantization experiment.

## How to run or evaluate it

Run from this directory:

```bash
cd candidates/202603300137_adaptive-clip-qat
```

The script resolves its default `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so the commands below work from inside this candidate directory without extra path overrides.

### Intended H100 training path

The defaults keep the same core 11-layer recipe as the 2026-03-22 base: EMA, XSA on the last 4 layers, Partial RoPE, LN scaling, SmearGate, BigramHash, late QAT, and GPTQ-lite-style export search.

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Optional local synthetic smoke path

This is only for catching obvious runtime regressions. It does **not** represent leaderboard evaluation.

```bash
TORCH_COMPILE=0 \
SMOKE_MODE=1 \
SMOKE_TOKENS=2048 \
QAT_ENABLED=1 \
SWA_ENABLED=0 \
ITERATIONS=1 \
WARMUP_STEPS=0 \
TRAIN_BATCH_TOKENS=128 \
VAL_BATCH_SIZE=256 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=1 \
TRAIN_SEQ_LEN=64 \
EVAL_SEQ_LEN=64 \
EVAL_STRIDE=32 \
VOCAB_SIZE=128 \
MODEL_DIM=64 \
NUM_HEADS=4 \
NUM_KV_HEADS=2 \
NUM_LAYERS=2 \
MLP_MULT=2 \
BIGRAM_VOCAB_SIZE=64 \
BIGRAM_DIM=32 \
XSA_LAST_N=0 \
ROPE_DIMS=8 \
VE_ENABLED=0 \
MAX_WALLCLOCK_SECONDS=0 \
RUN_ID=cpu_smoke \
python train_gpt.py
```

## Validation commands and outcomes

I ran the following lightweight checks in this workflow:

- `python -m compileall candidates/202603300137_adaptive-clip-qat/train_gpt.py`  
  Result: **passed**

- Synthetic CPU smoke command shown above  
  Result: **not runnable in this workflow container** because the local Python environment does not currently have the repository's declared runtime dependencies (`numpy`, `sentencepiece`, `torch`) installed, and the shell environment here does not have network access for `pip install`. The smoke path is implemented in the script, but I could not execute it in this container.

## Main expected risks or tradeoffs

- The learned clip scalars may help export quality, but they also introduce another late-training degree of freedom that could need retuning with the existing warmdown schedule.
- The LeakyReLU^2 swap and the adaptive clip path both encourage different weight distributions than the original base; the combination could improve quantization or could make EMA / GPTQ-lite interactions noisier.
- The CPU and non-FlashAttention fallbacks are for debugging and smoke checks only. They are not intended to match the performance characteristics of the H100 path.
- This is still a modest fork of the existing 11L recipe, not a fundamentally new architecture. If the remaining gap is now dominated by evaluation-time adaptation rather than export quality, the upside may be limited.
