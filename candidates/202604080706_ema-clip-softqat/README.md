# EMA-Clip SoftQAT

## Hypothesis

The current 11-layer EMA + GPTQ-lite stack is already strong on pre-quant quality, so the next practical gain is to make the final exported int6 model less sensitive to late-stage weight outliers and row-scale drift. This candidate replaces the repo's hard late-QAT toggle with a compile-stable, out-of-graph **quantization-consistency ramp** that only turns on during the late warmdown, and combines it with the recent **LeakyReLU(0.5)^2** MLP activation win.

## Why this is promising here

Local repo evidence points in the same direction:

- `2026-03-19_WarmdownQuantization` argued that smoother late-stage weights materially reduce the post-quantization penalty.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` documented that its class-flag late-QAT path did not actually activate under `torch.compile`.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved further with GPTQ-lite clip search, EMA, and an earlier late-QAT threshold, which suggests the export path is still sensitive to quantization quality.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` showed that LeakyReLU(0.5)^2 is worth a meaningful pre-TTT gain on a stronger stack.

External research also matches the repo's failure mode:

- **Scaling Law for Quantization-Aware Training** (Chen et al., 2025, <https://arxiv.org/abs/2505.14302>) argues that quantization error grows with training tokens and highlights late FC/FC2 outliers as a primary bottleneck.
- **Quantization Variation** (Huang et al., 2023/2024, <https://arxiv.org/abs/2307.00331>) frames transformer QAT instability in terms of outliers, module sensitivity, and dynamic oscillation.
- **QDrop** (Wei et al., 2022, <https://arxiv.org/abs/2203.05740>) motivates softer, less brittle quantization-aware objectives instead of abrupt all-or-nothing fake-quant branches.

I also reviewed selective cross-layer sharing literature (especially **ALBERT**, <https://arxiv.org/abs/1909.11942>) because it is an attractive artifact-budget idea for this challenge. I deferred it for this candidate because this codebase's export path currently quantizes every `state_dict` key independently, so naive parameter sharing would not actually buy artifact bytes without a larger alias-aware serialization change.

## Records and prior candidates that influenced this

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Activation port:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **QAT caveat:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Quantization-aware schedule motivation:** `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`
- **Prior candidates:** none existed in `candidates/` when this was created.

## What changed vs. the chosen base implementation

Starting from the `2026-03-22` 11-layer EMA + GPTQ-lite + warmdown3500 stack, this candidate makes four focused changes:

1. **LeakyReLU(0.5)^2 MLP**
   - Swaps `relu(x)^2` for `leaky_relu(x, 0.5)^2` in the MLP.

2. **Soft late quantization regularizer**
   - Removes reliance on a hard `CastedLinear._qat_enabled` branch.
   - Adds a late-stage **quantization-consistency loss** outside the compiled forward graph.
   - Ramps the regularizer from `0 -> 1` as the LR scale drops below `QREG_START_SCALE`.

3. **EMA-smoothed row scales**
   - The quantization target uses an EMA of per-row absmax scales instead of an abrupt per-step toggle, so the target is less noisy in late warmdown.

4. **Run-from-candidate quality-of-life**
   - Defaults for `DATA_PATH` and `TOKENIZER_PATH` resolve relative to the repository root, so the script runs correctly from this candidate directory.
   - Attention falls back to PyTorch SDPA if `flash_attn_interface` is unavailable.

The regularizer is deliberately narrow: it targets **attention output projections, MLP down projections, and the optional bigram / shared-value projection layers**, since those are cheap to enumerate and are the most plausible quantization-sensitive spots in this stack.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202604080706_ema-clip-softqat

SEED=1337 \
QREG_ENABLED=1 \
QREG_START_SCALE=0.30 \
QREG_WEIGHT=0.02 \
QREG_CLIP_DECAY=0.95 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script keeps the strong defaults from the `2026-03-22` base:

- 11 layers, 512 dim, 8 heads / 4 KV heads
- 3x MLP, partial RoPE, LN scale, XSA on the last 4 layers
- EMA + tight SWA, warmdown 3500, seq_len/eval_seq_len 2048
- BigramHash + shared value embeddings
- GPTQ-lite int6 export + sliding-window evaluation

Useful knobs if the regularizer is too weak or too strong:

- `QREG_START_SCALE` - earlier or later ramp entry
- `QREG_WEIGHT` - overall strength of the added penalty
- `QREG_CLIP_DECAY` - how fast the per-row clip EMA tracks new outliers

## Main risks / tradeoffs

- **Over-regularization:** if the ramp starts too early or `QREG_WEIGHT` is too high, the model may lose too much pre-quant fit before export.
- **Target selection risk:** the regularizer only hits projection/down matrices. The true best subset may be narrower (only MLP down) or broader.
- **Extra late-stage overhead:** the regularizer adds per-step parameter scans during the late warmdown. It is kept out of the compiled forward graph on purpose, but it still costs some wall-clock.
- **Untested runtime in this container:** I only ran syntax validation here because the container does not include the repo's core runtime dependencies.

## Validation

- `python -m compileall candidates/202604080706_ema-clip-softqat/train_gpt.py` - **passed**
- Minimal CPU smoke run - **not feasible in this container** because `torch`, `numpy`, and `sentencepiece` are not installed here, so a real startup test would require bootstrapping the full training environment first.
