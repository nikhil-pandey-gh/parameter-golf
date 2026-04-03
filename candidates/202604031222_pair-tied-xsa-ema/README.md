# Pair-tied shared-depth XSA + EMA

## Hypothesis

The strongest pre-TTT line in this repo already converged on an 11-layer, 512d, XSA/EMA/GPTQ-lite stack, but it still spends most of its artifact budget on fully unique block weights. This candidate tests whether **pair-tied shared-depth cores** can preserve the winning 11-layer logical topology while shrinking the heavy attention/MLP parameter footprint enough to improve compression efficiency without giving up the repo's best architectural priors.

The key twist is that this is **not** the naive "one recurrent block repeated everywhere" idea that underperformed in the 1x5090 non-record sweep. Instead, it keeps **11 logical layers**, **6 shared block cores**, and **per-layer wrapper parameters** (RMSNorms, residual mixing, per-layer attn/mlp scales, skip weights, VE scales), which is closer to modern "relaxed recursive" sharing than to hard recurrence.

## Why this is promising here

Repository evidence says the frontier improved by stacking together:

- deeper 10L/11L models plus 3x MLPs,
- compression-aware training and export,
- XSA on late layers,
- EMA / better averaging,
- partial RoPE + LN scaling,
- and better post-training quantization.

Those gains are all retained here. What is still largely missing from the repo is **modern layer sharing with lightweight depth-specific flexibility**. That is attractive for Parameter Golf because the metric only cares that the artifact stays under 16MB, not that every logical layer has unique weights.

## Prior experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`  
  Established late-layer XSA + EMA as a clear win on the 11-layer stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`  
  Added partial RoPE and layerwise LN scaling with zero or near-zero parameter cost.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`  
  Chosen base implementation. Added GPTQ-lite clip search, EMA, and warmdown tuning on the mature 11-layer stack.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`  
  Shows the current frontier still sits on the same 11-layer family, but the latest gains came from activation tweaks and TTT rather than parameter sharing.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`  
  Important negative result: naive layer recurrence looked weak there, so this candidate uses pair-tied shared cores plus per-layer wrappers instead of full hard recurrence.

## External research that informed it

- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** (`arXiv:1909.11942`)  
  Classic evidence that cross-layer parameter sharing can cut model size substantially while retaining depth.
- **Universal Transformers** (`arXiv:1807.03819`)  
  Motivates reusing transformation blocks across depth while keeping recurrent-in-depth inductive bias.
- **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** (`arXiv:2410.20672`)  
  Most directly relevant paper: argues that hard layer tying is too rigid, and that small depth-specific adaptations recover much of the lost quality.
- **BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks** (`arXiv:2407.09527`)  
  Reinforces the repo trend that compact-model training should combine architectural changes with compression-aware training rather than treating quantization as an afterthought.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. Added **shared block cores** via `SHARED_LAYER_MAP` (for the default 11-layer setup it resolves to `0,0,1,1,2,2,3,4,4,5,5`), yielding **6 unique cores across 11 logical layers**.
2. Split the old `Block` into:
   - `BlockCore`: shared attention + MLP weights
   - `Block`: per-layer wrapper parameters and norms
3. Kept late-layer **XSA**, but validated that a shared core cannot mix XSA and non-XSA logical layers.
4. Added a **FlashAttention-optional fallback** to PyTorch SDPA so the candidate can still run when `flash_attn_interface` is absent.
5. Added a **`SMOKE_TEST=1` path** that constructs the model and runs one forward/logits pass without tokenizer or dataset access, which makes cheap CPU startup validation possible.
6. Made `numpy` / `sentencepiece` imports fail only when the full training path actually needs them, so smoke runs do not require the full repo runtime environment.
7. Stored the resolved shared-depth wiring in the exported artifact metadata so eval cannot silently rebuild the wrong logical-to-core mapping.

Everything else stays aligned with the base 11-layer stack: EMA, GPTQ-lite quantization, BigramHash, SmearGate, partial RoPE, LN scaling, VE layers, and the same overall Muon/Adam split.

## How to run

From the candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
SHARED_LAYER_MAP=0,0,1,1,2,2,3,4,4,5,5 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Cheap startup check:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation run in this workflow

Commands run:

```bash
python -m compileall candidates/202604031222_pair-tied-xsa-ema/train_gpt.py
python -m py_compile candidates/202604031222_pair-tied-xsa-ema/train_gpt.py
python -m venv /tmp/gh-aw/agent/pg-smoke-venv
. /tmp/gh-aw/agent/pg-smoke-venv/bin/activate
python -m pip install torch
SMOKE_TEST=1 python candidates/202604031222_pair-tied-xsa-ema/train_gpt.py
```

Outcomes:

- `compileall`: passed
- `py_compile`: passed
- CPU smoke test: passed
  - logged `shared_depth:logical_layers:11 unique_cores:6 layer_map:0,0,1,1,2,2,3,4,4,5,5`
  - logged `smoke_test:ok loss:6.9542 logits_shape:(1, 32, 1024)`

## Main risks and tradeoffs

- The 1x5090 sweep suggests **naive recurrence can hurt**, so the main risk is that even pair-tied sharing still throws away too much specialization.
- This candidate uses per-layer wrapper parameters, not full depth-wise LoRA as in the relaxed-recursive paper, so the flexibility may still be too weak.
- Sharing compresses the artifact but does **not** reduce FLOPs, so quality has to improve through better parameter allocation rather than faster training.
- If the idea works, the next follow-up is probably to spend the recovered bytes on a slightly stronger lexical/value path (for example bigger BigramHash or VE) rather than immediately adding more complexity.
