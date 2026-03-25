# Mirror-Shared 11L + Depth Adapters

## Hypothesis

The current winning family already squeezes a lot out of 11-layer, 512-dim U-Net GPTs with XSA, EMA, Partial RoPE, and compression-aware export. The next clean frontier is to **share the expensive block matrices across mirrored encoder/decoder depths**, then restore layer-specific flexibility with **cheap per-depth control tensors and small low-rank residual adapters**.

Concretely, this candidate keeps the strong `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` training/export stack, but replaces 11 unique transformer blocks with 6 shared block cores mapped as:

```text
[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
```

Each execution layer still has its own:

- `q_gain`
- `attn_scale`
- `mlp_scale`
- `resid_mix`
- rank-16 depth adapter
- XSA-on/off behavior
- LN scale factor
- VE scale (where enabled)

The hope is that this preserves the compute path and most of the positional specialization of the 11-layer stack while reducing artifact pressure and improving compression robustness.

## Why this is promising for this repository

Repository history shows two strong patterns:

- compression savings are usually converted into better architecture capacity (`MLP3x`, 10L/11L, XSA, bigram features), and
- once the 11L XSA/EMA family landed, later wins were mostly small additive improvements rather than new structural bets.

This candidate makes a different trade: instead of only squeezing export harder, it tries to reduce the largest repeated weights directly. Under the 16MB budget, that opens room for cheap depth-specific flexibility without paying for 11 separate heavy blocks.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Key ancestry:

- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` for Partial RoPE + LN scale
- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` for the 11L + XSA4 + EMA stack
- `2026-03-20_11L_EfficientPartialXSA_FA3_SWA120` for efficient XSA in deep layers
- `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` for the SmearGate + BigramHash + MLP3x family

Notably, I did **not** find a prior repo experiment built around recursive / mirrored layer tying or shared-depth transformer blocks.

## External research that informed it

- **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** (Bae et al., arXiv:2410.20672). Main takeaway: aggressive layer tying can recover much more quality if depth-specific low-rank flexibility is preserved.
- **Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation** (Bae et al., arXiv:2507.10524). Main takeaway: shared recursive stacks can sit on a better quality/efficiency frontier than vanilla untied transformers.
- **pQuant: Towards Effective Low-Bit Language Models via Decoupled Linear Quantization-Aware Training** (Zhang et al., arXiv:2602.22592). Main takeaway: a dominant compressed branch plus a compact high-flexibility branch can preserve sensitive behavior efficiently.
- **Attn-QAT: 4-Bit Attention With Quantization-Aware Training** (Zhang et al., arXiv:2603.00040). This was useful mostly as a negative constraint: very low-bit attention improvements seem to need custom kernels and deeper infrastructure than this repo currently carries.

## What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Mirrored shared-depth blocks**
   - 11 execution layers now reuse 6 shared block cores.
   - Mapping is symmetric across encoder/decoder depth.

2. **Per-layer control tensors stay unique**
   - `q_gain`, `attn_scale`, `mlp_scale`, and `resid_mix` are now layer-specific outside the shared cores.
   - This keeps cheap depth specialization even when the large matrices are tied.

3. **Rank-16 depth adapters**
   - Each execution layer adds a tiny low-rank residual adapter after the shared block.
   - This is the minimal “relaxed recursion” twist intended to recover some of the flexibility lost by tying.

4. **Dynamic per-layer XSA and LN scale**
   - XSA remains active only on the last `XSA_LAST_N` execution layers.
   - LN scale remains tied to execution depth, not shared-core index.

5. **Validation-oriented fallback path**
   - If FlashAttention is unavailable, the script falls back to PyTorch SDPA.
   - `SMOKE_TEST=1` runs a tiny synthetic forward/backward path without dataset or tokenizer dependencies.

Everything else intentionally stays close to the base stack: EMA, warmdown, GPTQ-lite export, BigramHash, VE, Partial RoPE, and the same optimizer split.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603250756_mirror-share-adapter
```

Full training/eval command (same strong base stack, with shared depth enabled by default):

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 SWA_ENABLED=1 SWA_EVERY=50 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_WD=0.04 ADAM_WD=0.04 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
LATE_QAT_THRESHOLD=0.15 SHARED_DEPTH=1 DEPTH_ADAPTER_RANK=16 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

EMA is always applied in this script, matching the chosen base implementation.

Lightweight local smoke path:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation commands and outcomes

Commands run in this workflow:

```bash
python -m compileall candidates/202603250756_mirror-share-adapter/train_gpt.py
SMOKE_TEST=1 python candidates/202603250756_mirror-share-adapter/train_gpt.py
```

Outcomes:

- `python -m compileall ...` **passed**.
- `SMOKE_TEST=1 ...` could not be completed in this workflow environment because **`torch` was not installed** in the runner Python environment. The script includes a smoke-test path, but executing it still requires the PyTorch runtime.

## Main expected risks / tradeoffs

- Sharing block matrices may remove too much depth-specific specialization even with adapters.
- The mirrored tying pattern may be too restrictive for the late XSA-heavy decoder layers.
- The saved artifact budget is not yet reinvested into a wider model or larger side modules, so this may under-use the 16MB cap if the tying is too aggressive.
- If this direction looks promising, the next experiments should test:
  - adapter ranks `{8, 16, 32}`,
  - larger BigramHash tables funded by the saved bytes,
  - sharing only MLP cores or only attention cores,
  - leaving the final 2-4 layers untied.
