# Mirror-Shared Width-576

## Hypothesis

The repo's best public line is already saturated with deeper 11-layer stacks, XSA, EMA, GPTQ-lite, and TTT. The strongest open lever is to **share heavy transformer cores across depth without increasing effective depth**, then reinvest the saved bytes into a wider hidden state and a slightly larger bigram table.

This candidate uses a **6-core mirrored schedule across 11 effective layers**:

```text
0,1,2,3,4,5,4,3,2,1,0
```

Each effective layer keeps its own lightweight adapter (norms, residual mixing, layer scales, XSA flag, skip topology, value-embedding scale), while the attention+MLP weights are shared. The expectation is that this preserves late-layer specialization better than naive recurrence, avoids extra forward passes, and shifts the artifact budget toward width where this repo's strongest stacks already seem bottlenecked.

## Why this is promising here

Three repo signals point at this direction:

1. **Artifact pressure is central.** The leaderboard steadily improved by converting bytes into depth, MLP width, and better quantization/export behavior.
2. **Naive recurrence was a dead end when it increased compute.** Prior notes explicitly say looping layers hurt because it cut step count under the 10-minute wallclock.
3. **The strongest recent stack is mature enough that a new architecture-level byte reallocation is more interesting than another small quantizer tweak.**

So this candidate intentionally avoids "run the same block more times" and instead does **parameter sharing at fixed effective depth**.

## Prior repo work that influenced this candidate

- **`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`**  
  Chosen as the main base: 11L, XSA4, partial RoPE, LN scale, shared value embeddings, EMA, GPTQ-lite export, warmdown 3500.
- **`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`**  
  Supplied the strongest recent one-line model-side gain: **LeakyReLU(0.5)^2** in the MLP.
- **`records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`** and  
  **`records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`**  
  Both explicitly warn that **depth recurrence / looping layers** was counterproductive under wallclock constraints. This candidate is designed to avoid that failure mode.

There were **no prior `candidates/` runs** in the repository when this candidate was created.

## External research

- **ALBERT** — cross-layer parameter sharing as a direct way to reduce model size and memory while keeping depth.  
  https://arxiv.org/abs/1909.11942
- **Universal Transformers** — depth recurrence as an architectural primitive, with iterative refinement across depth.  
  https://arxiv.org/abs/1807.03819
- **Subformer** — generative transformers benefited from **sandwich-style parameter sharing** rather than naive full sharing.  
  https://arxiv.org/abs/2101.00234
- **Basis Sharing** — newer evidence that cross-layer sharing can improve compression efficiency at high compression ratios.  
  https://arxiv.org/abs/2410.03765
- **Intra-Layer Recurrence in Transformers for Language Modeling** — reports that recurrence pressure is best allocated toward **earlier layers**, supporting selective/shared-depth ideas rather than uniform looping everywhere.  
  https://arxiv.org/abs/2505.01855

## What changed vs the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. Replaces the flat 11-block stack with:
   - **shared heavy cores** (`shared_blocks`)
   - **per-layer adapters** (`layer_adapters`)
   - a **mirrored depth schedule** controlled by `NUM_SHARED_BLOCKS` and `SHARE_MODE`
2. Keeps **XSA, LN scale, skip connections, partial RoPE, shared value embeddings, EMA, GPTQ-lite, late QAT, and mixed int6 export** from the March 22 base.
3. Switches the MLP activation from ReLU^2 to **LeakyReLU(0.5)^2**.
4. Widens the default model from **512 -> 576** and increases the default bigram table from **2048 -> 3072** using the bytes saved by sharing.
5. Adds a **`SMOKE_TEST=1` path** that runs a tiny synthetic forward/backward plus quantize/dequantize roundtrip without touching the dataset.
6. Adds a **FlashAttention fallback** to standard PyTorch SDPA so the script can still import and execute the smoke path on machines without `flash_attn_interface`.

## How to run

From this directory:

```bash
RUN_ID=mirror_share_width576 \
NUM_LAYERS=11 \
NUM_SHARED_BLOCKS=6 \
SHARE_MODE=mirror \
MODEL_DIM=576 \
BIGRAM_VOCAB_SIZE=3072 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
VE_ENABLED=1 \
VE_LAYERS=8,9,10 \
ACTIVATION_SLOPE=0.5 \
TIED_EMBED_LR=0.035 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 \
TRAIN_BATCH_TOKENS=655360 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`SHARE_MODE=mirror` currently expects `NUM_SHARED_BLOCKS=ceil(NUM_LAYERS / 2)`, and the script defaults `NUM_SHARED_BLOCKS` that way. For 11 layers the intended mirrored run is therefore `NUM_SHARED_BLOCKS=6`.

For a local synthetic smoke path:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## How to evaluate

The script keeps the March 22 export/eval flow:

- EMA weights are applied before export
- weights are mixed-quantized to int6/int8
- the round-tripped model is reloaded
- final tokenizer-agnostic BPB is reported
- stride-64 sliding window evaluation is run when `EVAL_STRIDE=64`

Look for:

- `final_int6_roundtrip_exact`
- `final_int6_sliding_window_exact`

## Main risks / tradeoffs

- **Late-layer interference:** shared cores may still couple early and late representations too tightly even with per-layer adapters.
- **Step-time risk:** width 576 may cost enough throughput that the recovered steps offset some quality gains.
- **Quantization sensitivity:** repeated shared weights make some export errors effectively "show up" in multiple effective layers.
- **Schedule choice is underexplored:** mirrored sharing is a strong first guess, but a less aggressive or encoder-biased schedule may work better.

## Validation

Commands run from the repo root:

```bash
python -m compileall candidates/202604072154_mirror-share-width576/train_gpt.py
SMOKE_TEST=1 python candidates/202604072154_mirror-share-width576/train_gpt.py
```

Outcomes:

- `python -m compileall ...` **passed**
- `SMOKE_TEST=1 ...` **could not be completed in this runner** because the environment does not have `torch` installed, so the synthetic forward/backward smoke path was still not executable without extra setup
