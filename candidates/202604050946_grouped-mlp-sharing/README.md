# Grouped MLP Sharing

## Hypothesis

The current record lineage already exploits most of the obvious training, quantization, and evaluation gains. The next clean lever is to **reduce duplicated MLP weights without adding extra FLOPs**: share the heavy 3x-MLP blocks across adjacent logical layers, keep attention unique, and keep each layer's norms/scales/residual controls separate.

The bet is that this acts as a useful inductive bias, improves compressibility, and preserves most of the strong 11-layer recipe that already works in this repository.

## Why this is promising here

Repository evidence points in two directions:

1. **MLP capacity matters a lot.** The jump to 3x MLPs was one of the biggest recurring gains in the record history.
2. **Extra-compute recurrence did not work under the 10-minute cap.** Prior notes explicitly say looping layers was too expensive in fixed wall-clock training.

This candidate tries to thread that needle:

- keep the strong 11 logical layers,
- keep the same forward-pass depth and step-time order of magnitude,
- remove only redundant **MLP parameter copies**,
- leave attention, XSA, RoPE, EMA, GPTQ-lite, and the rest of the proven stack intact.

## Prior repository runs that informed this candidate

- **Chosen base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest clean train/export stack before TTT,
  - already includes the durable recipe: 11L, 3x MLP, seq2048, XSA4, partial RoPE, LN scale, EMA, VE, GPTQ-lite.
- **Activation influence:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the low-risk `LeakyReLU(0.5)^2` MLP activation change.
- **Negative-result constraint:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - both argue against adding extra recurrent passes under a fixed 10-minute budget, so this candidate does **sharing without extra depth**.

There were no prior runs under `candidates/` when this candidate was created.

## External research that informed it

- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** (arXiv:1909.11942)  
  Classic evidence that cross-layer parameter sharing can preserve quality while reducing memory and parameter count.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (arXiv:2505.01855)  
  Suggests selectively reusing transformer layers can be a useful inductive bias, especially earlier in the network.
- **Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization** (arXiv:2603.21676)  
  Reinforces the broader idea that depth and parameter count can be decoupled by shared-weight computation.
- **Parameter Reduction Improves Vision Transformers: A Comparative Study of Sharing and Width Reduction** (arXiv:2512.01059)  
  The most direct template for this candidate: adjacent-block GroupedMLP sharing improved stability and accuracy at the same compute budget.

The important repo-specific twist is that this candidate borrows the **same-compute sharing** idea, not the **more-steps recurrence** idea that prior local experiments already ruled out.

## What changed vs the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Adjacent grouped MLP sharing**
   - Added `GROUPED_MLP_SIZE` (default `2`).
   - MLP modules now live once at the GPT level and are indexed by logical block.
   - With the default setting, logical layers share MLP weights in adjacent pairs: `[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`, `[8,9]`, with the last layer left unique when needed.
   - Attention blocks remain unique.
   - Layer-specific RMSNorms, residual mixes, and learned scales remain unique.
   - The export path quantizes the shared `mlps.*` tensors only once, and the artifact records `GROUPED_MLP_SIZE`, `MLP_NEGATIVE_SLOPE`, and the derived block-to-MLP mapping so reloads stay self-describing.
2. **LeakyReLU(0.5)^2**
   - Swapped the base run's ReLU-squared MLP activation for the later record's `LeakyReLU(0.5)^2`.

Everything else is intentionally kept close to the proven base stack.

## How to run

From this directory:

```bash
RUN_ID=grouped_mlp_share \
GROUPED_MLP_SIZE=2 \
MLP_NEGATIVE_SLOPE=0.5 \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Executed from the repository root in this runner:

1. `python -m compileall train_gpt.py train_gpt_mlx.py data` — passed.
2. `python -m compileall candidates/202604050946_grouped-mlp-sharing/train_gpt.py` — passed.
3. A CPU-only structural smoke import was attempted, but this runner does not have `torch` installed, so runtime import/execution was not feasible here.

## Main risks and tradeoffs

- **Too much sharing may underfit.** The 3x MLPs are strong partly because they are large; tying adjacent layers may remove useful specialization.
- **Quantization distributions may shift.** Shared MLP weights plus LeakyReLU-squared activations may change the weight/value statistics that GPTQ-lite liked in the base run.
- **The win may be mostly artifact-side, not BPB-side.** This candidate is designed to preserve quality while improving parameter efficiency; whether that translates into lower validation BPB still needs a real GPU run.
- **Further reinvestment is still open.** If grouped sharing materially shrinks the artifact, the saved bytes could later be spent on larger hashed lexical memory, deeper VE coverage, or other small high-leverage additions.
