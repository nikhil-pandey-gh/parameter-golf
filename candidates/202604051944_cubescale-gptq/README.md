# CubeScale GPTQ-lite on the current SOTA stack

## Hypothesis

The current record stack is already strong on architecture and evaluation, but it still pays a noticeable export-time penalty when the trained weights are collapsed into the final int6 artifact. This candidate applies an **exact folded MLP equalization step** before GPTQ-lite quantization: for each hidden channel, rescale the LeakyReLU(0.5)^2 MLP up-projection and inversely rescale the down-projection so the full-precision function is unchanged, but the weight distribution presented to the quantizer is flatter.

Because this MLP is positively homogeneous of degree 2, the fold is exact:

```python
up'   = diag(s) @ up
down' = down @ diag(s^-2)
```

The scale is chosen with a simple cubic rule derived from balancing the up-row and down-column magnitudes:

```python
s_j = (down_col_stat_j / up_row_stat_j) ** (1/3)
```

This aims to reduce the final quantization error with:

1. zero training-time overhead,
2. no runtime metadata,
3. no change to the forward function before quantization.

## Why this is promising here

Repository evidence points to the export path as one of the last consistently productive surfaces:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed that a **better post-training clip search alone** still bought about `-0.0006 BPB` at zero training cost.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` kept the same low-bit export stack and reached the current best result, so the strongest next idea should stack onto that recipe rather than replace it.
- The model now uses **LeakyReLU(0.5)^2**, which makes exact hidden-channel folding unusually clean compared with a standard non-homogeneous activation.

MLP weights are also a large fraction of the artifact budget, so even a modest improvement in their quantization behavior can matter.

## Prior repo influences

### Chosen base implementation

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

This candidate keeps that stack as the base: parameter-banked 11-layer model, XSA on the last 4 layers, partial RoPE, LN scaling, shared value embeddings, EMA + tight SWA, GPTQ-lite int6 export, LeakyReLU(0.5)^2, and optional legal score-first TTT.

### Other influential records

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`  
  Reinforced that low-cost quantization improvements still move the metric.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`  
  Established partial RoPE + layerwise LN scaling as zero-parameter wins.
- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`  
  Established XSA + EMA as a durable backbone improvement.

### Prior candidates

There were **no existing `candidates/` experiments** in the repository at review time, so this is the first candidate folder.

## External research

This candidate is most directly informed by output-preserving quantization transforms:

| Source | Relevance |
|---|---|
| SmoothQuant — <https://arxiv.org/abs/2211.10438> | Equivalent offline scaling can migrate quantization difficulty without changing the full-precision function. |
| AWQ — <https://arxiv.org/abs/2306.00978> | Salient-channel protection via equivalent scaling is effective for weight-only PTQ. |
| SpinQuant — <https://arxiv.org/abs/2405.16406> | Output-preserving transforms that flatten outliers can materially improve low-bit quantization. |

The twist here is repository-specific: instead of activation-calibrated scaling or learned rotations, this candidate uses an **exact cubic fold tailored to the LeakyReLU^2 MLP already used by the best record**.

## What changed vs. the chosen base

1. Added `cubescale_leakyrelu2_mlp(...)`, which rewrites each unbanked MLP pair before export and then quantizes the transformed weights.
2. Added `EXPORT_MLP_EQUALIZE` and `EXPORT_MLP_EQUALIZE_MAX_SCALE` knobs.
3. Fixed default dataset/tokenizer paths so the script works when launched from inside this candidate directory.
4. Added a CPU-safe attention fallback plus `SMOKE_TEST=1` so the script has a cheap local startup/export-path check.

No repository files outside this candidate directory were modified.

## How to run

### Full training/eval run

From this directory:

```bash
cd candidates/202604051944_cubescale-gptq
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
EXPORT_MLP_EQUALIZE=1 EXPORT_MLP_EQUALIZE_MAX_SCALE=2.5 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Cheap smoke run

```bash
cd candidates/202604051944_cubescale-gptq
SMOKE_TEST=1 python train_gpt.py
```

## Validation run in this workflow

Commands run:

```bash
python -m compileall candidates/202604051944_cubescale-gptq/train_gpt.py
SMOKE_TEST=1 python candidates/202604051944_cubescale-gptq/train_gpt.py
```

Outcome:

```text
smoke_test:ok base_loss:4.849130 equalized_loss:4.849130 quant_loss:4.849129 logit_delta:1.192e-07 loss_delta:0.000e+00 equalized_layers:3
```

The smoke path verifies that:

1. the script imports and runs on CPU,
2. the equalized float model matches the original float model on a synthetic test to strict tolerance,
3. the real `torch.save` -> `lzma` -> file -> `lzma.decompress` -> `torch.load` roundtrip still works,
4. the dequantized export path still produces a finite loss.

## Risks and tradeoffs

- The scaling heuristic uses **weight statistics only**, not activation calibration, so it may miss the truly salient channels.
- CubeScale currently targets **MLP pairs only**; attention/export error may still dominate after the MLP is improved.
- Extra code bytes are small but non-zero, so the gain must come from better quantization rather than code-size reduction.
- This workflow only ran lightweight local validation; the real question is whether the transformed int6 artifact improves BPB on a full H100 run.
