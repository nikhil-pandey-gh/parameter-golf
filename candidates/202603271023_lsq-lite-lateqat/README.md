# LSQ-lite Late QAT on the 11L GPTQ-lite EMA stack

## Hypothesis

A real late-stage weight-only LSQ-style QAT path should outperform the repo's earlier fixed-scale late-QAT attempt because it lets each compression-critical weight row learn its own export step size during the final learning-rate decay phase.

This repository's frontier is already strongly compression-bound: the best static stacks cluster near 15.5-16.0MB and continue improving mostly through export-aware training, better quantization, and evaluation protocol rather than raw extra training alone. If the model can co-adapt to its eventual int6 grid during warmdown, the post-quantization gap should shrink without requiring new infrastructure.

## Why this is promising for this repository

Three repository facts point in the same direction:

- The strongest clean pre-TTT base is the 2026-03-22 `11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15` stack at `1.1233` mean / `1.1228` best.
- The 2026-03-21 partial-RoPE record explicitly notes that its late-QAT switch was accidentally a no-op under `torch.compile`, so there is still unfinished business in this area.
- Multiple records show that quantization-aware choices, mixed precision, and compression-friendly training are where the leaderboard has kept moving.

This candidate therefore keeps the proven 11-layer architecture and targets the remaining quantization gap more directly.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Key influences:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Important because its README documents the previous `torch.compile` late-QAT failure mode.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Contributed the LeakyReLU(0.5)^2 activation change, which is now carried into this candidate's static stack.
- `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`
  - Reinforced that int6-aware training is useful when it actually touches the weights that matter.

There were no prior `candidates/` directories when this candidate was created.

## External research that informed it

Primary research direction:

- PACT, Choi et al. (2018), arXiv:1805.06085
  - Learnable clipping is an effective way to align training with low-precision deployment.
- LSQ, Esser et al. (2019), arXiv:1902.08153
  - Learned step sizes let the quantizer itself be optimized rather than fixed heuristically.
- LSQ+, Bhalgat et al. (2020), arXiv:2004.09576
  - Refines learned-step quantization and supports the general learned-scale direction.

Repository-fit supporting evidence:

- Chen et al., "Scaling Law for Quantization-Aware Training" (2025), arXiv:2505.14302
  - Highlights that quantization bottlenecks increasingly matter with more training and that mixed-precision handling of sensitive regions is important.
- Xiao et al., "Exploring Layer-wise Information Effectiveness for Post-Training Quantization in Small Language Models" (2025), arXiv:2508.03332
  - Strong recent evidence that small language models benefit from sensitivity-aware quantization choices.

## What changed versus the chosen base implementation

Starting from the 2026-03-22 record script, this candidate makes four focused changes:

1. **Real LSQ-lite late QAT for `CastedLinear` weights**
   - Each linear layer now owns a trainable per-row log-scale parameter.
   - When late QAT turns on, those scales are initialized from the current weight distribution and then optimized directly.
   - The fake-quantized forward path uses an LSQ-style straight-through estimator instead of a fixed row-max rule.

2. **Avoid the previous `torch.compile` late-QAT failure mode**
   - The script starts compiled for speed.
   - When late QAT activates, it switches training to eager mode and enables LSQ on the real modules so the fake-quant branch cannot be constant-folded away.

3. **Export uses learned scales as an additional int6 candidate**
   - GPTQ-lite percentile search is preserved.
   - The learned LSQ row scales are supplied as an extra export candidate so the trained quantizer can influence the final serialized artifact.

4. **Carry forward LeakyReLU(0.5)^2**
   - The MLP now uses the same `LeakyReLU(0.5)^2` activation that helped the 2026-03-23 record.

## How to run or evaluate it

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_WD=0.04 ADAM_WD=0.04 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.15 LSQ_INIT_PERCENTILE=0.9995 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional CPU-safe smoke path:

```bash
SMOKE_TEST=1 python train_gpt.py
```

The smoke path instantiates a tiny local model, runs one backward pass, quantizes/dequantizes a roundtrip checkpoint, reloads it, and runs one evaluation forward pass. It is meant only as a startup sanity check, not a score estimate.

## Validation

Commands attempted for this candidate:

```bash
python -m compileall train_gpt.py
python -m compileall ../../train_gpt.py ../../train_gpt_mlx.py ../../data
SMOKE_TEST=1 python train_gpt.py
```

Outcomes in this workflow environment:

- `python -m compileall` on the candidate script: **passed**.
- `python -m compileall` on the root scripts plus `data/`: **passed**.
- `SMOKE_TEST=1 python train_gpt.py`: **not runnable here** because this workflow image does not currently provide an importable `torch` installation. The candidate script includes the smoke path and a CPU-safe attention fallback, but this environment cannot execute it without the repository's normal runtime dependencies.

## Main expected risks and tradeoffs

- **Late-QAT speed tradeoff**: switching from compiled to eager mode during the final phase may reduce the number of last-phase steps.
- **Scale-optimizer sensitivity**: learned quantizer scales may want different optimizer or decay settings than ordinary scalar parameters.
- **Activation carry-over uncertainty**: LeakyReLU(0.5)^2 helped the best TTT stack, but the exact gain on this non-TTT base still needs measurement.
- **Export mismatch risk**: GPTQ-lite plus learned scales is intentionally conservative, but the best export scale may still differ from the best training-time fake-quant scale.

## Suggested next experiments

- Compare this against the unmodified 2026-03-22 base with matched seeds and runtime.
- Ablate `LSQ_INIT_PERCENTILE` and whether learned scales should use zero weight decay or a lower LR.
- If the quant gap improves cleanly, try a slightly larger lexical memory feature such as a modestly larger BigramHash table or a trigram hash add-on.
