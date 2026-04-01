# Hadamard GPTQ-lite on the 11L EMA stack

## Hypothesis

Fixed orthogonal rotations can smooth out weight outliers before low-bit export, so applying a block-Hadamard transform to large int6 MLP and attention matrices **before** GPTQ-lite percentile search should reduce round-trip quantization error at essentially zero training-time cost. On this repo's best non-TTT stack, that should be a cleaner frontier than adding more QAT or longer-context complexity.

## Why this is promising here

- The repository's strongest recent gains are still heavily compression-aware: mixed int5/int6/int8 export, EMA/SWA, GPTQ-lite clip search, and weight-decay-tuned low-bit robustness kept beating plain training-only changes.
- The 2026-03-22 record showed GPTQ-lite plus EMA still had room to move without touching the training graph.
- External quantization work suggests orthogonal rotations reduce outlier severity and improve PTQ quality: QuaRot introduced fixed random/Hadamard-style rotations for end-to-end low-bit quantization, SpinQuant showed that better-chosen rotations materially improve quantized accuracy, and KurTail pushed the same idea with cheaper layerwise optimization.

## Prior repository work that influenced this candidate

- **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best non-TTT stack,
  - already centered on EMA + GPTQ-lite + warmdown tuning.
- **Activation port:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - LeakyReLU(0.5)^2 was the clearest cheap quality gain in the current best record.
- **Quantization context:** `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/` and related int6/int5 records
  - confirmed that export format and precision allocation are still first-order levers.

There were **no prior `candidates/` directories** in the repository at review time.

## External research

- **QuaRot**: fixed rotations can remove outliers and make low-bit quantization easier without changing full-precision model outputs.  
  <https://arxiv.org/abs/2404.00456>
- **SpinQuant**: rotations matter a lot, and better rotations significantly improve quantized accuracy.  
  <https://arxiv.org/abs/2405.16406>
- **KurTail**: newer PTQ work still improves on rotation-based outlier handling, which suggests this direction remains active rather than saturated.  
  <https://arxiv.org/abs/2503.01483>

## What changed vs the chosen base implementation

1. **LeakyReLU(0.5)^2 MLP**
   - Ported the activation change from the current best record onto the 2026-03-22 non-TTT stack.

2. **Hadamard-rotated int6 export**
   - Added a fixed block-Hadamard transform on the last dimension of large 2D int6-targeted tensors.
   - The export path now:
     1. rotates MLP/attention weights into a Hadamard basis,
     2. runs the existing GPTQ-lite percentile clip search in that basis,
     3. stores the rotated quantized tensors plus tiny metadata,
     4. inverse-rotates after dequantization during eval/load.
   - Default knobs:
     - `HADAMARD_INT6=1`
     - `HADAMARD_INT6_CATEGORIES=mlp,attn`
     - `HADAMARD_BLOCK_SIZE=512`

3. **Attention fallback for smoke/imports**
   - If FlashAttention 3 is unavailable, the script falls back to PyTorch SDPA for all attention calls.
   - That keeps local imports and smoke tests possible, but benchmark-like runs should still use FlashAttention 3 because step time and exact numerics can differ.

## How to run

From the candidate directory:

```bash
cd candidates/202604012056_hadamard-gptq-lite
RUN_ID=hadamard_gptq_lite \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
HADAMARD_INT6=1 HADAMARD_INT6_CATEGORIES=mlp,attn HADAMARD_BLOCK_SIZE=512 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This command assumes the usual Parameter Golf CUDA environment. If `flash_attn_interface` is missing, the script will still run via PyTorch SDPA, but that is mainly for local portability rather than leaderboard-style timing comparisons.

## How to evaluate the idea

- Compare against the 2026-03-22 stack with the same training settings and no Hadamard rotation.
- Inspect:
  - pre-quant `DIAGNOSTIC post_ema val_bpb`,
  - final round-trip `final_int6_roundtrip_exact`,
  - sliding-window `final_int6_sliding_window_s64_exact`,
  - final artifact size.
- The main question is whether the rotated basis improves the **post-quant / round-trip score** enough to justify any compression side effects.

## Main risks and tradeoffs

- Fixed Hadamard rotations are much simpler than learned SpinQuant-style rotations, so gains may be smaller or inconsistent.
- Smoother quantization error can still hurt **zstd compressibility** if the rotated int6 tensors become less structured.
- The best rotation target set is unclear; `mlp,attn` is a reasonable first pass, but `mlp`-only may end up stronger.
- LeakyReLU(0.5)^2 was beneficial on the TTT/parallel-Muon stack, but the exact gain may differ on this earlier base.

## Validation run in this environment

| Command | Outcome |
|---|---|
| `python -m compileall train_gpt.py train_gpt_mlx.py data` | Passed before candidate changes |
| `python -m compileall candidates/202604012056_hadamard-gptq-lite/train_gpt.py` | Passed |
| CPU import/forward smoke | Not completed here because the environment does not have `torch` installed |
