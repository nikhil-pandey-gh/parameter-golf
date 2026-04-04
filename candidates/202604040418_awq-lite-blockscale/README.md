# Candidate: AWQ-lite Shared BlockScale on the 2026-03-23 Stack

## Hypothesis

The current best stack already captures most of the obvious architecture and evaluation wins, so the next cheap gain is probably in the **int6 export path** rather than the core model. A small activation-aware, mathematically exact rescaling step before GPTQ-lite quantization should reduce the post-training int6 error on the largest banked weights without changing training dynamics or adding much artifact overhead.

This candidate adds a **shared per-block input scale** that is estimated from a few **already-seen late training batches during the live training loop**, then folded into the Q/K/V and MLP-up weights before the existing per-row int6 quantization pass. At inference time, the corresponding normalized block inputs are divided by that scale, so the float model is unchanged while the quantizer sees a smoother weight distribution.

## Why this is promising here

The record progression in this repository suggests that most recent wins come from either:

1. Better low-bit export (`int6`, GPTQ-lite, late-QAT attempts), or
2. Better evaluation protocol (sliding windows, then legal TTT).

The 2026-03-23 record already has strong training/eval machinery:

- 11-layer banked model
- XSA on the deepest layers
- Partial RoPE + LN scaling
- EMA + tight SWA
- GPTQ-lite int6 export
- legal score-first TTT
- LeakyReLU(0.5)^2

That leaves the remaining int6 roundtrip loss as one of the cleanest places to attack next. Early repository QAT records also showed that quantization robustness can matter materially, but the newer banked stack no longer gets much help from the old CastedLinear-only fake quantization path.

## Repository evidence that influenced this candidate

- **`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`**: best current stack and direct base for this candidate.
- **`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`**: showed that export-side quantization quality still has measurable headroom even on the modern 11-layer stack.
- **`records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`**: strong evidence that reducing the quantization gap can translate into leaderboard gains.
- **`records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`**: helpful negative-result log showing that high-risk architectural changes like recurrence can backfire under a fixed wall-clock budget.

There were **no prior `candidates/` directories** in the repository at review time, so this idea is differentiated against records only.

## External research that informed it

- **SmoothQuant** ([arXiv:2211.10438](https://arxiv.org/abs/2211.10438)): exact offline rescaling can migrate quantization difficulty away from troublesome channels.
- **AWQ** ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)): activation-aware channel protection is especially effective for weight-only low-bit quantization.
- **SmoothQuant+** ([arXiv:2312.03788](https://arxiv.org/abs/2312.03788)): reinforces that channel smoothing remains useful even for aggressive weight-only PTQ.
- **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) and **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)): newer work points to the same root cause — outlier management is often the bottleneck for low-bit PTQ — but with larger rotation machinery than this repo likely wants.

This candidate intentionally implements the **smallest exact transform** that fits the repository: one fp16 scale vector per transformer block, not full learned rotations or a separate calibration framework.

## What changed vs the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes:

1. Added candidate-local default path resolution so the script can be run from inside this candidate directory.
2. Added a persistent `block_input_scale` buffer per transformer block.
3. During quantization export:
   - use activation maxima collected from a few **live late-training batches** using hooks on the attention and MLP inputs,
   - aggregate those stats into one shared block scale,
   - fold that scale into `c_q`, `c_k`, `c_v`, and `mlp.fc` weight columns,
   - store the reciprocal behavior in the model via `block_input_scale`.
4. Left the rest of the stack unchanged: LeakyReLU^2, parameter banks, GPTQ-lite int6 export, sliding-window eval, and legal TTT all remain intact.

## How to run

From the repository root:

```bash
cd candidates/202604040418_awq-lite-blockscale
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
AWQ_LITE_ENABLED=1 AWQ_LITE_ALPHA=0.5 AWQ_LITE_CALIB_STEPS=8 \
AWQ_LITE_COLLECT_THRESHOLD=0.25 \
AWQ_LITE_CALIB_TOKENS=262144 AWQ_LITE_MAX_SCALE=4.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate now resolves `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root by default, so those environment variables are optional when launched from this directory.

## How to evaluate

The script preserves the base workflow:

1. train to the wall-clock cap,
2. apply EMA,
3. export with the activation-aware block scales already fit during late training,
4. export GPTQ-lite int6 + lzma,
5. evaluate standard, sliding-window, and optional legal TTT metrics.

No train shards are reopened during export or evaluation.

Look for the new log line:

```text
awq_lite:blockscale calib_tokens:... alpha:... scale_range:[..., ...] scale_mean:...
```

## Main risks and tradeoffs

- A single shared block scale is cheaper than per-matrix or rotation-based schemes, but it is also less expressive.
- The extra fp16 scale vectors slightly increase artifact size.
- The calibration heuristic uses activation maxima from a few live late-training batches; if those batches are not representative, the rescaling may do nothing or even hurt.
- Because this is still PTQ-side rather than true bank-weight QAT, it may leave some quantization error on the table.

## Validation

Commands run in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202604040418_awq-lite-blockscale/train_gpt.py
```

Outcomes:

- Both compile checks passed.
- A CPU-only runtime smoke test was **not feasible** here because this script is CUDA-only at runtime (`torch.cuda.is_available()` is required) and depends on `flash_attn_interface`.
