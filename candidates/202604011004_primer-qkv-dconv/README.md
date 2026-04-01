# Primer-style causal QKV depthwise conv on the LeakyReLU² + legal TTT stack

## Hypothesis

The repository has already validated the activation half of Primer-style efficient transformer design: `relu²` was present from the baseline and the current SOTA improved it further with LeakyReLU². The missing half is Primer's cheap local mixing trick: a causal depthwise convolution after the Q/K/V projections.

My hypothesis is that adding identity-initialized causal depthwise Q/K/V mixing to the current best stack will help this tiny model capture short-range structure more efficiently, letting attention spend more of its capacity on longer-range dependencies instead of relearning local token transitions.

## Why this is promising for this repository

Several strong repository trends point in the same direction:

- The biggest recent wins have come from compact, low-parameter architectural tweaks stacked onto a stable 11-layer compressed model: XSA, partial RoPE, LN scale, VE128, EMA, GPTQ-lite, and LeakyReLU².
- The leaderboard already rewards better local structure and better evaluation-time attention use: BigramHash, SmearGate, XSA, and legal score-first TTT all improved BPB.
- This change adds only a few thousand parameters, keeps the model fully transformer-compatible, and reuses the existing mixed int6 export path.
- The candidate is initialized to behave like the current best record at step 0, because each depthwise convolution starts as an exact causal identity filter.

## Prior repository work that influenced this candidate

Primary base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Most relevant prior influences:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - strongest overall stack in the repo
  - shows LeakyReLU² and legal score-first TTT working on top of parameter banking + XSA + partial RoPE + VE
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - confirms that better post-training quantization details still matter late in the stack's evolution
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - shows that small zero- or near-zero-parameter attention-side changes can still buy measurable gains
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md`
  - reinforces the importance of local inductive bias in this challenge

## External research that informed it

- **Primer: Searching for Efficient Transformers for Language Modeling** (So et al., 2021, arXiv:2109.08668)
  - Primer attributes most of its gains to two simple changes: squaring ReLU activations and adding depthwise convolution after Q/K/V projections.
  - This repo has already thoroughly explored the squared-activation half; this candidate tests the missing QKV-conv half on the current best stack.
- **Differential Transformer** (Ye et al., 2024, arXiv:2410.05258)
  - motivates the broader claim that ordinary attention often wastes capacity on irrelevant context.
  - A cheap local mixer in Q/K/V is a much lighter-weight way to reduce that burden than replacing attention entirely.
- **The Hidden Attention of Mamba Models / Mamba-2 Hybrid results** (Waleffe et al., 2024, arXiv:2406.07887)
  - supports the more general pattern that mixing lightweight sequence operators with attention can outperform pure-transformer baselines at fixed budgets.
  - This candidate keeps the change minimal by using only tiny depthwise convolutions rather than a full hybrid operator.

## What changed versus the chosen base implementation

Base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added `CausalDepthwiseConv1d`, a tiny depthwise 1D convolution module over the sequence axis.
2. Added three optional attention-side convolutions per block:
   - `q_conv` on the query stream
   - `k_conv` on the key stream
   - `v_conv` on the value stream
3. Applied the convolutions immediately after the linear Q/K/V projections and before reshaping into heads.
4. Initialized each convolution as an exact causal identity filter so the run starts from the proven base behavior.
5. Added two environment flags:
   - `QKV_CONV_ENABLED` (defaults to `1` in this candidate)
   - `QKV_CONV_KERNEL` (defaults to `3`)
6. Extended control-tensor handling so the new depthwise-conv weights stay in the same small-parameter optimization/export path as the other non-banked attention-side tensors.
7. Adjusted the candidate's default dataset/tokenizer paths to resolve relative to the repository root, so the script can be launched from inside this candidate directory as required.

Everything else intentionally stays aligned with the strong 2026-03-23 base:

- parameter banking + Parallel Muon
- 11-layer U-Net-like stack
- XSA on the last 4 layers
- partial RoPE and LN scale
- VE128 on layers 9-10
- LeakyReLU² MLP
- GPTQ-lite int6 export
- legal score-first TTT

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202604011004_primer-qkv-dconv
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
QKV_CONV_ENABLED=1 QKV_CONV_KERNEL=3 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To ablate back to the original base behavior without removing code:

```bash
cd candidates/202604011004_primer-qkv-dconv
QKV_CONV_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Commands run in this environment:

```bash
cd /home/runner/work/parameter-golf/parameter-golf
python -m compileall train_gpt.py candidates/202604011004_primer-qkv-dconv/train_gpt.py
python -m compileall candidates/202604011004_primer-qkv-dconv/train_gpt.py
```

Observed outcome:

- both compile checks passed without syntax errors

Why I did not run a runtime smoke test here:

- this environment does not have the candidate's runtime dependencies installed (`torch`, `flash_attn_interface`, `sentencepiece`, and related CUDA stack were unavailable to the local `python` interpreter), so even a CPU-only launch would fail at import time before reaching model startup
- because of that dependency gap, a truthful runtime smoke test was not feasible without introducing heavyweight new infrastructure

## Main expected risks and tradeoffs

- **Throughput risk**: even tiny sequence convolutions can interact badly with `torch.compile` or FlashAttention-heavy kernels, and a small step-time slowdown can erase any modeling gain in a strict 10-minute budget.
- **Short-horizon bias risk**: extra local mixing may overlap with what SmearGate and BigramHash already provide, reducing marginal benefit.
- **Quantization risk**: the new conv weights are small and should export cleanly, but they are still another tensor family entering the mixed int6 roundtrip.
- **Budget risk**: identity initialization reduces regression risk, but the 10-minute run may still be too short for the new filters to move far enough from identity to pay off.
