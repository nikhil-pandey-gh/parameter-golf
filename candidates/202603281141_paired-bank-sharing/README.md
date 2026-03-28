# Paired Bank Sharing

## Hypothesis

Pairwise sharing the large transformer weight banks across adjacent layers can buy back a meaningful amount of artifact budget without giving up the effective 11-layer computation path, because this codebase already keeps many layer-local adaptation knobs outside the banks: untied RMSNorm pathways, per-layer residual mixing, per-layer attention/MLP scales, per-layer `q_gain`, optional value embeddings, and U-Net skip weights.

The bet is that this behaves like a lightweight “relaxed recursive transformer”: share the expensive matrices, keep the cheap per-layer controls untied, and spend some of the recovered bytes on a larger lexical side channel (`BIGRAM_VOCAB_SIZE=3072` by default).

## Why this is promising here

The repository history shows that the leaderboard frontier is already heavily optimized along evaluation, quantization, and schedule axes:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the current best overall stack (`val_bpb: 1.1194`) and already combines LeakyReLU^2, legal score-first TTT, Parameter Banking, Parallel Muon, XSA, Partial RoPE, EMA/SWA-style averaging, and GPTQ-lite int6 export.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the strongest simpler training-only stack.
- Earlier records repeatedly show that increasing capacity becomes worthwhile once compression improves enough to stay under the 16 MB cap.

What is notably underexplored in this repo is **cross-layer parameter sharing**. The current winner already stores its largest tensors as contiguous banks, which makes pairwise sharing a surgical change instead of a new infrastructure project.

## Prior records and candidates that influenced this

There were **no prior `candidates/` directories** in the repository when this candidate was created.

The main repository influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - chosen as the direct code base because it is the current best overall stack and already uses parameter banks.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - confirms the underlying 11-layer XSA/Partial-RoPE/EMA/GPTQ-lite stack is strong even without TTT.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - suggests bigger bigram/hash capacity can help when the artifact budget allows it.

## External research that informed it

- **ALBERT** (Lan et al., 2019, `arXiv:1909.11942`): cross-layer parameter sharing can cut parameter cost substantially while preserving strong language performance.
- **WideNet** (Xue et al., 2021, `arXiv:2107.11817`): if you share depth, keeping normalization or other light layer-specific transforms untied helps recover modeling capacity.
- **Relaxed Recursive Transformers** (Bae et al., 2025, `arXiv:2410.20672`): layer tying works better when relaxed by depth-specific adaptation rather than forcing every layer to be identical.
- **ShishuLM** (Kumar and Srinivasan, 2025, `arXiv:2510.13860`): paired weight sharing can reduce memory/latency in small language models while preserving most of the useful computation path.

This candidate intentionally mirrors that pattern: the heavy bank tensors are shared in pairs, while the cheap per-layer control parameters stay unique.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

- Added `BANK_GROUP_SIZE` (default `2`) so the four large parameter banks are stored for `ceil(num_layers / BANK_GROUP_SIZE)` unique bank layers instead of one bank slice per logical layer.
- Kept the logical model depth unchanged at 11 layers; only the stored bank weights are shared.
- Preserved all per-layer non-bank parameters, so each logical layer still has its own norms, residual mix, attention/MLP scales, gates, `q_gain`, skip structure, and optional VE scales.
- Increased the default `BIGRAM_VOCAB_SIZE` from `2048` to `3072`, using some of the recovered storage budget for a larger lexical side channel.
- Updated the bank export / reload path so GPTQ-lite int6 quantization preserves the shared-bank layout.
- Added a FlashAttention fallback to `scaled_dot_product_attention` so the script is less brittle outside the exact H100 environment.
- Added `SMOKE_TEST=1` for a tiny CPU forward-pass sanity path when dependencies are available.
- Adjusted default dataset/tokenizer paths so the script can be launched **from this candidate directory** without overriding `DATA_PATH` / `TOKENIZER_PATH`.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603281141_paired-bank-sharing

BANK_GROUP_SIZE=2 \
BIGRAM_VOCAB_SIZE=3072 \
NUM_LAYERS=11 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

A dependency-equipped smoke command is:

```bash
cd candidates/202603281141_paired-bank-sharing
SMOKE_TEST=1 python3 train_gpt.py
```

## Validation

Validated in this workflow runner:

```bash
python3 -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603281141_paired-bank-sharing/train_gpt.py
```

Outcome: **passed**.

Attempted smoke validation in this runner:

```bash
SMOKE_TEST=1 python3 candidates/202603281141_paired-bank-sharing/train_gpt.py
```

Outcome: **not feasible here** because the runner's `python3` environment does not have the repository runtime dependencies installed (`torch` is missing). The candidate script now exposes a dependency-light smoke path, but it still needs PyTorch to construct the model.

## Main expected risks and tradeoffs

- Pairwise bank sharing may reduce genuine layer diversity enough to hurt BPB, even with the untied per-layer control tensors.
- Legal TTT updates shared banks, so one adaptation step now affects two logical layers at once; this could help regularize, or it could over-couple the adaptation dynamics.
- The larger bigram table is intentionally modest, but it is still a secondary guess layered on top of the main sharing idea.
- Quantization on shared banks may behave differently from per-layer banks: fewer unique matrices can improve compression, but each shared matrix now has to serve two logical depths.

## Suggested next experiments

- Sweep `BANK_GROUP_SIZE` across `1`, `2`, and `3`.
- Keep `BANK_GROUP_SIZE=2` but ablate `BIGRAM_VOCAB_SIZE` back to `2048` to isolate the effect of sharing itself.
- If pair sharing is promising, add a tiny depth-specific low-rank correction on only the output projections, following the spirit of relaxed recursive transformers without paying a large byte cost.
