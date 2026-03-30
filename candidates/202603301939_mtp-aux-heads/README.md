# MTP auxiliary heads on the Parallel-Muon + LeakyReLU^2 stack

## Hypothesis

A small **training-only multi-token prediction (MTP)** auxiliary loss should improve sample efficiency on this challenge without consuming artifact budget.

This candidate keeps the strongest current training/eval stack largely intact, but turns on two future-token heads during training and weights later horizons geometrically (`1.0, 0.5`). The key bet is that the extra supervision improves hidden-state quality enough to outweigh the modest extra logits work inside the 600-second wallclock cap.

## Why this is promising for this repository

The winning history in this repo strongly favors **high-leverage, low-byte changes**:

- `2026-03-19_SlidingWindowEval` showed that a zero-parameter eval change could deliver a massive win.
- `2026-03-20` through `2026-03-23` records kept stacking mostly cheap improvements: XSA on late layers, EMA, partial RoPE, LN scaling, GPTQ-lite clip search, LeakyReLU^2, and legal TTT.
- Several more invasive ideas lost on throughput grounds. `2026-03-18_FP16Embed_WD3600` explicitly reports that **SwiGLU** and **depth recurrence** hurt total progress in the 10-minute regime, and the 1x5090 non-record sweep also found **layer recurrence** net negative.

MTP fits this repo better than those ideas because the auxiliary heads are only used during training and are already excluded from export in the strongest recent scripts. That makes MTP an unusually clean way to trade a little more training FLOPs for potentially better final BPB while paying **zero model-byte cost**.

## Which prior records influenced this candidate

There were **no prior `candidates/` directories** in the repository when this candidate was created.

The main record influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current stack
  - showed that small MLP-side and eval-side changes are still moving the frontier
  - already contained dormant MTP plumbing, but did not enable it in the submitted recipe
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - same 11-layer family with GPTQ-lite, EMA, XSA, partial RoPE/LN scaling family features
  - also had dormant MTP hooks that were not used in the record README
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/`
  - evidence that this challenge rewards near-free quality improvements
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - cautionary evidence that ideas with too much per-step overhead can lose despite looking good per step

## External research that informed it

Primary sources considered:

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (2024), https://arxiv.org/abs/2404.19737
  - argues that predicting multiple future tokens from a shared trunk improves sample efficiency
  - highlights especially strong gains on generative and induction-heavy tasks
  - makes MTP attractive here because the challenge is wallclock-limited and sample-efficiency-limited
- Steven Esser et al., **"Learned Step Size Quantization"** (2020), https://arxiv.org/abs/1902.08153
  - considered as an alternative because quantization quality still matters a lot in this repo
  - not chosen for this candidate because it is more invasive and higher-risk with `torch.compile`
- Zhenzhong Lan et al., **"ALBERT"** (2019), https://arxiv.org/abs/1909.11942
  - considered for parameter sharing / recurrence-style savings
  - not chosen because repository evidence suggests recurrence-style ideas are more constrained by training-time budget than by model bytes

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS`: `0 -> 2`
2. **Add horizon decay for the auxiliary loss**
   - new env var: `MTP_LOSS_DECAY` (default `0.5`)
   - horizon losses are averaged with geometric weights, so the 2-step head does not overpower the main objective
3. **Keep MTP export-free**
   - the script still strips `mtp_heads.*` from the exported state dict before quantization and final artifact creation
4. **Add FlashAttention fallback**
   - if `flash_attn_interface` is unavailable, the candidate falls back to PyTorch SDPA so the module can still be imported and smoke-tested in an environment that has the normal Python ML dependencies but not FlashAttention

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603301939_mtp-aux-heads

RUN_ID=mtp_aux_heads \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 MTP_LOSS_DECAY=0.5 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a cheaper first pass focused on the training-side idea rather than the full evaluation stack, leave everything the same but set `TTT_ENABLED=0`.

## Main expected risks or tradeoffs

- **Training-time overhead:** even cheap auxiliary heads can reduce the number of optimizer steps reached in 600 seconds.
- **Horizon quality mismatch:** later-token targets are harder; if their gradients are too noisy, MTP can act like regularization instead of useful supervision.
- **Interaction with TTT:** if post-training TTT dominates the final improvement, the MTP gain may mainly show up in pre-TTT or post-EMA metrics rather than in the ultimate leaderboard number.
- **Need for sweep work:** the most likely useful follow-ups are `MTP_NUM_HEADS in {1,2,4}`, `MTP_LOSS_WEIGHT in {0.1,0.15,0.2}`, and possibly turning MTP on only after the early compile/warmup phase.

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603301939_mtp-aux-heads/train_gpt.py
```

Outcome:

- **Passed** for the root trainer scripts, `data/`, and this candidate `train_gpt.py`.

Attempted smoke test:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('candidates/202603301939_mtp-aux-heads/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_mtp_aux_heads', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# Instantiate a tiny CPU GPT and run one forward pass.
PY
```

Outcome:

- **Not feasible on this workflow runner** because the local Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
- The candidate now includes a FlashAttention fallback, so the same smoke import/forward path should work in a PyTorch-equipped environment without `flash_attn_interface`, provided the other normal dependencies (for example `sentencepiece`) are also installed.
- Full training still requires CUDA; the fallback only removes the hard dependency on `flash_attn_interface` for import/forward smoke paths.
