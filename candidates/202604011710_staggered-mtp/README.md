# Staggered Curriculum MTP on the LeakyReLU² + Legal TTT Stack

## Hypothesis

A forward-curriculum multi-token prediction (MTP) objective can improve sample efficiency on the strongest current stack without increasing the exported artifact size, because the extra MTP heads are used only during training and are excluded from the final saved model.

The key bet is that this repository is already in the "small, additive gains matter" regime: once sliding eval, mixed quantization, EMA/SWA, partial RoPE, XSA, LeakyReLU², and legal TTT are in place, one more low-infrastructure training-only objective may still buy a small BPB improvement.

## Why this is promising for this repository

- The strongest archived run is `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`, which already stacks many artifact-aware wins but still leaves dormant MTP code paths disabled in the logs (`mtp_num_heads:0`).
- The March 21-23 11-layer stack already carries explicit MTP heads in code, but every archived run keeps them off, so this is a real unexplored branch rather than a re-run of an existing record.
- Unlike another quantization tweak, MTP is training-only: if it helps optimization, it does so without spending extra bytes in the final artifact.
- The 2025 curriculum-MTP work is especially relevant here because it specifically calls out that smaller language models benefit from staged MTP rather than turning on all future-token heads at once.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Chosen base implementation.
  - Contributes the best current stack: LeakyReLU(0.5)^2, parameter banking + parallel Muon, EMA/SWA, GPTQ-lite int6 + lzma, and legal score-first TTT.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Earlier version of the same 11-layer family with dormant MTP plumbing and a cleaner non-TTT reference point.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Confirms the same family already depended on partial RoPE + LN scaling, while the archived logs still show `mtp_num_heads:0`.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/`
  - Reminder that this repo rewards low-byte, evaluation- or training-only gains when they integrate cleanly with the existing stack.

## External research that informed this candidate

- Fabian Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (2024): https://arxiv.org/abs/2404.19737
  - Motivates MTP as a sample-efficiency improvement on a shared trunk.
- Ansar Aynetdinov and Alan Akbik, *Pre-Training Curriculum for Multi-Token Prediction in Language Models* (2025): https://arxiv.org/abs/2505.22757
  - Most important paper for this candidate: small language models benefit from a forward curriculum from NTP to MTP.
- Tianle Cai et al., *Medusa* (2024): https://arxiv.org/abs/2401.10774
  - Additional evidence that extra future-token heads can be trained on top of a shared LM trunk with useful behavior and manageable overhead.
- Yehui Tang et al., *A Survey on Transformer Compression* (2024): https://arxiv.org/abs/2402.05964
  - Useful framing: training-only improvements are attractive in compression-constrained settings because they can improve quality without expanding the deployable artifact.

## What changed versus the chosen base implementation

This candidate starts from `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` and makes four focused changes:

1. **Turn the dormant MTP path into a deliberate candidate configuration**
   - The candidate run uses `MTP_NUM_HEADS=2`.
   - `MTP_LOSS_WEIGHT` defaults to `0.1`, and `MTP_NUM_HEADS` remains opt-in so the training-only heads can still be dropped from the plain export checkpoint without creating a reload mismatch for the default config.

2. **Add a forward curriculum for MTP head weights**
   - New knobs:
     - `MTP_HEAD_DECAY` default `0.5`
     - `MTP_START_FRAC` default `0.10`
     - `MTP_STAGE_FRAC` default `0.15`
   - Head 1 ramps first.
   - Head 2 ramps later and at lower weight.
   - This directly follows the small-model curriculum intuition from the ACL 2025 paper.

3. **Make the schedule compile-safe**
   - The live per-head weights are stored in a non-persistent tensor buffer and updated each step.
   - This avoids relying on Python or class attributes that are easy for `torch.compile` to constant-fold away, which matters because this repo already hit that exact failure mode with late QAT.

4. **Add a FlashAttention fallback for local smoke imports**
   - CUDA + FlashAttention remains the fast path.
   - When `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA.
   - This is mainly for local importability and tiny CPU smoke checks, not the intended challenge runtime.

## Running from the candidate directory

From this folder:

```bash
cd candidates/202604011710_staggered-mtp

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.1 MTP_HEAD_DECAY=0.5 \
MTP_START_FRAC=0.10 MTP_STAGE_FRAC=0.15 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script now resolves dataset and tokenizer defaults relative to the repository root, so it can be launched directly from this candidate directory without rewriting `DATA_PATH` or `TOKENIZER_PATH`.
MTP is opt-in via `MTP_NUM_HEADS`, which keeps the plain `final_model.pt` export consistent with the training-only head stripping that this candidate uses.
EMA is always applied in-code with the base stack's `0.997` decay, so there is no separate `EMA_ENABLED` flag in this candidate.

## Main risks and tradeoffs

- **Training overhead**: MTP adds extra logits/loss work during training, so the schedule may help quality but still reduce total steps under the 10-minute wallclock.
- **Already-strong base**: this stack is so optimized that the remaining headroom may be only a few thousandths of BPB.
- **Interaction with TTT**: some gains may be partially hidden once legal TTT is applied at eval time.
- **Schedule sensitivity**: the exact ramp fractions and head weighting (`0.1`, decay `0.5`) are heuristic defaults, not yet tuned on real H100 runs.

## Validation

Commands run from the repository root during this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202604011710_staggered-mtp/train_gpt.py
```

Outcome:

- Both compileall checks succeeded.

Attempted smoke validation:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch
path = Path('candidates/202604011710_staggered-mtp/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

Outcome:

- Not feasible in this runner because its Python environment does not have `torch` or `sentencepiece` installed (`importlib.util.find_spec('torch')` and `find_spec('sentencepiece')` both returned `None`).
- The candidate includes a FlashAttention fallback specifically so this kind of tiny import/forward smoke check is possible once the repository dependencies are installed.
