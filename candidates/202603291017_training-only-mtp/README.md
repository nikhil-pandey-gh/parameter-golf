# Training-Only MTP on the LeakyReLU + Legal TTT Stack

## Hypothesis

Turn on a single **training-only multi-token prediction (MTP)** head on top of the current best `LeakyReLU² + Legal TTT + Parallel Muon` stack so the model gets an auxiliary **two-step-ahead** prediction signal during training, while still exporting the same artifact family and size at evaluation time.

The key bet is that MTP improves sample efficiency under the fixed 10-minute wallclock budget, and that this repo is especially well suited for it because the auxiliary head is **stripped before export**, so the artifact budget does not pay for the extra training-only parameters.

## Why this is promising for this repository

The repository history points to two hard constraints:

- the best recent stacks are already close to the `16 MB` artifact limit, and
- most of the easy eval and quantization wins have already been mined.

That makes **training-only improvements** unusually attractive. In particular:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` is the best current stack (`val_bpb 1.1194` with legal TTT),
- multiple recent record scripts already contain MTP code paths,
- but the public record configs kept MTP disabled, and
- after the parameter-banking refactor, the `2026-03-23` script no longer wired `mtp_heads` into any optimizer, so enabling MTP there would not have actually trained the auxiliary heads.

This candidate keeps the strongest known architecture and export path, then adds the missing optimizer wiring plus sensible MTP defaults.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - chosen as the main base because it is the best current overall result and already contains the mature LeakyReLU², TTT, Parallel Muon, parameter banking, XSA, partial RoPE, LN scale, VE, and mixed int6 export stack.

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - useful as the clean non-TTT reference for the 11-layer XSA/partial-RoPE/EMA family.

- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
  - important because its README explicitly shows `MTP_NUM_HEADS=0`, which makes MTP an obvious underexplored lever rather than a repeated prior record.

## External research that informed it

- **Multi-Token Prediction** — https://arxiv.org/abs/2404.19737  
  Motivates the core idea: extra future-token heads can improve training efficiency even though they are discarded at inference/export time.

- **ALBERT** — https://arxiv.org/abs/1909.11942  
  Relevant because parameter sharing is attractive under a byte cap, but local repo evidence already shows naive recurrence/depth reuse is risky here, so I treated sharing as a secondary direction rather than the next candidate.

- **Universal Transformer** — https://arxiv.org/abs/1807.03819  
  Useful for the depth-reuse search space, but again higher risk for this repo’s strict 10-minute budget.

- **QLoRA** — https://arxiv.org/abs/2305.14314  
  Reinforces that post-training/export tricks still matter, but the repo already explored quantization aggressively enough that a zero-export-byte training trick looked like the better next move.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes:

1. **Enable one training-only MTP head by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.1`

2. **Actually optimize the MTP head**
   - the copied parameter-banked base created `mtp_heads` and used them in the forward loss,
   - but did not add them to any optimizer,
   - this candidate routes their matrix weights through Muon so the auxiliary objective is real rather than dead.

3. **Add a FlashAttention fallback**
   - if `flash_attn_interface` is unavailable, attention falls back to `torch.nn.functional.scaled_dot_product_attention`,
   - this is mainly for smoke-testability and local portability,
   - CUDA + FlashAttention 3 remains the intended fast path for real runs.

## How to run or evaluate it

Leaderboard-style run (same base recipe, with MTP enabled):

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.1 \
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

Faster non-TTT check:

```bash
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.1 TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- **Extra compute per step**: even one auxiliary head costs more logits work, so total training steps in 600 seconds may fall.
- **Objective interference**: the MTP signal could help representation learning or could dilute the late-stage next-token objective.
- **Quantization interaction**: better pre-quant features do not always survive the mixed int6 export path.
- **Base complexity**: because this candidate intentionally builds on the strongest current stack, the code is not minimal in an absolute sense even though the delta is small.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202603291017_training-only-mtp/train_gpt.py
```

Outcome:

- `python -m compileall` **passed**

Attempted smoke test:

```bash
python - <<'PY'
import importlib.util
import pathlib
import torch
path = pathlib.Path('candidates/202603291017_training-only-mtp/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

Outcome:

- blocked in this workflow environment because the runner Python does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`)
- the candidate still adds a FlashAttention-to-SDPA fallback so that import/forward smoke tests are possible in a normal local environment with PyTorch installed, even without `flash_attn_interface`
