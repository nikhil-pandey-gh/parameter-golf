# Distance-weighted MTP on the Parallel-Muon + Legal-TTT stack

## Hypothesis

Enable the dormant train-only multi-token prediction (MTP) heads that already exist in the recent 11-layer leaderboard scripts, but make the auxiliary loss **distance-weighted** so nearer futures matter more than farther ones. The expected win is better **sample efficiency** inside the fixed 10-minute training budget, while keeping the final artifact essentially unchanged because the MTP heads are still excluded from export.

## Why this is promising for this repository

- The current best reviewed stack is `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`, which already combines the strongest repo-local ideas: LeakyReLU^2, XSA, partial RoPE, EMA/SWA, bigram features, GPTQ-lite-style mixed quantization, parameter banking, and legal score-first TTT.
- The same recent 11-layer scripts already contain MTP hooks plus explicit export-time exclusion for `mtp_heads`, but all checked training logs still ran with `mtp_num_heads:0`. That makes MTP a real untested direction here rather than a recycled record.
- External research directly supports the core idea: Gloeckle et al., *Better & Faster Large Language Models via Multi-Token Prediction* (`https://arxiv.org/abs/2404.19737`) reports improved sample efficiency from auxiliary future-token heads on top of a shared trunk.
- I also reviewed more invasive compact-model ideas such as factorized embeddings and cross-layer sharing. They are interesting future work, but MTP has the best "high upside / lowest integration risk" ratio for a single copied candidate script.

## Prior records and experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best current repo-reviewed stack
  - provides the LeakyReLU^2 + parameter-banking + legal-TTT base copied here
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - confirms the strong 11L / XSA / VE / EMA / GPTQ-lite lineage
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - earliest strong XSA + EMA 11-layer stack with dormant MTP hooks present but disabled
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful negative result: depth recurrence hurt badly under a fixed wall-clock budget, which is why this candidate avoids layer looping and instead spends compute on train-only auxiliary heads

## External research that informed it

- **Primary source:** Fabian Gloeckle et al., *Better & Faster Large Language Models via Multi-Token Prediction* (`https://arxiv.org/abs/2404.19737`)
  - motivates auxiliary future-token heads as a way to improve sample efficiency
  - especially relevant because this challenge is strongly wall-clock-limited
- **Considered but not chosen for this candidate**
  - ALBERT (`https://arxiv.org/abs/1909.11942`) for factorization / sharing
  - Universal Transformer (`https://arxiv.org/abs/1807.03819`) for recurrent depth
  - AWQ (`https://arxiv.org/abs/2306.00978`) for activation-aware quantization

Those alternatives remain plausible future work, but they require either broader export-path surgery or they conflict with repo-specific negative evidence around recurrence.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps that base stack and makes three targeted changes:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS` now defaults to `2` instead of `0`
   - the candidate therefore trains with two auxiliary future-token heads unless explicitly disabled

2. **Distance-weight the auxiliary losses**
   - new knob: `MTP_LOSS_DECAY` (default `0.5`)
   - horizon `k=1` gets weight `1.0`, `k=2` gets weight `0.5`, etc.
   - this makes the auxiliary signal less aggressive than uniform averaging and keeps the main next-token objective dominant

3. **Add smoke-testability without changing the CUDA path**
   - `flash_attn_interface` import now falls back to PyTorch SDPA when FlashAttention is unavailable
   - `SMOKE_TEST_ONLY=1` runs a tiny CPU-side forward/backward sanity check and exits before any CUDA/dataset setup

The export behavior stays the same: `mtp_heads` are still dropped before serialization, so the candidate pays training compute for MTP but not artifact bytes.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603290918_distance-weighted-mtp
```

Recommended 8xH100-style run, building directly on the current best stack and adding distance-weighted MTP:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 MTP_LOSS_DECAY=0.5 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`EMA` is always active in this script; there is no separate `EMA_ENABLED` flag.

CPU-only smoke check after installing the repo dependencies:

```bash
SMOKE_TEST_ONLY=1 python train_gpt.py
```

## Validation run for this candidate

Commands run in this workflow:

```bash
python -m compileall candidates/202603290918_distance-weighted-mtp/train_gpt.py
python - <<'PY'
from pathlib import Path
import ast
path = Path('candidates/202603290918_distance-weighted-mtp/train_gpt.py')
ast.parse(path.read_text(encoding='utf-8'))
print('ast_parse_ok')
PY
```

Outcomes:

- `compileall`: passed
- AST parse: passed

Attempted smoke check:

```bash
SMOKE_TEST_ONLY=1 python candidates/202603290918_distance-weighted-mtp/train_gpt.py
```

Outcome:

- not runnable in this workflow environment because the runner does not currently have the repository's required Python packages installed (`torch`, `numpy`, `sentencepiece`)
- the candidate script now includes a dedicated smoke path plus FlashAttention fallback so the same command should work in a properly provisioned repo environment

## Expected risks and tradeoffs

- **Training-speed tradeoff:** extra vocab heads cost compute. If the auxiliary signal is too expensive, the model could lose enough steps to wipe out any sample-efficiency gain.
- **Objective-mismatch risk:** too much MTP weight can over-regularize toward farther-future prediction and hurt the main next-token objective.
- **TTT interaction risk:** if MTP changes hidden-state geometry meaningfully, the existing legal TTT recipe may need retuning rather than transferring directly.
- **No full GPU verification here:** this candidate is syntax-checked and smoke-ready, but not benchmarked in the current runner because the Python runtime dependencies are missing.

## Suggested next experiments

1. Sweep `MTP_NUM_HEADS in {1,2,3}` with fixed `MTP_LOSS_WEIGHT=0.2`.
2. Sweep `MTP_LOSS_DECAY in {0.25, 0.5, 0.75, 1.0}` to measure how much farther-future supervision helps before it starts conflicting with NTP.
3. If MTP helps pre-TTT quality, retune `TTT_FREEZE_BLOCKS` and `TTT_LR` rather than assuming the old best TTT settings remain optimal.
4. If MTP is neutral but stable, combine it later with the factorized-embedding idea from the external research shortlist.
