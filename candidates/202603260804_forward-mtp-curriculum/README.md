# Forward-Curriculum MTP on the LeakyReLU² + Legal TTT Stack

## Hypothesis

The strongest current stack is already close to saturating obvious architecture and quantization wins, so the next incremental gain is more likely to come from **better sample efficiency during the fixed 600s training window** than from shipping a larger artifact. This candidate adds **forward-curriculum multi-token prediction (MTP)** to the current best `2026-03-23` stack.

The key bet is that a tiny model can benefit from MTP **if it is introduced gradually** rather than turned on at full strength from step 0. The auxiliary heads are **training-only** and are still excluded from the exported int6 artifact, so the idea targets train-time efficiency without spending additional artifact bytes.

## Why this is promising for this repository

The repository history shows a clear pattern:

- recent gains came from stronger objectives, evaluation, and quantization rather than from radically new infrastructure,
- the top stack already has dormant MTP support in code,
- but the recorded leaderboard runs kept `MTP_NUM_HEADS=0`, so MTP was never actually exercised in the winning configs.

That makes MTP an unusually attractive next candidate here: it is **already close to the codebase**, **zero-cost at export time**, and it attacks the exact bottleneck this track has: getting more useful learning signal out of a hard 10-minute train budget.

## Prior records and experiments that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - current best score in the repo (`val_bpb: 1.1194`)
  - provides the LeakyReLU(0.5)^2 MLP, legal score-first TTT, parameter banking, Parallel Muon, GPTQ-lite int6, and lzma export stack.

Supporting precedent:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest pure-model ancestor of the current record stack
  - still contains the MTP code path and shows the pre-TTT training stack is already strong enough that small objective improvements are worth testing.

- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
  - another strong 11-layer ancestor whose README explicitly kept `MTP_NUM_HEADS=0`, reinforcing that MTP remains a real gap rather than a repeated idea.

## External research that informed it

- **Fabian Gloeckle et al., “Better & Faster Large Language Models via Multi-token Prediction” (arXiv:2404.19737, 2024)**
  - argues that predicting multiple future tokens with auxiliary heads improves sample efficiency and downstream behavior while keeping the shared trunk unchanged.

- **Ansar Aynetdinov and Alan Akbik, “Pre-Training Curriculum for Multi-Token Prediction in Language Models” (arXiv:2505.22757, ACL 2025)**
  - specifically relevant here because it reports that **small language models struggle with raw MTP**, but a **forward curriculum** that moves from NTP to MTP gradually makes MTP useful again.

This candidate is essentially the repository-specific adaptation of that research result: use MTP, but introduce it conservatively enough for a small 11-layer model under a short wallclock budget.

## What changed versus the chosen base implementation

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

- changes the default training objective from plain NTP to **2-head MTP**,
- sets a conservative default `MTP_LOSS_WEIGHT=0.15`,
- adds a **forward curriculum** controlled by:
  - `MTP_CURRICULUM_ENABLED`
  - `MTP_CURRICULUM_START_FRAC`
  - `MTP_CURRICULUM_END_FRAC`
- implements the curriculum as runtime-updated loss weights so it can coexist with the compiled training loop,
- restores actual MTP training by wiring the MTP head weights back into the matrix optimizer path.

That last point matters: the `2026-03-23` script still had MTP code in the model, but after the parameter-banking refactor the MTP heads were no longer added to any optimizer group. Enabling MTP on that script would therefore have been ineffective. This candidate fixes that gap.

## How the curriculum works

The schedule is intentionally simple and compile-friendly:

- before `MTP_CURRICULUM_START_FRAC`, the model trains as plain next-token prediction,
- between `START_FRAC` and `END_FRAC`, MTP heads are enabled **progressively**,
- after `END_FRAC`, the full configured MTP objective is active.

By default:

- `MTP_NUM_HEADS=2`
- `MTP_LOSS_WEIGHT=0.15`
- `MTP_CURRICULUM_START_FRAC=0.10`
- `MTP_CURRICULUM_END_FRAC=0.45`

So the run spends the earliest phase learning the baseline task, then gradually layers in farther-ahead prediction once the trunk has stabilized.

## How to run or evaluate it

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 \
MTP_CURRICULUM_ENABLED=1 MTP_CURRICULUM_START_FRAC=0.10 MTP_CURRICULUM_END_FRAC=0.45 \
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

To isolate the new idea from the TTT path, the cleanest ablation is:

```bash
TTT_ENABLED=0 MTP_NUM_HEADS=2 MTP_CURRICULUM_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- **Training-speed risk:** MTP adds extra vocab-head work every step. If the step-count loss is too large, the auxiliary objective may not pay for itself.
- **Small-model fragility:** the ACL 2025 paper is encouraging, but it also explicitly says smaller models can regress under naive MTP; the curriculum may still need tuning.
- **TTT interaction risk:** better pretraining may combine well with legal TTT, but it could also reduce or reshape the later TTT gain.
- **Optimizer interaction risk:** MTP heads are now genuinely trained again, but their best optimizer settings on top of the Parallel Muon stack may still need retuning once real GPU runs are available.

## Validation

Commands run in this workflow:

- `python -m compileall candidates/202603260804_forward-mtp-curriculum/train_gpt.py`
  - **Passed**
- `python -m compileall train_gpt.py train_gpt_mlx.py data`
  - **Passed**

Attempted smoke check:

- I attempted a CPU-only smoke test by stubbing FlashAttention and importing the candidate model for a tiny forward/backward pass.
- That runtime smoke test was **not feasible in this workflow environment** because the available Python interpreter does not currently have `torch` or `sentencepiece` installed, and the script also depends on FlashAttention/CUDA for real execution.
- As a result, validation here is limited to syntax-level checks plus static inspection of the modified training path.
