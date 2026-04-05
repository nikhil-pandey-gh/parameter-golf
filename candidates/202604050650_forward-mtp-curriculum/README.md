# Forward-Curriculum MTP on the LeakyReLU TTT Stack

## Hypothesis

The strongest next low-risk candidate is to keep the current best 11-layer stack intact and add **training-only multi-token prediction (MTP)** with a **forward curriculum**. The extra heads should improve sample efficiency and representation quality during the 10-minute training budget, while costing **0 artifact bytes at export time** because they are stripped before serialization.

## Why this is promising here

The record history in this repository already squeezed most of the obvious gains out of quantization, evaluation, and small architectural tweaks. The top runs from 2026-03-20 through 2026-03-23 all carry dormant MTP plumbing in `train_gpt.py`, but keep it disabled with `MTP_NUM_HEADS=0`, so this line of attack has not actually been tried in a record stack yet.

That makes MTP a good fit for this repo's constraints:

- it is a **training-objective change**, not a new backbone or data pipeline,
- it can reuse the existing export path that already excludes `mtp_heads`,
- it attacks the challenge's main bottleneck here: **quality per training step**, not just raw compression.

## Prior repository influences

### Base implementation

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

This candidate starts from the current best reviewed 10-minute stack:

- 11 layers / 512 dim / 4 KV heads
- BigramHash + VE late-layer reinjection
- Partial RoPE + LN scale
- LeakyReLU(0.5)^2 MLP
- EMA + tight SWA
- GPTQ-lite int6 + lzma export
- legal score-first TTT
- parameter banking + parallel Muon

### Other record evidence

- `2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

These runs show the repo trend clearly: the frontier is made of many small stacked gains on the same 11L core. MTP is attractive precisely because it is one of the few remaining unused levers already close to the winning codepath.

### Prior candidates

There were no existing `candidates/` directories in the repository when this candidate was created.

## External research that informed this candidate

1. **Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737)**  
   The paper reports that MTP improves sample efficiency by training the model to predict several future tokens from a shared trunk, and specifically notes gains on small algorithmic tasks through better induction-head development.

2. **Aynetdinov and Akbik, "Pre-Training Curriculum for Multi-Token Prediction in Language Models" (arXiv:2505.22757)**  
   This is the most relevant paper for this repo: it explicitly says naive MTP can hurt **small language models**, while a **forward curriculum** lets SLMs recover NTP gains and improve output quality.

## What changed versus the chosen base implementation

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes four focused changes:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.12`

2. **Add a forward curriculum**
   - MTP stays off at the very start of training.
   - Head 1 ramps in first.
   - Head 2 ramps in later.
   - Only the currently active prefix of MTP heads is evaluated, so the curriculum gates compute as well as loss.
   - The schedule is driven by training progress, using wallclock-aware progress when a wallclock cap is active.

3. **Disable MTP late in warmdown**
   - Once the LR scale falls below `MTP_DISABLE_LR_SCALE=0.25`, auxiliary MTP loss goes to zero.
   - This is a deliberate repo-specific twist: late training in these records is very quantization-sensitive, so the final export-facing phase should bias back toward pure next-token training.

4. **Actually optimize the MTP heads**
   - The copied base script already excluded `mtp_heads` from export, but the heads were still dormant.
   - This candidate adds the MTP head weights to the AdamW scalar/small-matrix optimizer path so the auxiliary heads receive gradients and updates during training.

## How to run

From the repository root:

```bash
cd candidates/202604050650_forward-mtp-curriculum
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.12 \
MTP_CURRICULUM_START_FRAC=0.10 MTP_CURRICULUM_END_FRAC=0.55 \
MTP_DISABLE_LR_SCALE=0.25 \
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

## Validation

### Commands run

1. `python -m compileall train_gpt.py train_gpt_mlx.py data`
2. `python -m compileall candidates/202604050650_forward-mtp-curriculum/train_gpt.py`

### Outcomes

- Both compile checks passed.
- A CPU smoke run was **not feasible** in this workspace because the required local challenge assets are missing:
  - no `data/tokenizers/fineweb_1024_bpe.model`
  - no `data/datasets/fineweb10B_sp1024/fineweb_train_*.bin` shards
- The candidate script also remains designed for the repository's CUDA/FlashAttention training environment, matching the record stack it extends.

## Main risks and tradeoffs

- **Step-time regression:** even two extra vocabulary heads add compute every step; if the overhead is too high, the candidate may lose more steps than it gains in sample efficiency.
- **Curriculum sensitivity:** the forward schedule may need retuning relative to the actual step count reached under the 600s cap.
- **Late-phase interaction:** disabling MTP during warmdown is plausible for quantization quality, but it is still a heuristic.
- **Optimizer choice for auxiliary heads:** the current implementation routes MTP heads through the small-matrix AdamW path; a different LR or optimizer split could work better.
