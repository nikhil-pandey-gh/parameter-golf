# Training-Only MTP on the March 23 Parallel-Muon Stack

## Hypothesis

Adding a **small training-only multi-token prediction (MTP) auxiliary loss** to the strongest published March 23 stack can improve sample efficiency under the fixed 600-second training budget without increasing exported artifact size.

The concrete bet here is conservative:

- keep the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` model family intact,
- add **one** auxiliary future-token head by default (`MTP_NUM_HEADS=1`),
- keep the auxiliary loss light (`MTP_LOSS_WEIGHT=0.1`),
- and continue to **strip `mtp_heads` from export** so the saved artifact stays focused on the main model.

## Why this is promising for this repository

Repository review suggests the obvious wins have mostly been harvested already:

- evaluation-context improvements are already strong (`records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md`),
- compression-aware export has been heavily optimized (`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`),
- and the latest record stacks already combine many small architectural wins (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`).

By contrast, the March 20-23 code family already carried dormant `mtp_num_heads` support in `train_gpt.py`, but the released logs and READMEs still used `mtp_num_heads:0`. That makes MTP attractive here: it is a genuine new training-objective change, but it reuses existing code paths instead of requiring a broad systems rewrite.

## Prior records and candidates that influenced this choice

There was **no pre-existing `candidates/` directory** in the checkout when this candidate was created, so the comparison set is entirely the `records/` lineage.

The most relevant prior records were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest published stack in this checkout,
  - contributes the base architecture, LeakyReLU(0.5)^2 MLP, parameter banking, parallel Muon, and legal TTT path.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - reinforces the importance of post-training/export quality and careful weight averaging,
  - shows that small quality gains in the quantized model are still meaningful at this stage.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - useful because it already contained MTP plumbing in the script, even though released runs kept it disabled.

## External research that informed it

- Fabian Gloeckle et al., **“Better & Faster Large Language Models via Multi-token Prediction”** (`arXiv:2404.19737`)
  - argues that auxiliary future-token prediction improves sample efficiency with multiple output heads on a shared trunk.
- DeepSeek-AI et al., **“DeepSeek-V3 Technical Report”** (`arXiv:2412.19437`)
  - includes a multi-token prediction training objective in a modern high-performing model family.

I also considered more quantization-centric ideas such as QuaRot / SpinQuant-lite, but those looked like wider reparameterization changes for a first candidate in this repo. MTP fit the repository’s current structure better.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate makes four focused changes:

1. **Turn MTP on by default**
   - `MTP_NUM_HEADS` now defaults to `1`.
   - `MTP_LOSS_WEIGHT` now defaults to `0.1`.

2. **Seed the auxiliary head from the tied embedding**
   - new flag: `MTP_INIT_FROM_EMBED=1` by default,
   - when tied embeddings are enabled, the auxiliary head copies `tok_emb.weight` at init so the extra future-token objective has a useful lexical projection immediately instead of waiting for a zero-init head to learn one.

3. **Make the MTP head actually train on the March 23 optimizer split**
   - the latest parallel-Muon stack no longer added `mtp_heads` to any optimizer parameter group,
   - this candidate restores that wiring by routing the 2D MTP head weights into the Muon matrix-parameter set.

4. **Keep export behavior artifact-safe**
   - `mtp_heads` are still excluded from `export_sd`,
   - eval-time roundtrip reconstruction still reinstantiates the model with `mtp_num_heads=0`.

## How to run / evaluate

From the repository root:

```bash
cd candidates/202603301721_training-only-mtp

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.1 MTP_INIT_FROM_EMBED=1 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To isolate the training-objective change from legal TTT at evaluation time, repeat with `TTT_ENABLED=0`.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202603301721_training-only-mtp/train_gpt.py
```

Outcome: **passed**

Attempted smoke check:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch
PY
```

Outcome: **not feasible in this workflow environment** because the available Python runtime did not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`), so I could not run a local CPU forward-pass smoke harness here.

## Main expected risks / tradeoffs

- The auxiliary head adds another full-vocabulary projection during training, so **step time may rise enough to erase the sample-efficiency gain**.
- Copying the tied embedding into the MTP head may stabilize early optimization, but it could also reduce diversity between the main and auxiliary objectives.
- The best gains may show up in pre-quant quality more than post-quant `val_bpb`, especially once GPTQ-lite int6 export dominates the final score.
- Because this stack still carries legal TTT, evaluation gains from MTP may be harder to isolate unless runs are compared with `TTT_ENABLED=0` and `TTT_ENABLED=1`.
