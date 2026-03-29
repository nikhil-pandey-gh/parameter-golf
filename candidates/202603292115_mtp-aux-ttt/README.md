# Candidate: Horizon-Decayed MTP on the LeakyReLU² + Legal TTT Stack

## Hypothesis

The current best stack already spends almost the full 16 MB budget, so the cleanest remaining lever is a **training-only** one: add a lightweight multi-token prediction (MTP) auxiliary loss that improves sample efficiency during the 10-minute run, then discard the auxiliary heads at export time.

This candidate uses **two future-token heads** with a **horizon-decayed weighting** so the +2 token target still regularizes the trunk but does not overpower the main next-token objective. The artifact stays governed by the same quantized base model because the MTP heads are never exported.

## Why this is promising for this repository

Repository evidence points to two persistent truths:

- The biggest durable gains came from better **sample efficiency** and **evaluation-aware training/export choices**, not from spending more artifact bytes.
- The current best record, `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, is already near the artifact cap at about 15.95 MB, so ideas that improve the trunk **without changing the exported model** are disproportionately attractive.

The older `11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` codebase already carried dormant MTP hooks, but published runs kept `MTP_NUM_HEADS=0`, and the newer parameter-banked trainer no longer wires MTP head weights into an optimizer. That makes MTP a strong “not fully tried yet” direction rather than a repeated failed experiment.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - Best current result (`1.1194` mean post-TTT bpb).
  - Supplies the base architecture: 11 layers, bigram hash, partial RoPE, value embeddings, GPTQ-lite int6 export, legal score-first TTT, and parameter banking / parallel Muon.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - Reinforced that export-aware tweaks can still move the needle late in the search.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
  - Important because it contains the original dormant MTP path that inspired this candidate.
- `records/track_10min_16mb/2026-03-17_LoRA_TTT` and `records/track_10min_16mb/2026-03-19_SlidingWindowEval`
  - Showed how much this benchmark rewards stronger context use and evaluation-aware recipes.

## External research that informed the choice

### Chosen direction

- **Gloeckle et al., “Better & Faster Large Language Models via Multi-Token Prediction”** (`arXiv:2404.19737`)
  - Argues that predicting multiple future tokens from a shared trunk improves sample efficiency and helps induction-like behavior.
  - This is a strong fit for a 600-second training budget because it changes the training signal more than the deployed model.
- **Cai et al., “Medusa”** (`arXiv:2401.10774`)
  - Focuses on decoding speed, but it reinforces the practicality of extra future-token heads as a lightweight extension on top of an autoregressive trunk.

### Researched but not chosen for this candidate

- **QuaRot** (`arXiv:2404.00456`) and **SpinQuant** (`arXiv:2405.16406`)
  - Very relevant to the repo’s quantization bottleneck, but they require rotation machinery across many weights and activations and are riskier to adapt precisely in one candidate iteration.
- **SSMax** (`arXiv:2501.19399`)
  - Interesting for long-context attention, but replacing the attention normalization throughout this highly optimized FlashAttention-based stack is a broader architectural edit than this candidate aims to make.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
   - `MTP_LOSS_DECAY=0.7`

2. **Horizon-decayed auxiliary weighting**
   - The +1 token head gets full weight.
   - Each later horizon is downweighted by `MTP_LOSS_DECAY ** k`.
   - This keeps the auxiliary helpful without letting longer-horizon targets dominate a tiny-model training run.

3. **Optimizer wiring for MTP heads**
   - The parameter-banked trainer comments that MTP heads should ride Adam with other small replicated parameters, but the base script never actually adds them.
   - This candidate explicitly adds MTP head weights to the Adam-managed parameter set, making the auxiliary trainable.

4. **Export remains unchanged in size-critical ways**
   - MTP heads are still excluded from `export_sd`.
   - The quantized eval/TTT model is still rebuilt with `mtp_num_heads=0`, preserving the deployed architecture and legal TTT behavior.

## How to run or evaluate it

From this candidate directory. The default `DATA_PATH` and `TOKENIZER_PATH` now resolve from the trainer file location, so the command works without rewriting them when launched from here:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_LOSS_DECAY=0.7 \
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

```bash
python -m compileall candidates/202603292115_mtp-aux-ttt/train_gpt.py
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location(
    "candidate_train_gpt",
    "candidates/202603292115_mtp-aux-ttt/train_gpt.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("module import smoke test passed")
PY
```

### Outcomes

- `python -m compileall candidates/202603292115_mtp-aux-ttt/train_gpt.py`: **passed**.
- CPU-only import smoke test: **blocked by the runner environment**, which does not currently have the repo's Python dependencies installed (`ModuleNotFoundError: No module named 'numpy'` before trainer setup). That means a no-GPU start-up smoke test was not feasible here without first provisioning the full dependency stack.

## Main expected risks or tradeoffs

- **Training overhead**: extra heads add forward and backward work, so step count may drop slightly within the 600-second cap.
- **Small-model over-regularization**: if the auxiliary is too strong, the model may trade next-token sharpness for broader-but-weaker future-token features.
- **Incremental upside ceiling**: the base record is already very strong, so the expected gain is probably modest unless MTP genuinely improves trunk efficiency in this low-parameter regime.
- **TTT interaction**: although export and TTT rebuilds drop the MTP heads, any pretraining-side changes must still translate into better post-quant and post-TTT behavior to matter.
