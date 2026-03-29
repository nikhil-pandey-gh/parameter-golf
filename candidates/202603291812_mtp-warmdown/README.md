## Candidate: Warmdown-Annealed Multi-Token Prediction on the LeakyReLU² + Legal TTT stack

### Hypothesis

Add a small multi-token prediction (MTP) auxiliary loss to the current best stack so the model learns more from each 10-minute training token budget, then anneal that auxiliary loss away during warmdown so the final weights re-specialize for next-token prediction before EMA, quantization, and eval.

### Why this is promising for this repository

This repository is bottlenecked by fixed wallclock training rather than by raw parameter count alone, so sample efficiency matters a lot. The strongest recent record already carries dormant MTP scaffolding and excludes MTP heads from export, which makes MTP unusually cheap to test here: training can benefit from the auxiliary objective while the final artifact size stays unchanged because the extra heads are dropped before serialization.

The current record history also shows that "late specialization" ideas help. Recent improvements came from warmdown tuning, EMA, and post-training quantization refinements, so it is natural to treat MTP as an early-to-mid training aid rather than a permanent objective all the way to the exported model.

### Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Best overall result in the repository snapshot.
  - Supplies the base stack: LeakyReLU(0.5)^2, Parameter Banking + Parallel Muon, legal score-first TTT, partial RoPE, LN scale, VE128, and the existing dormant MTP code path.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best pure-model stack without TTT.
  - Reinforced the importance of warmdown discipline and keeping export-time changes cheap and artifact-aware.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Showed that zero-parameter training refinements like partial RoPE and LN scaling can still matter late in the search.

### External research that informed it

- **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`)
  - Argues that predicting multiple future tokens with independent heads improves training sample efficiency and downstream capability with no inference-time obligation to keep those heads.
- **Elias Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"** (`arXiv:2210.17323`)
  - Motivates preserving the existing export-time quantization path rather than spending the change budget on a broader infrastructure rewrite.
- **Haotian Tang et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"** (`arXiv:2306.00978`)
  - Reinforces the repo trend that export-aware choices can matter as much as architecture changes, so the candidate keeps the artifact path intact and focuses the new idea on training efficiency.

### What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`

2. **Anneal MTP during warmdown**
   - New hyperparameter: `MTP_WARMDOWN_THRESHOLD=0.25`
   - While LR scale is above `0.25`, MTP uses the full auxiliary weight.
   - Once LR scale falls below `0.25`, the auxiliary weight decays linearly to zero by the end of training.

3. **Make the MTP weight compile-safe**
   - The runtime MTP weight is passed into `GPT.forward(...)` as a tensor instead of relying on a mutable Python attribute that `torch.compile` could constant-fold.
   - If `MTP_LOSS_WEIGHT<=0` at launch, the script now builds the model with `MTP_NUM_HEADS=0`, so the no-MTP ablation path stays outside compiled control flow.

4. **Fix a dormant MTP optimizer gap in the base stack**
   - The copied `2026-03-23` script defined `mtp_heads` and used them in the loss, but did not add their weights to the non-banked optimizer parameter lists.
   - This candidate explicitly adds `mtp_heads` to the replicated AdamW parameter set so the auxiliary heads actually train.

5. **Add logging for the MTP schedule**
   - Training logs now record the current runtime `mtp_weight`.

6. **Keep export behavior artifact-safe**
   - Export still strips `mtp_heads.*` before saving `final_model.pt` and the int6 artifact.
   - As a result, any manual reload of exported weights should use `MTP_NUM_HEADS=0`; the script already does this automatically for its own post-quant eval path.

### How to run

From the candidate directory:

```bash
cd candidates/202603291812_mtp-warmdown
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_WARMDOWN_THRESHOLD=0.25 \
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

The MTP defaults are already baked into this candidate, so the three `MTP_*` environment variables are only needed if you want to override them. For an ablation, either `MTP_NUM_HEADS=0` or `MTP_LOSS_WEIGHT=0` now disables the extra MTP head work entirely. If you reload an exported `final_model.pt` manually, set `MTP_NUM_HEADS=0` because exported artifacts omit the auxiliary MTP heads.
EMA with decay `0.997` remains part of the copied base script and does not need a separate environment flag.

### Main expected risks and tradeoffs

- **Throughput risk**: even lightweight MTP adds extra output-head and cross-entropy work, so step time may rise enough to offset the sample-efficiency gain.
- **Objective mismatch risk**: if the MTP weight is too large or decays too late, the model may improve future-token structure while slightly hurting the final next-token objective used at export/eval.
- **Optimizer interaction risk**: this base stack already uses Parameter Banking + Parallel Muon, EMA, TTT, and mixed quantization, so any new training signal can interact nonlinearly with the existing recipe.
- **No leaderboard run yet**: this candidate is code-complete and lightly validated, but it has not been benchmarked on the real 8xH100 path in this environment.

### Validation

Commands run from the repository root:

```bash
python -m compileall candidates/202603291812_mtp-warmdown/train_gpt.py
```

Outcome:

- `compileall` succeeded.

Attempted smoke test:

```bash
python - <<'PY'
# import candidate module and run a tiny CPU forward/backward with a local flash-attn stub
PY
```

Outcome:

- Not completed in this runner because `python` does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`), so a true CPU execution smoke test was not feasible without adding new infrastructure.
