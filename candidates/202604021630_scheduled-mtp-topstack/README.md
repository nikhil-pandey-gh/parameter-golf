# Scheduled MTP on the top stack

## Hypothesis

The repository's best training stack already carries dormant multi-token prediction (MTP) heads, but the serious record logs all ran with `mtp_num_heads:0`. Recent MTP results suggest auxiliary future-token prediction improves sample efficiency, so a scheduled 2-head MTP objective should be a better fit for this 600-second budget than deeper recurrent reuse. Because the MTP heads are still excluded from export, this trades only training-time compute for better optimization signal rather than spending artifact bytes.

## Why this is promising here

- The challenge is wallclock-limited, so training-only sample-efficiency gains are attractive.
- The top stack is already mature on quantization, XSA, EMA/SWA, partial RoPE, and legal TTT; a new lever should ideally live outside that saturated lane.
- Repo evidence argues against simple depth recurrence under the 10-minute cap, while MTP adds supervision without changing the exported model.
- This candidate also fixes a wiring issue in the inherited MTP path: the record code defined `mtp_heads` but did not add their weights to any optimizer group, so enabling the loss would not have actually trained those heads.

## Repository influences

- **Primary base:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Strongest overall public stack: LeakyReLU(0.5)^2, parameter banking, parallel Muon, partial RoPE, LN scale, VE, GPTQ-lite int6, legal TTT.
- **Static-stack reference:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best non-TTT record and the clearest GPTQ-lite / EMA / warmdown baseline.
- **Negative evidence used to reject recurrence:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` and `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - Both explicitly note depth recurrence / looped layers underperforming at this wallclock budget.
- **Prior candidates:** none existed when this candidate was created.

## External research

- **Primary source:** Fabian Gloeckle et al., *Better & Faster Large Language Models via Multi-Token Prediction* ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737))
  - Motivates MTP as a training-time auxiliary objective that improves sample efficiency.
- **Alternatives reviewed but not chosen:**
  - *Universal Transformer* ([arXiv:1807.03819](https://arxiv.org/abs/1807.03819))
  - *ALBERT* ([arXiv:1909.11942](https://arxiv.org/abs/1909.11942))
  - Both support parameter sharing / recurrent depth in principle, but the repo's own experiments already report recurrence as a poor trade in this challenge regime.

## What changed versus the chosen base

Base file: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

1. **MTP defaults are turned on**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
2. **Scheduled MTP weighting**
   - Linearly ramps in over `MTP_WARMUP_STEPS` (default 250)
   - Linearly decays once LR warmdown scale drops below `MTP_DECAY_START_SCALE` (default 0.25)
   - Hard-disables the auxiliary path once the scheduled scale becomes negligible, so late warmdown can recover throughput instead of only shrinking the loss weight
   - This keeps the auxiliary signal strong during the high-learning-rate phase and fades it during late quantization-sensitive refinement.
3. **MTP head initialization from the main LM head**
   - MTP heads start from the tied embedding / main LM head weights instead of zero-init.
4. **Runtime tensor control for MTP scale**
   - The training loop passes the MTP scale as a runtime tensor to the forward pass instead of relying on a mutable compile-time constant.
5. **Optimizer wiring fix**
   - `mtp_heads` are now explicitly added to the Adam parameter set so the auxiliary heads actually train.
6. **Path defaults resolve from the script location**
   - Default dataset and tokenizer paths now work when running from inside this candidate directory.

## How to run

From the repository root:

```bash
cd candidates/202604021630_scheduled-mtp-topstack

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 \
MTP_WARMUP_STEPS=250 MTP_DECAY_START_SCALE=0.25 MTP_INIT_FROM_MAIN_HEAD=1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script keeps the inherited export behavior: MTP heads are excluded from `final_model.pt` and `final_model.int6.ptz`, so the artifact budget is still paid only by the base model plus code.

## Validation

Commands run for this candidate:

- `python -m compileall candidates/202604021630_scheduled-mtp-topstack/train_gpt.py` - passed

CPU-only runtime smoke testing was **not feasible** without changing behavior, because this script intentionally preserves the CUDA + FlashAttention training path from the chosen base implementation and raises when CUDA is unavailable.

## Main risks and tradeoffs

- **Throughput risk:** two extra full-vocab auxiliary heads add training-time matmuls; if step time rises too much, any sample-efficiency gain could be canceled out.
- **TTT interaction risk:** legal TTT may compress the gap between candidate variants, so the cleanest signal may show up first in pre-TTT or static sliding-window metrics.
- **Initialization bias:** copying the main LM head into each MTP head should stabilize early training, but it may also reduce specialization if the heads stay too correlated.
- **Compile sensitivity:** this candidate deliberately avoids a compile-time constant for the MTP schedule, but it still inherits a heavily compiled top-stack script and should be watched for graph/regression surprises on real H100 runs.
