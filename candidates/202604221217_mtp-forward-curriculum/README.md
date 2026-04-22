# MTP Forward Curriculum on the 1.1194 LeakyReLU² + Legal TTT stack

## Hypothesis

The latest winner already carries dormant multi-token prediction (MTP) heads and excludes them from export, so the cleanest high-upside next step is to **actually train those heads** and do it with a **forward curriculum** instead of turning MTP on from step 0.

For this tiny 11-layer model, the bet is:

1. **Raw MTP is too abrupt for a small backbone**, so enabling it late and in stages should preserve the strong next-token baseline.
2. **Auxiliary future-token supervision should improve sample efficiency** inside the fixed 10-minute budget.
3. Because `mtp_heads.*` are already excluded from export, this can improve the trunk **without paying artifact bytes**.

## Why this is promising here

The repository evidence points to two things at once:

1. The strongest stack is already very refined on architecture, quantization, and evaluation.
2. The best scripts still have unused MTP plumbing, which means there is an untried training-only lever with near-zero integration cost.

That makes MTP unusually attractive for this repo: it changes optimization pressure during training, but it does **not** need extra inference infrastructure and does **not** count toward the 16MB artifact because the auxiliary heads are still dropped before export.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current stack,
  - already includes dormant `mtp_num_heads` / `mtp_loss_weight` support,
  - already excludes `mtp_heads.*` from export.
- **Quantization/export reference:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - confirms the current 11L stack is mature on export-side tuning,
  - makes a training-only change more attractive than another near-duplicate quantization tweak.

## External research that informed it

- **Gloeckle et al., “Better & Faster Large Language Models via Multi-token Prediction” (arXiv:2404.19737)**  
  MTP can improve sample efficiency by adding future-token heads on a shared trunk.
- **Aynetdinov and Akbik, “Pre-Training Curriculum for Multi-Token Prediction in Language Models” (arXiv:2505.22757)**  
  Small language models benefit much more from a **forward curriculum** than from naïvely applying the full MTP objective from the start.
- **Mehra et al., “On multi-token prediction for efficient LLM inference” (arXiv:2502.09419)**  
  Hidden states are strongly specialized for next-token prediction, which is a good reason to ramp MTP in gradually instead of making it the full objective immediately.

## What changed versus the chosen base

Starting from `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes four focused changes:

1. **Turns MTP on by default**
   - `MTP_NUM_HEADS` now defaults to `2`.
   - `MTP_LOSS_WEIGHT` now defaults to `0.15`.

2. **Adds a forward MTP curriculum**
   - stage 1: NTP-only,
   - stage 2: half-width MTP (`1` active head for the default `2-head` setup) at reduced loss scale,
   - stage 3: full MTP.

3. **Wires the MTP heads into optimization**
   - the copied base code defined MTP heads and excluded them from export, but did not optimize them,
   - this candidate adds a dedicated Adam optimizer for the MTP head weights and includes them in replicated gradient reduction.

4. **Makes default dataset/tokenizer paths file-relative**
   - so the candidate script can be launched from its own directory without relying on repo-root cwd behavior.

## How to run

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 \
MTP_CURRICULUM_ENABLED=1 MTP_START_FRAC=0.35 MTP_FULL_FRAC=0.75 MTP_MID_LOSS_SCALE=0.5 \
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

Notes:

- The MTP heads are still excluded from the exported model artifact.
- Legal TTT is unchanged from the chosen base.
- EMA remains hardcoded in the copied base, as in the original winner.
- If you want a cleaner ablation, set `TTT_ENABLED=0` and compare the pre-TTT sliding score first.

## Validation

Commands run for this candidate:

```bash
python -m compileall candidates/202604221217_mtp-forward-curriculum/train_gpt.py
```

Outcome:

- `compileall` succeeded.
- A minimal CPU-only runtime smoke test was **not** run, because this script is still intentionally on the CUDA/FlashAttention-3/Hopper-oriented execution path and raises on non-CUDA execution.

## Main risks and tradeoffs

1. **Training-time overhead:** even export-free MTP still adds forward/backward work, so the sample-efficiency win has to beat the step-time slowdown.
2. **Small-model fragility:** the curriculum is motivated by recent small-model MTP work, but the exact stage boundaries may matter a lot.
3. **Compile behavior:** changing MTP stages during training may trigger recompilation at stage boundaries.
4. **Loss balancing:** `MTP_LOSS_WEIGHT=0.15` is a reasoned starting point, not yet a tuned optimum.
