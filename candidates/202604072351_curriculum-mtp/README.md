# Curriculum MTP on the 2026-03-23 stack

## Hypothesis

The strongest local stack already spends almost all of its permanent-parameter budget on architecture, quantization, and evaluation tricks. A **forward multi-token-prediction (MTP) curriculum** should be a better next bet than another permanent architectural change, because it improves training signal **without adding export bytes**: the auxiliary heads are used only during training and are stripped before final serialization.

The specific bet here is that this repo's tiny models are in the exact regime where **always-on MTP can be too hard early**, but **NTP -> shallow-horizon MTP -> full MTP** can improve sample efficiency enough to beat plain next-token training under the fixed 10-minute budget.

## Why this is promising for this repository

- The record progression shows most durable wins now come from **training/eval efficiency** and **better use of the existing 16MB budget**, not from blindly adding permanent parameters.
- The best current record already contains dormant MTP support and already excludes `mtp_heads` from export, so the repo has a natural insertion point for this idea.
- No prior `candidates/` directory existed, and none of the reviewed `records/` actually turned MTP on in their shipped configs.
- Recent MTP literature says **small language models struggle with naive MTP**, but a **forward curriculum** helps them benefit from it. That makes this more targeted than simply flipping `MTP_NUM_HEADS` from `0` to `2`.

## Prior repository runs that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current stack in this repo snapshot
  - LeakyReLU(0.5)^2, parameter banking + Parallel Muon, XSA on late layers, partial RoPE, VE128, GPTQ-lite int6 + lzma, legal score-first TTT
- **Pre-TTT backbone evolution:**
  - `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
- **Negative evidence I explicitly avoided repeating:**
  - recurrence/depth reuse looked poor under fixed wallclock in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
- **Prior candidates:** none existed.

## External research that informed the choice

1. **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"**  
   https://arxiv.org/abs/2404.19737  
   Core result: predicting multiple future tokens from a shared trunk improves sample efficiency and can improve downstream capability.

2. **Ansar Aynetdinov and Alan Akbik, "Pre-Training Curriculum for Multi-Token Prediction in Language Models"**  
   https://arxiv.org/abs/2505.22757  
   Most relevant paper for this repo: small language models benefit much more from a **forward curriculum** than from naive always-on MTP.

3. **Somesh Mehra et al., "On multi-token prediction for efficient LLM inference"**  
   https://arxiv.org/abs/2502.09419  
   Useful caution: post-hoc or poorly integrated MTP is hard; this supports doing **joint training** rather than bolting MTP on later.

4. **Catherine Olsson et al., "In-context Learning and Induction Heads"**  
   https://arxiv.org/abs/2209.11895  
   Mechanistic motivation: better future-token training signals may help earlier emergence of induction-like behavior.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

1. **Enables MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`

2. **Adds a forward curriculum over wallclock progress**
   - `MTP_CURRICULUM_ENABLED=1`
   - `MTP_CURRICULUM_START_FRAC=0.15`
   - `MTP_CURRICULUM_END_FRAC=0.55`
   - horizon `+1` ramps in first, then horizon `+2`, and both are fully active by 55% of the training budget

3. **Implements the curriculum with runtime tensor weights, not Python flags**
   - this is deliberate so the schedule is not vulnerable to the same `torch.compile` constant-folding failure mode that previously bit late-QAT in this repo

4. **Skips zero-weight MTP heads entirely**
   - the early NTP-only phase now also avoids the extra MTP projections/losses
   - this preserves the intended compute benefit of the curriculum under the 10-minute budget

5. **Adds a generic causal-attention fallback**
   - uses FlashAttention when available on CUDA
   - otherwise falls back to PyTorch SDPA
   - this makes a cheap CPU smoke path possible without changing the GPU intent of the script

6. **Adds `SMOKE_TEST=1`**
   - instantiates the model on CPU with small settings
   - runs one forward/backward pass
   - avoids dataset/tokenizer/GPU requirements for basic startup validation

7. **Finds the repo root dynamically**
   - default data/tokenizer paths now work from this candidate directory **and** continue to work if the run is later moved under `records/...`

## How to run or evaluate it

Run from this candidate directory:

```bash
cd candidates/202604072351_curriculum-mtp

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 \
MTP_CURRICULUM_ENABLED=1 MTP_CURRICULUM_START_FRAC=0.15 MTP_CURRICULUM_END_FRAC=0.55 \
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

Quick CPU-only startup smoke:

```bash
cd candidates/202604072351_curriculum-mtp

SMOKE_TEST=1 \
VOCAB_SIZE=128 NUM_LAYERS=2 MODEL_DIM=64 NUM_HEADS=4 NUM_KV_HEADS=2 \
MLP_MULT=2 TRAIN_SEQ_LEN=16 BIGRAM_VOCAB_SIZE=0 VE_ENABLED=0 XSA_LAST_N=0 ROPE_DIMS=8 \
python train_gpt.py
```

Useful ablations:

- disable curriculum but keep MTP: `MTP_CURRICULUM_ENABLED=0`
- single future-token head: `MTP_NUM_HEADS=1`
- isolate pre-TTT gain: `TTT_ENABLED=0`

## Main expected risks and tradeoffs

- **Tiny-model fragility:** even with a curriculum, 2-head MTP may still slow training enough to lose too many optimizer steps.
- **Schedule sensitivity:** the `0.15 -> 0.55` ramp is a research-backed heuristic, not an H100-tuned optimum.
- **Interaction effects:** TTT may amplify or hide the pre-TTT gain, so both `TTT_ENABLED=0` and `1` should be checked.
- **Compile/runtime behavior:** the buffer-based implementation should be safer than Python flag toggles, but the real test is still a GPU run.

## Validation run here

I ran the following low-cost checks in a temporary virtualenv with this repo's `requirements.txt` installed:

1. `python -m compileall train_gpt.py`  
   **Outcome:** passed.

2. CPU smoke test:

   ```bash
   SMOKE_TEST=1 \
   VOCAB_SIZE=128 NUM_LAYERS=2 MODEL_DIM=64 NUM_HEADS=4 NUM_KV_HEADS=2 \
   MLP_MULT=2 TRAIN_SEQ_LEN=16 BIGRAM_VOCAB_SIZE=0 VE_ENABLED=0 XSA_LAST_N=0 ROPE_DIMS=8 \
   python train_gpt.py
   ```

   **Outcome:** passed with  
   `smoke_test_ok loss:4.8519 batch:8 seq_len:16 mtp_head_weights:[0.0, 0.0]`

No real GPU training run was performed in this container.
