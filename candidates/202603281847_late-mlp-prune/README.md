# Candidate: Late MLP Prune + LeakyReLU^2

## Hypothesis

The repo has already squeezed a lot out of evaluation tricks, quantization tuning, and deep-layer attention changes. The next underexplored lever is **storage-aware late pruning**: gradually zero a modest fraction of MLP weights only during the late warmdown, so the final dense checkpoint stays trainable but compresses better under the existing int6/zstd export path.

I pair that with the already-proven **LeakyReLU(0.5)^2** activation from the current top record, because it is a cheap quality win that does not require new infrastructure.

## Why this is promising for this repository

Three repo trends point in the same direction:

1. The strongest records keep finding gains by **recovering bytes** and reinvesting them in model quality rather than changing the whole training stack.
2. MLP-heavy designs already quantize/compress well here, including the mixed int5/int6 record where MLPs were the safest tensors to squeeze.
3. Pruning has only appeared as a very small side tweak so far (for example the 10-layer int5 MLP record mentions **3% magnitude pruning**), not as a real late-training schedule.

That makes late MLP sparsification a good next bet: it directly targets the artifact objective, fits the existing export path, and only adds a small amount of training logic.

## Prior records that influenced this candidate

There is no `candidates/` directory in this checkout, so the prior-experiment review came from `records/` only.

The main influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - This is the implementation base. It is the cleanest strong non-TTT stack: 11 layers, 3x MLP, XSA on the last 4 layers, partial RoPE, LN scaling, VE128, EMA, GPTQ-lite export, and tuned warmdown.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - This contributed the `LeakyReLU(0.5)^2` MLP activation idea.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - This record is the clearest repo evidence that MLP weights are the safest place to trade precision/compression and that small pruning is at least plausible.
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`
  - This record explicitly highlights why compression-aware tensor choices matter in this challenge.

## External research that informed it

The main research inputs were:

- Han et al., **Deep Compression** (2015): pruning + quantization directly improve storage efficiency.
  - <https://arxiv.org/abs/1510.00149>
- Zhu & Gupta, **To prune, or not to prune** (2017): gradual pruning schedules preserve quality better than one-shot pruning.
  - <https://arxiv.org/abs/1710.01878>
- Zhai et al., **Exclusive Self Attention (XSA)** (2026): useful context for why this repo's deep-layer attention stack is already strong, so the next candidate should avoid redundant attention surgery.
  - <https://arxiv.org/abs/2603.09078>

## What changed versus the chosen base implementation

Relative to the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` base, this candidate makes four changes:

1. **LeakyReLU(0.5)^2 MLP activation**
   - Replaces `relu^2` with the stronger activation used in the current top record.

2. **Late gradual pruning for MLP weights only**
   - Adds `PRUNE_START_SCALE`, `PRUNE_END_SCALE`, `MLP_TARGET_SPARSITY`, and `PRUNE_EVERY` knobs.
   - Applies a cubic late-training schedule that ramps toward the target sparsity only after the learning-rate scale has fallen below `PRUNE_START_SCALE`.
   - Targets only `blocks.*.mlp.fc.weight` and `blocks.*.mlp.proj.weight`.

3. **Mask reapplication plus optimizer-state cleanup**
   - Reapplies the pruning masks after Muon/Adam updates so pruned weights stay zero.
   - Zeros Muon momentum buffers on pruned entries to prevent immediate regrowth.
   - Reapplies the final mask after EMA weights are loaded so export uses the intended sparse model.

4. **Candidate-folder usability fixes**
   - Default data/tokenizer paths are resolved relative to the repository root, so the script can be run from inside this candidate directory.
   - Falls back to PyTorch SDPA if `flash_attn_interface` is unavailable, which helps local smoke setups even though the intended fast path is still FlashAttention on CUDA.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202603281847_late-mlp-prune
RUN_ID=late_mlp_prune \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
MUON_WD=0.04 ADAM_WD=0.04 \
PRUNE_START_SCALE=0.20 PRUNE_END_SCALE=0.05 \
MLP_TARGET_SPARSITY=0.18 PRUNE_EVERY=50 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
VAL_LOSS_EVERY=4000 TRAIN_LOG_EVERY=500 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Suggested first ablations:

- `MLP_TARGET_SPARSITY=0.12`, `0.18`, `0.24`
- `PRUNE_START_SCALE=0.25` vs `0.20`
- keep pruning on MLP weights only before touching attention/embeddings

## Main expected risks / tradeoffs

- The compression gain may not fully offset the quality loss from zeroing MLP weights.
- A global threshold across all MLP layers may over-prune a few sensitive late layers.
- The extra pruning pass every 50 steps adds some overhead during the most timing-sensitive part of training.
- This candidate intentionally stays off legal TTT and parameter banking so the pruning effect is easier to isolate; if it works, the next step would be to port it onto the final 2026-03-23 stack.

## Validation

I ran the lowest-cost checks that fit this environment:

- `python -m compileall candidates/202603281847_late-mlp-prune/train_gpt.py`
  - **Passed**.
- Minimal CPU forward smoke test
  - **Not feasible in this runner** because the available Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
  - Because of that environment limitation, I could not execute a real forward pass here without introducing new heavy infrastructure.
