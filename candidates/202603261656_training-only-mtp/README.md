# Training-Only Multi-Token Prediction on the 2026-03-23 Frontier Stack

## Hypothesis

The strongest next step is to add a small, training-only multi-token prediction (MTP) objective to the current best 11-layer stack so the trunk learns faster under the fixed 600-second budget, while keeping the exported artifact size effectively unchanged.

This candidate enables a conservative `t+2` auxiliary prediction head by default (`MTP_NUM_HEADS=1`, `MTP_LOSS_WEIGHT=0.15`) and strips that head from export. The bet is that extra future-token supervision improves sample efficiency enough to outweigh the modest extra training compute.

## Why this is promising for this repository

Repository review showed a clear pattern:

- the winning line moved from the simple 9-layer baseline to larger but still compressible 10-11 layer models,
- sliding-window evaluation, mixed low-bit export, EMA/SWA, and cheap architectural nudges kept stacking gains,
- but full depth recurrence already looked negative in this exact 10-minute regime,
- and the current frontier (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`) is already close to the artifact ceiling, so training-only improvements are especially attractive.

That makes MTP a good fit here: it targets training efficiency rather than exported capacity.

## Prior work that influenced this candidate

Primary local base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Key inherited ideas from the frontier line:

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` for the 11-layer XSA/EMA direction.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` for partial RoPE and LN scaling.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` for the export-aware 11-layer stack and warmdown/averaging tuning.

Negative evidence that shaped the choice:

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` and `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` both report layer/depth recurrence as a bad trade under a fixed 10-minute wall clock, so I did not choose the externally appealing shared-depth/recurrent path.

There were no prior experiments under `candidates/` when this candidate was created.

## External research that informed the choice

Primary sources:

- Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-Token Prediction" (arXiv:2404.19737). The paper argues that predicting multiple future tokens improves sample efficiency and helps induction-style behavior while leaving the main autoregressive trunk intact.
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (arXiv:2412.19437). DeepSeek-V3 explicitly uses a multi-token prediction training objective as part of a modern high-performance training recipe.

Other ideas considered during research but not chosen for this repo fork included shared-block recurrence, more aggressive quantization-aware training, and stronger KV sharing. The deciding factor was local repository evidence: recurrence had already underperformed here, while training-only auxiliary heads fit the challenge constraints cleanly.

## What changed vs the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`:

1. Enabled training-only MTP by default.
   - `MTP_NUM_HEADS` default changed from `0` to `1`.
   - `MTP_LOSS_WEIGHT` default changed from `0.20` to `0.15`.

2. Fixed MTP head optimizer wiring.
   - The base code had MTP heads in the model and in the export-exclusion path, but they were not actually attached to any optimizer group.
   - This candidate adds the MTP head parameters to the AdamW-managed unbanked parameter set, so the auxiliary objective can meaningfully train the trunk.

3. Added a safe attention fallback for local validation.
   - If `flash_attn_interface` is unavailable, the candidate falls back to PyTorch `scaled_dot_product_attention`.
   - This is meant for local import/forward smoke checks and CPU-side debugging; the intended competitive training path is still the CUDA/FlashAttention stack.

4. Improved runtime logging.
   - The script now logs whether it is using `flash_attn_3` or `torch_sdpa` attention.

## How to run

From the candidate directory:

```bash
cd candidates/202603261656_training-only-mtp
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The exported artifact still excludes `mtp_heads`, so the auxiliary objective should not directly increase submission bytes.

## Evaluation notes

- Competitive evaluation should use the repository's normal sliding-window path inside the script (`EVAL_STRIDE=64`) plus the same legal score-first TTT flow inherited from the base record.
- For a quick local model smoke check on a machine without FlashAttention, importing the module and running a tiny CPU forward pass should now work through the SDPA fallback, assuming PyTorch is installed.

## Validation run for this candidate

Commands attempted locally in this workflow:

```bash
python3 -m compileall candidates/202603261656_training-only-mtp/train_gpt.py
```

Outcome:

- Passed.

Attempted but not fully feasible in this container:

```bash
python3 - <<'PY'
import torch
# import candidate module and run a tiny CPU forward/backward smoke test
PY
```

Outcome:

- Blocked by the local container environment because `torch` is not installed in the available Python runtime, so I could not execute a true model-level smoke test here.
- Because the script now has a non-FlashAttention fallback, that import/forward smoke test should be the first thing to rerun on any machine with PyTorch available.

## Main risks and tradeoffs

- MTP may not pay for itself at this scale if the extra output-head compute cuts step count too much.
- A single extra horizon (`t+2`) is intentionally conservative; the best setting may turn out to be `0`, `1`, or `2` extra heads depending on throughput.
- The newly wired MTP heads are training-only, so any gain must come indirectly through better trunk gradients; if the loss weight is too high, they could distract from the main objective.
- The CPU fallback is for smoke testing and portability, not a claim that full competitive training is practical off-GPU.

## Suggested next experiments

1. Measure step-time overhead and validation delta for `MTP_NUM_HEADS in {0,1,2}`.
2. Sweep `MTP_LOSS_WEIGHT` in `{0.05, 0.10, 0.15, 0.20}`.
3. If MTP helps but slows training too much, try keeping the same idea but only on part of training or with fewer evaluated positions per auxiliary head.
4. If MTP is neutral, revisit other training-only ideas from the research pass, especially stronger export-aware QAT or more selective legal TTT updates.
