# Train-only MTP on the strongest pre-TTT stack

## Hypothesis

Enable **multi-token prediction (MTP)** as a training-only auxiliary loss on top of the current strongest pre-TTT stack. The main bet is that a single future-token head can improve sample efficiency during the fixed 10-minute training budget, while the head is stripped from the exported artifact so the final submission size stays effectively unchanged apart from code bytes.

## Why this is promising for this repository

The repo's frontier is now bottlenecked by how much quality each step can buy under a hard wallclock cap. Recent records already converged on a strong systems-and-compression stack, but their logs still show `mtp_num_heads:0`, so this sample-efficiency lever appears untested in the competitive line.

This is also a particularly good fit for Parameter Golf because the extra head only exists during training. If it helps the trunk learn better hidden states, the final quantized model can keep the gain without paying parameter bytes for the auxiliary head at export time.

## Prior records and experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - strongest overall stack; this candidate copies its banked 11-layer LeakyReLU^2 + Parallel Muon core.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - best reusable pre-TTT compression/export recipe: EMA, GPTQ-lite-style clip search, warmdown 3500, late QAT threshold 0.15.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - confirms Partial RoPE + LN scale are worth preserving.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md`
  - earliest strong 11-layer XSA stack; its run command explicitly kept `MTP_NUM_HEADS=0`.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train.log`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log`
  - both show `mtp_num_heads:0`, reinforcing that nonzero MTP remains unexplored in the recent frontier runs.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`)
  - argues that predicting multiple future tokens improves sample efficiency and representation quality, which is exactly the regime this challenge cares about.

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. `MTP_NUM_HEADS` now defaults to `1` instead of `0`.
2. `MTP_LOSS_WEIGHT` now defaults to `0.1` to keep the auxiliary objective present but conservative.
3. The MTP heads are now included in the head-side Adam optimizer so the auxiliary branch actually learns during training.
4. The existing export path that strips `mtp_heads` from the final checkpoint is preserved, so the auxiliary head remains training-only.
5. Added a small `scaled_dot_product_attention` fallback when `flash_attn_interface` is unavailable so the module can be imported and a CPU forward-pass smoke test can run locally. CUDA training behavior is unchanged because the FlashAttention path still takes precedence on GPU.

## How to run or evaluate it

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to compare against the full 2026-03-23 evaluation stack after training, you can also enable its legal score-first TTT overlay with `TTT_ENABLED=1`.

## Main expected risks or tradeoffs

- The extra vocabulary projection can slow training enough to erase the sample-efficiency gain.
- One auxiliary head may be too weak to matter, while more heads may be too expensive.
- The MTP objective may improve pre-quant trunk quality but not survive quantization/export as well as hoped.
- The current defaults are intentionally conservative; a follow-up sweep over `MTP_LOSS_WEIGHT` or `MTP_NUM_HEADS` may still be needed.

## Validation

- `python -m compileall train_gpt.py`
  - **Passed** on this workflow runner.
- Attempted CPU import + forward-pass smoke test using the SDPA fallback
  - **Blocked by missing local dependencies on the workflow runner**: `torch`, `numpy`, and `sentencepiece` are not installed in the default Python environment here.
  - Once the repo environment is provisioned with `pip install -r requirements.txt`, the fallback path should make a local CPU import/forward check feasible even without FlashAttention.
