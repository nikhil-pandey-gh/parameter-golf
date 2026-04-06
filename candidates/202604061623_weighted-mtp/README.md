# Weighted MTP on the 11L EMA + GPTQ-lite stack

## Hypothesis

The current 11-layer record stack is already strong on architecture, export, and evaluation, so the best next gain is likely **sample efficiency** rather than another large structural change. Multi-token prediction (MTP) should help the shared trunk learn richer next-token features in the same 600-second budget, and a **geometric horizon weighting plus warm ramp** should make that auxiliary objective safer for a tiny model than a flat, always-on MTP loss.

## Why this is promising for this repository

- The winning record trend is a stable 11L/512d/8H/4KV stack with additive improvements layered on top: XSA, EMA, Partial RoPE, GPTQ-lite, and then LeakyReLU^2 + legal TTT.
- Repository review found that **MTP support already exists in the strongest train scripts, but the reviewed winners leave `MTP_NUM_HEADS=0`**, so this is an obvious untested gap on the best available base.
- External research reports that MTP improves sample efficiency with little extra training-time overhead when implemented as lightweight future-token heads over a shared trunk.

## Prior experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` — chosen implementation base: 11L stack, EMA, GPTQ-lite, Partial RoPE, XSA4, BigramHash, VE, and export path.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` — confirms Partial RoPE + LN scale were additive wins on the same family.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` — shows the core 11L + XSA + EMA stack is the right base family.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` — evidence that objective/eval changes can still unlock gains even when the core 11L recipe is already strong.
- No prior `candidates/` directory existed when this candidate was created.

## External research

- Fabian Gloeckle et al., **Better & Faster Large Language Models via Multi-token Prediction**, arXiv:2404.19737. Main takeaway used here: predicting multiple future tokens with lightweight auxiliary heads can improve sample efficiency without changing the inference trunk.
- LSQ/QDrop-style quantization-aware ideas were also reviewed (`arXiv:1902.08153`, `arXiv:2203.05740`), but the repository already invests heavily in quantization/export tricks, while MTP is more orthogonal to the current record path.

## What changed vs the chosen base implementation

Base: `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate keeps the entire March 22 model/export stack and only changes the MTP path:

1. **Turns MTP on by default** with `MTP_NUM_HEADS=4`.
2. Adds **horizon-decayed MTP weighting** via `MTP_LOSS_DECAY=0.5`, so nearer future tokens matter more than farther ones.
3. Adds **MTP warm ramp** via `MTP_RAMP_STEPS=750`, gradually increasing the auxiliary loss during early training.
4. Threads a runtime `mtp_scale` into training so the auxiliary loss schedule is explicit instead of hard-coded.
5. Keeps the existing export rule that **drops `mtp_heads` from the serialized artifact**, so the added heads only affect training.

## How to run / evaluate

From this candidate directory:

```bash
RUN_ID=weighted_mtp \
SEED=1337 \
MTP_NUM_HEADS=4 \
MTP_LOSS_WEIGHT=0.2 \
MTP_LOSS_DECAY=0.5 \
MTP_RAMP_STEPS=750 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script keeps the March 22 defaults for the rest of the 11L stack, including EMA, GPTQ-lite export, Partial RoPE, XSA4, BigramHash, and VE.

## Validation

- `python -m compileall candidates/202604061623_weighted-mtp/train_gpt.py` — passed.
- Minimal CPU-only smoke test — **not feasible with the current record base**. This script hard-requires CUDA and imports `flash_attn_interface` directly, so there is no existing CPU path to exercise without introducing extra infrastructure.

## Main risks / tradeoffs

- Extra MTP heads add training-time matmuls and optimizer state even though they are excluded from export.
- The best tiny-model setting may be fewer heads (`1-2`) or a different decay/ramp schedule; these defaults are informed but not yet swept.
- MTP can improve pre-quant training efficiency without necessarily improving the final int6 roundtrip gap, so the main metric to watch is post-export `val_bpb`.
