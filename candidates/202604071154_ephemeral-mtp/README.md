# Ephemeral MTP on the LeakyReLU2 + Legal TTT stack

## Hypothesis

A **training-only** 1-head multi-token prediction (MTP) auxiliary loss should improve sample efficiency on this repository's strongest 11-layer stack without materially hurting the 16 MB artifact budget, because the auxiliary head can be optimized during training and then **excluded from the exported checkpoint** before quantization and evaluation.

## Why this is promising for this repository

The local record history points in two directions:

1. **The current best recipe is already capacity-limited rather than obviously architecture-broken.** Recent wins came from piling small but real improvements onto the same 11-layer, 512-dim core: 3x MLP, BigramHash, SmearGate, XSA, EMA, Partial RoPE, GPTQ-lite, LeakyReLU(0.5)^2, and legal TTT.
2. **The challenge strongly rewards training-time tricks that do not consume final artifact bytes.** The non-record 4-hour baseline showed that post-training quantization becomes a bottleneck quickly; improving training efficiency without making the exported model larger is therefore especially attractive.

This candidate leans into that second point. It keeps the record architecture and export path almost unchanged, but makes the dormant MTP path actually operative.

## Prior repository work that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Quantization/export baseline:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Architecture trend setters:** the 2026-03-20 to 2026-03-23 record stack showing that 11L + 3x MLP + BigramHash/SmearGate + XSA + EMA/weight averaging is the strongest local family.

There were **no prior `candidates/` experiments** in this repository when this folder was created.

## External research that informed it

- **Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"** ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737)) argues that jointly predicting multiple future tokens improves sample efficiency and can strengthen algorithmic structure learning.
- **Mehra et al., "On multi-token prediction for efficient LLM inference"** ([arXiv:2502.09419](https://arxiv.org/abs/2502.09419)) finds that retrofitting MTP onto frozen next-token models is difficult, but **joint training with the backbone** is the right regime when using MTP as an auxiliary objective.

That combination fits this repository well: we train from scratch, keep the backbone intact, and only pay a modest training-time compute cost.

## What changed versus the chosen base implementation

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes the MTP path real instead of leaving it as dormant scaffolding:

1. **MTP enabled by default** with `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.15`.
2. **Auxiliary MTP heads are now optimizer-wired.** In the base script, MTP heads were instantiated and later excluded from export, but they were not added to any optimizer param group, so enabling them would not have trained them.
3. **Auxiliary heads are initialized from the tied embedding / main vocab geometry** so the MTP loss can influence the trunk immediately instead of waiting for zero-initialized heads to learn first.
4. **Export behavior stays artifact-friendly.** MTP heads are still excluded from `final_model.pt` and the quantized round-trip artifact, so the extra training-only head does not directly spend artifact bytes.

Everything else intentionally stays close to the current best stack:

- 11 layers, 512 dim, 8 heads / 4 KV heads
- BigramHash + SmearGate
- XSA on the last 4 layers
- Partial RoPE (16 dims)
- LN scaling
- EMA + tight SWA
- GPTQ-lite style int6 export
- LeakyReLU(0.5)^2 MLP
- legal score-first TTT

## How to run / evaluate

From this candidate directory:

```bash
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To override the auxiliary objective explicitly:

```bash
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.15 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still exports an MTP-free artifact and evaluates the quantized model plus the legal TTT path, just like the chosen base record.

## Main expected risks / tradeoffs

- **Training-time overhead:** even a single auxiliary head adds extra vocab projections and cross-entropies, so improved supervision must outweigh any lost step count inside the 600s cap.
- **Objective mismatch:** the repository is evaluated on next-token compression, not multi-token decoding, so too-large MTP weight could distract from the main objective.
- **Quantized gain may be smaller than pre-quant gain:** better training loss does not always survive post-training quantization.
- **No local GPU smoke test here:** this script still expects the same CUDA/FlashAttention environment as the base record, so only lightweight syntax validation was practical in this workflow environment.

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py
```

Outcome:

- `python -m compileall train_gpt.py` — succeeded

CPU smoke testing was not feasible here because this record-family script hard-requires CUDA and imports the FlashAttention interface used by the original submission path.
