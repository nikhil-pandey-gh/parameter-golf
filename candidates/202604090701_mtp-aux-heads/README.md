# MTP Auxiliary Heads on the 11L EMA + GPTQ-lite Stack

## Hypothesis

Enable lightweight multi-token prediction (MTP) auxiliary heads during training so the model gets denser future-token supervision inside the same 600-second budget, while keeping the final artifact unchanged because the auxiliary heads are dropped before export.

## Why this is promising for this repository

The strongest records already look saturated on the usual late-stage levers: sliding-window eval, int6/int8 compression, EMA/SWA, XSA, partial RoPE, LN scale, SmearGate, and BigramHash. What they do **not** report is an actual MTP run, even though the stronger 11-layer scripts already ship an MTP code path with `MTP_NUM_HEADS=0` by default.

That makes MTP attractive here for three reasons:

1. It targets **sample efficiency**, which is the core bottleneck under a hard 10-minute wallclock.
2. It is **training-only** in this implementation: `mtp_heads` are explicitly excluded from the exported state dict, so the artifact budget is unaffected.
3. It is a **minimal fork** of an already strong stack instead of a broad new architecture.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the direct base. It is a strong pre-TTT stack with EMA, GPTQ-lite, XSA, partial RoPE, LN scale, SmearGate, BigramHash, and shared value embeddings.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` showed that the 11-layer XSA/EMA stack is already competitive before extra evaluation tricks.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` showed that the frontier is now being moved by small, targeted gains on top of a strong base rather than wholesale redesigns.
- The broader record history shows quantization- and eval-aware ideas are already crowded, which makes a training-signal improvement more interesting than another tiny PTQ/QAT tweak.

## External research that informed it

- Fabian Gloeckle et al., **Better & Faster Large Language Models via Multi-token Prediction** — https://arxiv.org/abs/2404.19737
- Guoliang Zhao et al., **Self-Distillation for Multi-Token Prediction** — https://arxiv.org/abs/2603.23911
- Raghavv Goel et al., **Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing** — https://arxiv.org/abs/2603.17942
- Lorenzo Noci et al., **Thinking into the Future: Latent Lookahead Training for Transformers** — https://arxiv.org/abs/2603.20219

The common thread is that extra future-token supervision or lookahead can improve sample efficiency or latent future modeling without requiring a larger final deployed model.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

- changed the default `MTP_NUM_HEADS` from `0` to `2`
- changed the default `MTP_LOSS_WEIGHT` from `0.2` to `0.15`
- added one short comment near the defaults clarifying that MTP heads are training-only and excluded from export

Everything else is intentionally left alone so this candidate isolates the MTP hypothesis instead of mixing multiple new ideas at once.

## How to run

From the repository root:

```bash
RUN_ID=mtp_aux_heads \
torchrun --standalone --nproc_per_node=8 candidates/202604090701_mtp-aux-heads/train_gpt.py
```

Equivalent explicit MTP knobs:

```bash
RUN_ID=mtp_aux_heads \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
torchrun --standalone --nproc_per_node=8 candidates/202604090701_mtp-aux-heads/train_gpt.py
```

To disable the candidate idea and recover the base behavior:

```bash
MTP_NUM_HEADS=0 MTP_LOSS_WEIGHT=0.0 \
torchrun --standalone --nproc_per_node=8 candidates/202604090701_mtp-aux-heads/train_gpt.py
```

## Main risks and tradeoffs

- The extra auxiliary heads add training-time projection and loss work, so they may reduce step count enough to offset the sample-efficiency gain.
- MTP wins are well-supported in recent literature, but most published evidence is on larger models than this challenge regime.
- Because the auxiliary heads are discarded before export, the gain must transfer into the shared trunk rather than survive directly in the final logits.
- Two heads and a `0.15` loss weight are plausible but unverified defaults; `1` head or a smaller weight may be a better fit if throughput loss is noticeable.

## Validation

- `python -m compileall candidates/202604090701_mtp-aux-heads/train_gpt.py` — **succeeded**
- Import-time CPU smoke test — **not feasible in this runner**. The environment is missing the required training dependencies (`torch`, `flash_attn_interface`, `sentencepiece`, and `numpy` were all absent), so the script cannot be imported or started here without extra infrastructure.
