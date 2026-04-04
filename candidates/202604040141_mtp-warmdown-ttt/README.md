# Warmdown-Faded MTP on the LeakyReLU² + Legal TTT Stack

## Hypothesis

Train-time-only multi-token prediction (MTP) should improve sample efficiency in this 10-minute regime, and fading the auxiliary loss to zero during warmdown should keep the final EMA/export aligned with plain next-token evaluation. Because the extra MTP head is excluded from the exported artifact, this should be able to improve `val_bpb` without spending model bytes.

## Why this is promising for this repository

The repo's strongest trend is to keep the deep-and-thin 11-layer GQA stack and improve *training efficiency* or *evaluation efficiency* rather than attempting broad architectural rewrites. Recent winners stacked XSA, partial RoPE, LN scaling, EMA, longer warmdown, GPTQ-lite quantization, and legal TTT; earlier notes explicitly called out depth recurrence and slower activations like SwiGLU as bad trades under the 600-second wallclock.

MTP fits that pattern unusually well:

- it is a training-only auxiliary objective,
- it adds no exported artifact bytes,
- it targets sample efficiency rather than raw parameter count,
- and the best current stack already carried dormant MTP code, so the change can stay surgical.

## Prior repo records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - current best post-TTT stack (`1.1194` mean),
  - provides the LeakyReLU(0.5)^2 MLP, parameter banking, and legal score-first TTT recipe,
  - but its dormant MTP path was no longer attached to any optimizer after the parameter-banking refactor, so that auxiliary path was effectively inert.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best stable non-TTT stack,
  - reinforced that longer warmdown, EMA, and careful late-phase quantization matter on this architecture.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - established partial RoPE + LN scaling as zero-byte wins on the same 11-layer family.
- Negative evidence from `2026-03-18_FP16Embed_WD3600`, `2026-03-17_LoRA_TTT`, and the non-record recurrence notes helped rule out broader rewrites in favor of a training-only efficiency idea.

## External research that informed it

- Fabian Gloeckle et al., *Better & Faster Large Language Models via Multi-Token Prediction* (arXiv:2404.19737): predicts multiple future tokens with auxiliary heads on a shared trunk and reports better sample efficiency with no inference-time penalty.
- Zechun Liu et al., *MobileLLM* (arXiv:2402.14905): argues that for sub-billion models, architecture and training choices matter disproportionately; its deep-thin, grouped-query-attention framing supports staying on this repo's 11L/512d/GQA backbone instead of widening the model.
- I also reviewed parameter-sharing literature such as ALBERT and Universal Transformer, but did not choose that direction because this repo's existing notes already flag depth recurrence as a weak trade under the 10-minute budget.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default** with one auxiliary future-token head (`MTP_NUM_HEADS=1`, `MTP_LOSS_WEIGHT=0.15`).
2. **Actually train the MTP head** by wiring its weights into the AdamW optimizer; the inherited parameter-banked stack logged MTP knobs but no longer stepped `mtp_heads`, so the path was effectively dead.
3. **Fade MTP through warmdown** with a runtime buffer driven by the same wallclock-aware LR scale, so the auxiliary objective is strongest early and vanishes by the end.
4. **Bake in the top-record defaults** that mattered for the chosen stack, including `BIGRAM_VOCAB_SIZE=1536`, `TTT_ENABLED=1`, and `TTT_FREEZE_BLOCKS=0`.
5. **Make the script runnable from inside the candidate directory** by resolving dataset/tokenizer defaults relative to the repository root.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202604040141_mtp-warmdown-ttt
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
cd candidates/202604040141_mtp-warmdown-ttt
TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

```bash
cd candidates/202604040141_mtp-warmdown-ttt
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script writes logs and artifacts into the candidate directory, and the exported artifact still excludes the auxiliary MTP head.

## Main expected risks or tradeoffs

- Even one extra MTP head increases training compute; if the step-time hit is larger than the sample-efficiency gain, total `val_bpb` could regress.
- The warmdown fade may still be too slow or too aggressive; the best setting could require adjusting `MTP_LOSS_WEIGHT` or disabling fade for part of training.
- The score interaction with legal TTT is uncertain: the gain may show up mostly in the pre-TTT model and only partially survive post-TTT adaptation.

## Validation

- `python -m compileall candidates/202604040141_mtp-warmdown-ttt/train_gpt.py` — succeeded
- Minimal CPU smoke test was **not** feasible without broader rewrites because this script imports `flash_attn_interface` and unconditionally enters the CUDA/NCCL training path.
