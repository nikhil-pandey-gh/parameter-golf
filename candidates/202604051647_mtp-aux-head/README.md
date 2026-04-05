# 202604051647_mtp-aux-head

## Hypothesis

Adding a **single train-only multi-token prediction (MTP) head** to the current best 11-layer banked/XSA/EMA/LeakyReLU/TTT stack should improve sample efficiency inside the fixed 10-minute training budget, while preserving the exported artifact size because the auxiliary head is excluded from export.

## Why this is promising here

- The repository's strongest gains after the early architecture jump came from changes that improved **effective use of fixed training/eval compute** without paying many extra artifact bits: sliding-window eval, EMA, partial RoPE, GPTQ-lite, and legal TTT.
- The copied parent stack already contains dormant `MTP_NUM_HEADS` support and already excludes `mtp_heads` from export, so this idea fits the current codebase with almost no infrastructure risk once the auxiliary heads are actually wired into optimization.
- The MTP paper reports that predicting multiple future tokens as an auxiliary objective improves sample efficiency and downstream quality with a shared trunk, which matches this repo's fixed-wallclock setting especially well.

## Prior experiments that influenced this candidate

- **Current best absolute score:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - supplied the base architecture, legal score-first TTT recipe, parameter banking, and LeakyReLU(0.5)^2 activation.
- **Best clean pre-TTT stack:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - reinforced that the 11-layer EMA/XSA/partial-RoPE family is the right parent line.
- **Earlier eval and adaptation records:** `records/track_10min_16mb/2026-03-19_SlidingWindowEval/` and `records/track_10min_16mb/2026-03-17_LoRA_TTT/`
  - showed that evaluation-aware improvements can dominate small architecture tweaks.

## External research that informed it

- **Better & Faster Large Language Models via Multi-token Prediction** (arXiv:2404.19737)  
  <https://arxiv.org/abs/2404.19737>
- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** (arXiv:1909.11942)  
  <https://arxiv.org/abs/1909.11942>
- **MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases** (arXiv:2402.14905)  
  <https://arxiv.org/abs/2402.14905>

MTP was chosen over layer-sharing ideas because it targets the same fixed-wallclock regime, is already wired into this codebase, and does not require broader changes to the banked export/quantization path.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

1. Baked in the parent run defaults that matter for this branch:
   - `BIGRAM_VOCAB_SIZE=1536`
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`
2. Enabled a **single auxiliary MTP head by default**:
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.1`
3. Added the auxiliary `mtp_heads` to the non-bank Adam parameter group so the MTP loss actually updates both the head and, after the first step, the shared trunk.
4. Set a stable default `RUN_ID` for this candidate directory.

Other than making the dormant MTP path truly trainable, the point of this candidate is to isolate the MTP training objective on top of the current best stack, not to mix in unrelated architecture edits.

## How to run

From this directory on the normal multi-GPU challenge setup:

```bash
RUN_ID=202604051647_mtp_aux_head \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Parent behavior without the new hypothesis
MTP_NUM_HEADS=0 TTT_ENABLED=1 BIGRAM_VOCAB_SIZE=1536 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Stronger auxiliary objective sweep
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.2 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

- `python -m compileall candidates/202604051647_mtp-aux-head/train_gpt.py`
- CPU smoke test in a temporary venv (`torch`, `numpy`, `sentencepiece`) with a stubbed `flash_attn_interface` module to instantiate `GPT` and run a tiny forward pass without CUDA

Observed outcomes on this runner:

- `compileall`: **passed**
- CPU smoke test: **passed**
  - instantiated the 11-layer candidate model with `MTP_NUM_HEADS=1`
  - verified the auxiliary head is present in the non-bank Adam parameter group
  - ran `GPT.forward()` on a tiny random batch and returned `loss=7.6440`
  - ran `GPT.forward_logits()` and returned `logits_shape=(1, 16, 1024)`

I did **not** run a real training or FlashAttention-backed smoke test here because this runner does not provide the CUDA/FlashAttention stack that the full script expects for challenge-style execution.

## Main risks and tradeoffs

- The auxiliary head adds training-time compute, so improved sample efficiency has to outweigh the reduced step count under the same 600s cap.
- MTP may help the pre-quant model more than the quantized export, so gains could partly wash out after the int6+lzma roundtrip.
- The best TTT recipe in the parent stack was tuned without MTP, so there may be interactions between the auxiliary objective and the later score-first adaptation stage.
