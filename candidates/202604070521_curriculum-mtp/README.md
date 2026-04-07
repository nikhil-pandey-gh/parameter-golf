# Curriculum MTP on the LeakyReLU/TTT stack

## Hypothesis

Training-only multi-token prediction (MTP) should improve sample efficiency for this tiny-language-model setting, **if it is introduced gently enough for a small trunk**. This candidate keeps the current best architectural stack and adds a **forward curriculum** for MTP: train on next-token prediction only at the start, then enable horizon-decayed auxiliary future-token heads once the model has stabilized.

The key bet is that this repo's current frontier is already very strong on **compression-aware architecture, quantization, and evaluation-time adaptation**, so the best remaining training-side idea is one that can improve the learned representation **without costing artifact bytes**. The existing code already strips `mtp_heads.*` from the exported state dict, so the auxiliary heads only affect training.

## Why it is promising here

- The records show that this challenge has already harvested many of the obvious byte-saving and eval-time wins: int6 mixed quantization, GPTQ-lite, EMA/SWA, XSA, Partial RoPE, LN scaling, BigramHash, SmearGate, and legal score-first TTT.
- The repo also already contains dormant MTP support, but every record I reviewed still logs `mtp_num_heads:0`, so this is a real gap rather than a repeat.
- Because MTP heads are **training-only** here, this idea can plausibly buy better representation learning under the 10-minute budget **without adding bytes to the submitted artifact**.

## Repository evidence that shaped this candidate

### Main influences

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - Current best local stack (`1.1194` mean post-TTT BPB).
   - Contributed the LeakyReLU(0.5)^2 MLP, legal score-first TTT recipe, and the parameter-banking/Parallel-Muon training path.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - Best non-TTT training/quantization base.
   - Reinforced the 11L / 2048 / int6 / EMA+SWA / GPTQ-lite recipe.
3. `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
   - Shows the importance of deep-layer XSA and the 11L + 3x MLP stack.
4. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
   - Confirms Partial RoPE + LN scale are strong zero-byte gains.

### Trends I followed

- Favor ideas that **help training quality without increasing final artifact size**.
- Preserve the current winning structure: 11 layers, 3x MLP, XSA late layers, Partial RoPE, BigramHash, SmearGate, VE, EMA/SWA, late fake quant.
- Avoid dead ends already reported in records: naive recurrence, heavy compute-slowing activations, and fragile always-on QAT.

### Prior candidates

There was **no existing `candidates/` directory** when this candidate was created, so there were no prior candidate iterations to avoid or extend.

## External research that informed it

1. **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"**  
   arXiv:2404.19737 — <https://arxiv.org/abs/2404.19737>  
   Motivation: shared-trunk, multi-head future-token prediction improves training efficiency and downstream quality.

2. **Ansar Aynetdinov and Alan Akbik, "Pre-Training Curriculum for Multi-Token Prediction in Language Models"**  
   arXiv:2505.22757 — <https://arxiv.org/abs/2505.22757>  
   Motivation: smaller language models benefit more from **forward curriculum** MTP than from turning on a hard MTP objective from step 0.

Those papers pushed this candidate toward a **small-model-friendly MTP variant** rather than a naive always-on auxiliary loss.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enable training-only MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
   - MTP heads remain excluded from export, as in the underlying stack.

2. **Add a forward curriculum**
   - `MTP_START_FRAC=0.2`
   - Before 20% of planned steps: standard next-token training only.
   - After that point: MTP auxiliary loss turns on.

3. **Add horizon decay for auxiliary targets**
   - `MTP_HEAD_DECAY=0.5`
   - Closer future tokens get more weight than farther ones.

4. **Keep the rest of the strong stack intact**
   - LeakyReLU(0.5)^2 MLP
   - XSA on late layers
   - Partial RoPE
   - BigramHash + SmearGate + VE
   - EMA/SWA, late fake quant, int6+lzma export path
   - Optional legal TTT at eval time

5. **Add a local fallback path**
   - If `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA.
   - `CPU_SMOKE_TEST=1` runs a tiny synthetic forward/backward pass without dataset or CUDA requirements.

## How to run or evaluate it

From this directory:

```bash
RUN_ID=curriculum_mtp \
BIGRAM_VOCAB_SIZE=1536 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_HEAD_DECAY=0.5 MTP_START_FRAC=0.2 \
TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a lightweight local start-up check:

```bash
CPU_SMOKE_TEST=1 python train_gpt.py
```

That smoke path uses synthetic tokens, a tiny 2-layer model, and the SDPA fallback if FlashAttention is not installed.

## Validation

The following low-cost checks were run for this candidate:

1. `python -m compileall train_gpt.py`
   - **Passed**
2. `CPU_SMOKE_TEST=1 python train_gpt.py`
   - **Passed**
   - Observed output: `cpu_smoke_ok loss:5.5839 logits_shape:(2, 32, 128) flash_fallback:True`

## Main expected risks and tradeoffs

- **Step-time regression**: even training-only MTP adds extra logits/CE work, so the added objective may reduce total steps seen inside 600 seconds.
- **Small-model sensitivity**: the curriculum is meant to reduce this, but tiny trunks can still regress under MTP.
- **Quantization interaction**: a pre-quant gain may partially wash out after int6 export.
- **TTT interaction**: better pretrained representations may help TTT, but the objectives could also interfere.
- **Fallback path is only for smoke testing**: the intended fast path on challenge hardware is still FlashAttention-backed CUDA execution.
