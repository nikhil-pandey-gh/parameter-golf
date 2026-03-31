# Candidate: MTP1 auxiliary head on the 2026-03-23 LeakyReLU² + Legal TTT stack

## Hypothesis

Enable a **single multi-token-prediction (MTP) auxiliary head** during training on top of the strongest March 23 stack. The extra head should improve sample efficiency and representation quality under the repository's strict 10-minute budget, while leaving the exported artifact essentially unchanged because the code already strips `mtp_heads.*` tensors from the saved submission checkpoint.

## Why this is promising for this repository

- The current record trajectory has already harvested most of the obvious artifact-side gains: sliding-window eval, int6 mixed quantization, GPTQ-lite clipping, EMA/SWA, partial RoPE, and LeakyReLU².
- The March 23 stack already contains a working MTP implementation, but the feature is left off by default with `MTP_NUM_HEADS=0`.
- In this repo, MTP is especially attractive because it is **training-only**: `mtp_heads` are explicitly excluded from the exported state dict before quantization, so the main tradeoff is step-time overhead rather than artifact size.
- That makes it a rare knob that can improve the learning problem without directly spending the scarce 16 MB budget.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md`
  - Establishes the baseline 9L/512d/GQA trainer and export path.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - Confirms the value of partial RoPE and layerwise norm scaling on the 11-layer stack.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - Shows the current non-TTT near-SOTA recipe: EMA, GPTQ-lite clipping, longer warmdown, VE, XSA, partial RoPE.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - Chosen base implementation. It adds LeakyReLU(0.5)^2, legal score-first TTT, and parameter-bank/parallel-Muon systems work.

## External research that informed it

- Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, Gabriel Synnaeve, **"Better & Faster Large Language Models via Multi-token Prediction"**, arXiv:2404.19737, 2024.
  - URL: `https://arxiv.org/abs/2404.19737`
  - Key point for this repo: predicting multiple future tokens with independent heads on a shared trunk improves sample efficiency as an auxiliary training task. That maps well to Parameter Golf's wall-clock-limited regime, where better learning per update matters more than architectural novelty for its own sake.
- Joshua Ainslie et al., **"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"**, arXiv:2305.13245, 2023.
  - URL: `https://arxiv.org/abs/2305.13245`
  - This was the strongest alternative direction from the research sweep: push `NUM_KV_HEADS` from `4` toward `2` or `1` and reinvest the saved bytes. I did not choose it for this candidate because it changes the exported model, the quantization surface, and the speed/quality tradeoff all at once, while MTP is a narrower training-only intervention on the current best stack.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Set `MTP_NUM_HEADS` default from `0` to `1`.
2. Set `MTP_LOSS_WEIGHT` default from `0.2` to `0.1`.
3. Left the rest of the March 23 stack unchanged on purpose.

This keeps the candidate focused on a single hypothesis: **one low-weight auxiliary MTP head may improve training efficiency enough to beat its small compute overhead**.

## How to run or evaluate it

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

That uses the March 23 defaults plus this candidate's MTP defaults (`MTP_NUM_HEADS=1`, `MTP_LOSS_WEIGHT=0.1`). The copied script also rewrites its default `DATA_PATH` and `TOKENIZER_PATH` to resolve from the repository root, so launching from inside this candidate directory still finds `../../data/...` correctly without extra environment variables.

For the full March 23-style evaluation recipe with legal TTT enabled:

```bash
TTT_ENABLED=1 \
TTT_FREEZE_BLOCKS=0 \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 \
TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 \
TTT_GRAD_CLIP=1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to sweep the auxiliary strength, the main knobs are:

```bash
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.05 torchrun --standalone --nproc_per_node=8 train_gpt.py
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.10 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks or tradeoffs

- **Step-time regression**: even one auxiliary head adds projection and softmax work every training step.
- **Interaction risk with TTT**: MTP could improve the pre-TTT trunk while also changing how much gain remains available for score-first adaptation.
- **Quantization mismatch**: the heads are excluded from export, which is desirable for size, but it also means the auxiliary objective only shapes the trunk indirectly.
- **Schedule sensitivity**: the best loss weight may be lower than `0.1` on this already-strong stack.

## Validation

Commands run locally in this workflow:

```bash
python -m compileall candidates/202603302350_mtp1-auxloss/train_gpt.py
```

Outcome:

- Passed syntax compilation.

CPU-only smoke test:

- Not run. This candidate inherits the March 23 record stack, which hard-depends on CUDA, `flash_attn_interface`, and the challenge dataset/tokenizer layout. Adding a fake-data or CPU fallback path would be new infrastructure rather than validating the candidate as implemented.
