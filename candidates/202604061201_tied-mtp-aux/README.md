# Tied-output MTP auxiliary loss on the 03-22 11L stack

## Hypothesis

The strongest clean pre-TTT base in this repository already carries dormant multi-token prediction (MTP) scaffolding, but every shipped record leaves `MTP_NUM_HEADS=0`. Turning on a **small, training-only MTP auxiliary loss** should improve sample efficiency for this step-limited challenge without adding exported model bytes.

This candidate keeps the 03-22 stack intact and adds the lightest version of MTP I could justify:

- **1 future-token auxiliary head**
- **tied to the deployed output projection / tied embedding head**
- **with tiny per-offset affine conditioners so the future-token branch is not forced onto the main logits**
- **excluded from the exported model by construction**

## Why this is promising here

Repository review pointed to a clear pattern:

- the leaderboard moved hardest from better **evaluation**, **quantization**, and the **11-layer 03-22 stack**;
- recent gains on the pre-TTT path are already small;
- the best 03-22-style scripts already contain MTP code, but submitted runs still log `mtp_num_heads:0`.

That makes MTP a good fit for this repo: it targets **training sample efficiency** instead of adding more evaluation cost, and it is unusually cheap to test because the implementation base already exists.

## Prior repository influences

- **Chosen base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best clean pre-TTT stack in the repo;
  - already includes XSA4, Partial RoPE, LN scale, VE, EMA, GPTQ-lite export, and dormant MTP hooks.
- **Also informed by:** `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
  - explicitly documents `MTP_NUM_HEADS=0`, which helps confirm MTP remains unexplored in shipped results.
- **Prior candidates:** none existed in `candidates/` when this candidate was created.

## External research that informed it

1. **Gloeckle et al., _Better and Faster Large Language Models via Multi-Token Prediction_**  
   https://arxiv.org/abs/2404.19737  
   Key takeaway: predicting multiple future tokens from a shared trunk can improve sample efficiency and downstream quality.
2. **Zhao et al., _Self-Distillation for Multi-Token Prediction_**  
   https://arxiv.org/abs/2603.23911  
   Key takeaway: MTP can be strengthened with only modest additional training cost, which is important under a tight wall-clock budget.
3. **Goel et al., _Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing_**  
   https://arxiv.org/abs/2603.17942  
   Key takeaway: even next-token-trained LMs exhibit latent MTP structure, which supports trying a lightweight auxiliary objective here instead of a heavy new inference stack.

## What changed vs. the 03-22 base

1. **Enabled MTP by default** with `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.1`.
2. **Added `MTP_TIED=1` default**, which reuses the deployed output head for auxiliary future-token logits instead of training fully separate export-excluded heads.
3. **Added tiny per-offset affine conditioners** before that shared output head, so each future offset gets its own lightweight adaptation while staying out of the exported artifact.
4. **Kept export behavior compact**: the MTP branch remains training-only and is stripped before quantization/export.
5. **Made the script runnable from the candidate directory** by resolving default dataset/tokenizer paths from the repository root instead of the working directory.
6. **Added a lightweight `SMOKE_TEST=1` path** plus a FlashAttention fallback to PyTorch SDPA so the candidate can be sanity-checked without a GPU-only runtime.

Everything else is intentionally left close to the 03-22 base:

- 11 layers, 512 width, 8 heads / 4 KV heads
- 3x MLP
- XSA on the last 4 layers
- Partial RoPE (16 dims)
- LN scale
- VE on layers 9 and 10
- EMA + GPTQ-lite-style mixed int6 export

## How to run

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
MTP_NUM_HEADS=1 MTP_TIED=1 MTP_LOSS_WEIGHT=0.1 \
RUN_ID=tied_mtp_aux \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

CPU-only smoke:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation run for this candidate

Commands executed during creation:

```bash
/tmp/gh-aw/agent/pg-venv/bin/python -m compileall candidates/202604061201_tied-mtp-aux/train_gpt.py
SMOKE_TEST=1 /tmp/gh-aw/agent/pg-venv/bin/python candidates/202604061201_tied-mtp-aux/train_gpt.py
```

Outcomes:

- `compileall`: success
- smoke test: `smoke_test:ok loss:7.6237 seq_len:32 mtp_num_heads:1 mtp_tied:True`

## Main risks and tradeoffs

- **Step budget risk:** even a 1-head auxiliary loss adds compute, so any sample-efficiency gain has to outweigh fewer total steps in 600s.
- **Head sharing risk:** sharing the output head plus only tiny offset-specific affine conditioners is more artifact-friendly, but may underperform fully independent MTP heads.
- **Quantization interaction risk:** better pre-quant loss does not guarantee better post-quant BPB.
- **No 8xH100 result yet:** this is a targeted candidate implementation, not a verified leaderboard improvement.
