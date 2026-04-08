# 202604081755 Teacher-Forced MTP

## Hypothesis

A **training-only teacher-forced multi-token prediction (MTP)** auxiliary should improve sample efficiency on this repo's strongest 11-layer stack without paying an artifact-size cost. The idea is to make the shared trunk learn slightly longer-horizon structure during the 10-minute train budget, then discard the auxiliary heads before export so the final artifact stays focused on the next-token model.

## Why this looks promising here

- The repo's durable gains have come from **better use of the fixed training/eval budget**: sliding-window eval, deeper 11-layer stacks, better quantization, EMA/SWA, and then a small activation tweak plus legal TTT.
- Several recent record scripts already carried dormant `MTP_*` knobs, but every shipped log I found still reports `mtp_num_heads:0`, so this direction has **not actually been tried** on the record stacks.
- The current best stack already has strong compression, attention, and optimizer choices. That makes **sample-efficiency improvements** more attractive than another broad architecture rewrite.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - Base implementation copied from here.
   - Keeps parameter banking + parallel Muon, LeakyReLU(0.5)^2, XSA, partial RoPE, VE, GPTQ-lite int6 + lzma, and optional legal TTT.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - Confirms the strong pure-training stack before TTT.
3. `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
   - Shows the earlier code path already had dormant MTP hooks, but the published logs still ran with `mtp_num_heads:0`.

## External research that informed it

1. **Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`)
   - Argues that predicting multiple future tokens with multiple heads on a shared trunk improves sample efficiency and especially helps generative / algorithmic behavior.
2. **Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"** (`arXiv:2401.10774`)
   - Reinforces that lightweight extra future-token heads on top of one trunk are practical and useful, even when the main model stays unchanged.
3. **Zhai et al., "Exclusive Self Attention"** (`arXiv:2603.09078`)
   - Not new in this candidate, but it remains part of the carried-forward base stack.

## What changed vs the chosen base implementation

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

1. **Turns MTP on by default** with `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15`.
2. Replaces the dormant independent future heads with **teacher-forced residual MTP heads**:
   - each horizon consumes the previous future token's embedding,
   - adds a tiny residual adapter,
   - predicts the next future token from that adapted state.
3. **Initializes each MTP head from the tied token embedding**, so the auxiliary predictors start close to the main vocabulary projection instead of starting cold.
4. **Actually wires MTP parameters into optimization** on the parameter-banked stack. The copied 2026-03-23 script logged MTP settings but did not add the MTP weights to any optimizer group.
5. Excludes **all** training-only `mtp_*` parameters from export, not just `mtp_heads`, so the auxiliary path stays artifact-free.

## How to run / evaluate

From the repo root:

```bash
cd candidates/202604081755_teacher-forced-mtp
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
# Turn off the auxiliary for ablation
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Match the current score-first TTT style evaluation on the carried-forward stack
TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

- `python -m compileall candidates/202604081755_teacher-forced-mtp/train_gpt.py` — passed
- A CPU startup smoke test was **not feasible** in this environment because the script hard-requires CUDA and FlashAttention kernels at runtime, and the repo does not ship a CPU fallback path for this record-style trainer.

## Main risks / tradeoffs

- Even two auxiliary future heads add extra training-time softmax work, so the gain from better sample efficiency has to beat the loss in steps/second.
- Teacher-forced future conditioning could over-regularize the trunk if `MTP_LOSS_WEIGHT` is too high.
- The copied base stack is already highly optimized and somewhat brittle (`torch.compile`, FlashAttention, parameter banking), so small graph changes may still behave differently on the real 8xH100 path than they do in static compilation checks.
