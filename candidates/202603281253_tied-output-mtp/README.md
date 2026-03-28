# 202603281253 Tied-Output MTP

## Hypothesis

A small, training-only multi-token prediction (MTP) auxiliary loss should improve sample efficiency on this repository's fixed 600-second budget without increasing the exported artifact size. The main idea is to make the trunk predict short-horizon futures (`t+2`, `t+3`) during training, then drop the auxiliary modules before export so inference and submission size remain on the standard single-token path.

## Why this is promising for this repository

Recent winning records in this repo mostly stack byte-efficient architectural tweaks, better quantization, and evaluation-time improvements onto the same strong 11-layer backbone. The repo review showed two important things:

- depth recurrence and looping layers were already tried and were reported as net-negative under the 10-minute wall-clock cap,
- training-only auxiliary heads were not part of the documented record lineage, even though they are a natural fit for a fixed-time, fixed-artifact challenge.

That makes MTP attractive here: it targets sample efficiency directly, it does not spend artifact bytes if the heads are dropped at export, and it composes cleanly with the already-proven EMA + GPTQ-lite + partial-RoPE stack.

## Prior records that influenced this candidate

This candidate is based primarily on `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which is the cleanest strong pre-TTT implementation in the repo and already includes the current mature stack:

- 11 layers, 512 width, GQA, 3x MLP,
- XSA on the last 4 layers,
- partial RoPE + LN scale,
- SmearGate + BigramHash,
- shared value embeddings,
- EMA + warmdown tuning,
- GPTQ-lite int6 export.

It also borrows the proven `LeakyReLU(0.5)^2` activation change from `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`, but intentionally does **not** pull in the heavier legal-TTT and parameter-banking machinery.

There were no prior `candidates/` experiments in this repository when this directory was created.

## External research that informed the choice

The implementation is grounded in the following primary sources:

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737)). This paper argues that predicting multiple future tokens improves sample efficiency while leaving the shared trunk useful for the original next-token objective.
- Kwangjun Ahn et al., **"Efficient Joint Prediction of Multiple Future Tokens"** ([arXiv:2503.21801](https://arxiv.org/abs/2503.21801)). This motivates keeping the auxiliary objective lightweight and focused on enriching the hidden state rather than building a large separate prediction stack.
- Anastasios Gerontopoulos et al., **"Multi-Token Prediction Needs Registers"** ([arXiv:2505.10518](https://arxiv.org/abs/2505.10518)). While this candidate does not add register tokens, it uses the same core lesson: MTP works best when the added machinery is lightweight and does not distort the base next-token interface too aggressively.

## What changed versus the chosen base implementation

Compared with the 2026-03-22 record, this candidate makes three targeted changes:

1. **Turn on MTP by default.**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
   - the model predicts `t+2` and `t+3` during training in addition to the normal next-token loss.

2. **Use tied-output residual MTP blocks instead of full vocab auxiliary heads.**
   - each auxiliary branch is a zero-initialized `model_dim -> model_dim` residual block,
   - the transformed hidden state is re-normalized and projected through the same tied output path as the base model,
   - this keeps the training-only branch smaller and better aligned with the compact model's token interface.

3. **Use `LeakyReLU(0.5)^2` in the MLP.**
   - this preserves the repo's successful squared-activation bias while inheriting the March 23 record's better negative-gradient flow.

The export path explicitly strips the training-only `mtp_blocks` parameters before serialization and quantization.

## Running from the candidate directory

The script was adjusted so its default dataset and tokenizer paths resolve relative to the repository root, which means it can be launched directly from this candidate directory.

Example 8xH100 run:

```bash
cd candidates/202603281253_tied-output-mtp
RUN_ID=tied_output_mtp \
SEED=1337 \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable the new idea entirely
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Keep MTP but reduce its strength
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Validation run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603281253_tied-output-mtp/train_gpt.py
```

Outcome:

- passed successfully.

Attempted CPU-side import / constructor smoke test:

```bash
python - <<'PY'
# import candidate module and instantiate a tiny GPT on CPU
PY
```

Outcome:

- blocked in this runner before model init because the repository Python dependencies are not installed here (`numpy` is missing even though it is listed in `requirements.txt`),
- a full training smoke test is also not feasible in this environment because the script hard-requires CUDA plus the challenge runtime dependencies.

## Main expected risks and tradeoffs

- **Step-time overhead:** even lightweight MTP still adds training compute. If throughput drops too much, the sample-efficiency gain may disappear.
- **Objective mismatch:** better short-horizon future prediction does not automatically translate into lower final BPB after quantization.
- **Late-training usefulness is uncertain:** MTP is most compelling as a representation-learning aid early in training; constant use through the full 600 seconds may or may not be optimal.
- **Interaction with quantization is unproven:** the auxiliary loss may improve the trunk, but its effect on GPTQ-lite robustness and final int6 BPB still needs measurement.

## Suggested next experiments

1. Ablate `MTP_NUM_HEADS` in `{0, 1, 2}` while keeping the rest fixed.
2. Sweep `MTP_LOSS_WEIGHT` in `{0.05, 0.10, 0.15, 0.20}`.
3. If step-time overhead is noticeable, gate MTP to the early/mid phase of training only.
4. If MTP helps pre-quant BPB but hurts post-quant BPB, combine it with a quantization-specific regularizer rather than increasing the MTP weight further.
