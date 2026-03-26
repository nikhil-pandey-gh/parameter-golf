# MTP + LeakyReLU² on the GPTQ-lite EMA stack

## Hypothesis

The best clean train-only stack in this repo already has strong compression-aware modeling and export, but it still trains with a plain next-token loss. My hypothesis is that adding a small number of **export-free multi-token prediction (MTP) heads** will improve sample efficiency during the 10-minute budget, while **LeakyReLU(0.5)^2** gives the same MLP-side gain that helped the current top record.

This combination targets a gap in the current records: the strongest published train-only runs are around the `1.123x` range, while the best overall score uses extra evaluation-time TTT. If MTP improves the pre-export trunk without adding artifact bytes, it could narrow that gap in a cleaner way.

## Why this is promising for this repository

- Repo history says the biggest durable wins came from train-time quality improvements that survive quantization: 11-layer depth, 3x MLP, XSA, partial RoPE, LN scaling, EMA, and GPTQ-lite.
- The latest record code already contains MTP hooks and explicitly excludes `mtp_heads` from export, so this idea fits the artifact budget unusually well.
- The top record shows **LeakyReLU(0.5)^2** is worth about `-0.002` to `-0.003` BPB on a nearby architecture, and it is trivial to port.

## Influential prior records

- Base implementation: [`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`](../../records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md)
- LeakyReLU² activation source: [`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`](../../records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md)
- Earlier ingredients already carried by the chosen base:
  - XSA + EMA: [`2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`](../../records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md)
  - Partial RoPE + LN scale: [`2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`](../../records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md)

No prior `candidates/` directory existed in this checkout, so this candidate does not overlap with earlier candidate iterations.

## External research that informed this candidate

- **Multi-token prediction**: Fabian Gloeckle et al., ["Better & Faster Large Language Models via Multi-token Prediction"](https://arxiv.org/abs/2404.19737), 2024. This paper argues that auxiliary future-token heads improve sample efficiency by training a shared trunk to predict several offsets at once.
- **Medusa**: Tianle Cai et al., ["Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774), 2024. While mainly an inference paper, it reinforces the practical value of parallel future-token heads on top of a shared backbone.

## What changed versus the chosen base implementation

Starting from the `2026-03-22` GPTQ-lite + EMA record script, this candidate makes three focused changes:

1. **Turns on MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
   - The candidate still excludes `mtp_heads` from the exported artifact, so the auxiliary heads are train-time only.

2. **Ports LeakyReLU(0.5)^2 into the MLP**
   - Replaces `relu(x)^2` with `leaky_relu(x, 0.5)^2`
   - This is copied from the March 23 record idea, but used here without TTT and without the heavier parameter-banking stack.

3. **Adds a FlashAttention fallback for local validation**
   - If `flash_attn_interface` is unavailable or the tensors are not on CUDA, attention falls back to `torch.nn.functional.scaled_dot_product_attention`.
   - This does not change intended GPU behavior, but it makes dependency-light smoke validation possible in environments that do not have FlashAttention installed.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603260037_mtp-leaky-gptqlite
RUN_ID=mtp_leaky_candidate \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Remove MTP and keep the rest of the candidate stack
MTP_NUM_HEADS=0 MTP_LOSS_WEIGHT=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a clean activation ablation, compare against the March 22 base record script and port only the MTP-default change without the LeakyReLU² line. The script remains self-contained and uses the same dataset/tokenizer environment variables as the record it is based on.

## Validation

Commands run in this workflow:

```bash
# Baseline lightweight repo syntax check
python -m compileall train_gpt.py train_gpt_mlx.py data

# Candidate syntax check
python -m compileall candidates/202603260037_mtp-leaky-gptqlite/train_gpt.py
```

Outcomes:

- `python -m compileall train_gpt.py train_gpt_mlx.py data` — passed
- `python -m compileall candidates/202603260037_mtp-leaky-gptqlite/train_gpt.py` — passed

Attempted CPU smoke validation:

- I attempted a tiny import + forward/backward smoke test for `GPT(...)`, but this runner's system Python does not have `torch` installed, even though `torch` is listed in the repository `requirements.txt`.
- Because the workflow environment lacked the required runtime dependency, a meaningful CPU smoke run was **not feasible here**.

## Main expected risks and tradeoffs

- **Step-time regression**: MTP adds extra logits/loss work every step. If the added compute reduces total training steps too much, it may erase the sample-efficiency win.
- **Objective mismatch near convergence**: the auxiliary heads are not exported, so too much MTP weight could improve trunk features early while slightly hurting final next-token specialization.
- **Quantization interaction is uncertain**: the heads are excluded from export, but MTP may still change the trunk's weight distribution in ways that help or hurt GPTQ-lite.
- **Activation transfer risk**: LeakyReLU² helped the top record, but this code path uses a simpler optimizer/export stack than the parameter-banked March 23 submission.
