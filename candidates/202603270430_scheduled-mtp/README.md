# Scheduled MTP on the 03-22 GPTQ-lite 11L base

## Hypothesis

A small amount of **train-time-only multi-token prediction (MTP)** should improve sample efficiency on this challenge without consuming artifact budget at evaluation time.

This repository already contains a strong 11-layer training-only stack with EMA, GPTQ-lite int6 export, XSA, partial RoPE, LN scaling, VE, SmearGate, and BigramHash. It also already had dormant `mtp_heads` support in the later record scripts, but the public record READMEs do not show a run that actually leans on that path. The key bet here is that **future-token supervision is worth a little extra train FLOPs** when the heads themselves are stripped from export.

## Why this is promising for this repository

This challenge is bottlenecked by both the 10-minute train budget and the 16 MB artifact budget.

MTP is attractive here because it pays almost entirely in **training compute**, not artifact bytes:

- the auxiliary heads are used only during training,
- `train_gpt.py` already excludes `mtp_heads` from the exported state dict,
- the candidate keeps the main 11-layer inference graph unchanged,
- the new scheduling keeps the auxiliary objective strongest when learning rates are highest and tapers it during warmdown, while keeping a small nonzero floor so the auxiliary heads never run at zero weight.

That makes MTP a better fit here than broad architectural rewrites or recurrent depth reuse, both of which previous records and non-record notes suggest are harder to make pay off inside the 10-minute window.

## Prior records that influenced this candidate

There were **no prior `candidates/` directories** in this repository when this candidate was created.

The main repository influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best archived **training-only** base,
  - already contains the dormant MTP path and export exclusion,
  - already stacks the current strongest non-TTT architectural ideas.

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - shows there is still headroom beyond the 03-22 base,
  - but does so with much more evaluation complexity (TTT + banked optimizer path).
  - This candidate instead tries to recover some of that headroom from the training objective side while keeping evaluation simple.

- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md`
  - explicitly documents `MTP_NUM_HEADS=0`, which reinforced that this path exists in the codebase but was not yet a public focus.

## External research that informed it

Two recent papers motivated the direction:

- **MTP-D: Self-Distillation for Multi-Token Prediction** (`arXiv:2603.23911`)
  - <https://arxiv.org/abs/2603.23911>
  - argues that MTP can improve efficiency with *minimal additional training cost* if the auxiliary heads are trained carefully and main-head quality is preserved.

- **Thinking into the Future: Latent Lookahead Training for Transformers** (`arXiv:2603.20219`)
  - <https://arxiv.org/abs/2603.20219>
  - reinforces the broader idea that extra future-looking supervision can help autoregressive transformers make better token predictions when compute is invested before commitment.

This candidate uses the lightest-weight version that already fits the repo: plain auxiliary MTP heads, no distillation machinery, no extra inference-time path, and no export-time cost.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS` default changed from `0` to `2`.

2. **Add horizon decay for auxiliary heads**
   - new `MTP_HORIZON_DECAY` env var, default `0.5`.
   - head `k` is weighted by `MTP_HORIZON_DECAY ** k`, so the 2-step target matters less than the 1-step target.

3. **Schedule MTP weight with the LR scale**
   - the MTP loss weight is now updated every step as `MTP_LOSS_WEIGHT * (0.1 + 0.9 * lr_scale)`.
   - this makes the auxiliary objective strongest early/mid training and taper during warmdown without ever reaching a zero-signal compute path.

4. **Use a tensor buffer for the scheduled weight**
   - the dynamic MTP weight lives in a non-persistent buffer instead of a Python float.
   - this is intentional so the schedule can be updated safely without relying on behavior that `torch.compile` might constant-fold.

5. **Keep export behavior artifact-safe**
   - the raw `final_model.pt` checkpoint keeps `mtp_heads` so it matches the candidate's default training config,
   - the final compressed inference artifact still excludes `mtp_heads`, so the scored artifact reflects the main model only.

6. **Log the live auxiliary weight**
   - training logs now report `mtp_w:` so runs can be compared against the baseline overhead more easily.

## How to run / evaluate

From the repository root:

```bash
cd candidates/202603270430_scheduled-mtp
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate keeps the 03-22 defaults for the main architecture and training recipe, while making these MTP settings active by default:

```bash
MTP_NUM_HEADS=2
MTP_LOSS_WEIGHT=0.2
MTP_HORIZON_DECAY=0.5
```

Useful override example for a more conservative sweep:

```bash
cd candidates/202603270430_scheduled-mtp
SEED=1337 \
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.1 \
MTP_HORIZON_DECAY=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Evaluation/export behavior is mostly unchanged from the 03-22 base:

- EMA-applied checkpoint for final evaluation,
- GPTQ-lite int6 mixed quantization,
- zstd-22 if available, otherwise zlib,
- regular in-training validation still uses the base script's non-overlapping full-val pass,
- the final post-export report still includes sliding-window scoring (`EVAL_STRIDE=64` by default),
- `mtp_heads` excluded from the saved artifact.

## Validation

Commands run for this candidate:

```bash
python -m compileall candidates/202603270430_scheduled-mtp/train_gpt.py
```

Outcome:

- **Passed**.

Attempted extra smoke validation:

- I attempted a CPU-only import/forward smoke with a temporary FlashAttention shim.
- That could not be completed in this runner because the local Python environment does **not** have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
- I therefore did **not** run a real runtime forward pass here.

Why a full local training smoke was not feasible in this environment:

- the script expects PyTorch,
- the normal training/eval path is CUDA/NCCL-oriented,
- and the runner here lacks the required local PyTorch install.

## Main expected risks / tradeoffs

- **Extra train FLOPs:** if the MTP heads cost more throughput than they recover in sample efficiency, the result could regress.
- **Objective mismatch:** too much auxiliary weight late in training can hurt the single-step objective actually used for final scoring. The LR-scaled schedule is meant to reduce that risk.
- **Need for retuning:** the best setting may be `1` head, not `2`, or a lower `MTP_LOSS_WEIGHT` than the default here.
- **No guaranteed artifact gain:** this is a training-quality bet, not a compression trick. If the auxiliary supervision does not help learning, export size will stay similar and score may not improve.

## Suggested next experiments

If this candidate shows even a small positive signal, the next obvious sweeps are:

1. `MTP_NUM_HEADS`: `1` vs `2`
2. `MTP_LOSS_WEIGHT`: `0.05`, `0.1`, `0.2`
3. `MTP_HORIZON_DECAY`: `0.25`, `0.5`, `0.75`
4. combine this with the separately validated **LeakyReLU(0.5)^2** activation from the 03-23 record
5. measure step-time overhead directly against the 03-22 base to determine whether MTP or simpler training-only tweaks have the better ROI
