# Annealed MTP on the 11L EMA + GPTQ-lite stack

## Hypothesis

A single training-only multi-token prediction (MTP) head should improve sample efficiency for this repository's strongest non-TTT stack because it adds extra supervision during the fixed 600s training budget **without adding any exported artifact bytes**. Annealing that auxiliary loss away during warmdown should keep the early-training benefit while reducing mismatch with the final next-token + post-training quantization objective.

## Why this is promising here

- The strongest recent records already cluster around the same 11-layer recipe: XSA on the deepest layers, partial RoPE, EMA/SWA, and smarter post-training quantization.
- Repo review showed that **depth recurrence** was already tried and looked net-negative under the 10-minute wallclock cap, so the next candidate should prefer better supervision over more reused depth.
- The current best non-TTT script already contains MTP infrastructure, but records keep it disabled and exclude MTP heads from export. That makes MTP unusually attractive for Parameter Golf: it spends **training compute instead of artifact bytes**.

## Prior experiments that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best pre-TTT stack in the repo: 11L, XSA-last-4, partial RoPE, LN scale, VE, EMA, GPTQ-lite int6 export.
- **Related frontier stack:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Confirms the frontier is now decided by small, composable improvements rather than broad architectural rewrites.
- **Negative evidence used to reject other ideas:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
  - Explicitly reports depth recurrence as promising in theory but too step-hungry for the 10-minute budget.
- **Late-QAT caution:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - Notes that one late-QAT path was accidentally compiled away, so this candidate does **not** rely on late-QAT by default.

There was no pre-existing `candidates/` directory in the repository when this candidate was prepared.

## External research

- **Better & Faster Large Language Models via Multi-token Prediction** (Gloeckle et al., 2024)  
  https://arxiv.org/abs/2404.19737  
  The core idea is to train a shared trunk with independent heads that predict multiple future tokens, improving sample efficiency without changing the inference-time trunk.

## What changed vs the chosen base

1. **Enabled one MTP head by default** (`MTP_NUM_HEADS=1`), which predicts the token two steps ahead.
2. **Set the auxiliary loss weight to 0.15** (`MTP_LOSS_WEIGHT=0.15`) to keep the extra task meaningful but not dominant.
3. **Added warmdown-aware annealing** (`MTP_DECAY_THRESHOLD=0.25`): while the LR multiplier is above 0.25, MTP stays at full weight; below that, it decays linearly to 0 by the end of training.
4. **Disabled late QAT by default** (`LATE_QAT_THRESHOLD=0.0`) so this candidate does not depend on the compile-prone late-QAT path.
5. **Added a FlashAttention fallback to PyTorch SDPA** for importability and local CPU smoke attempts without changing the CUDA fast path when FlashAttention is available.
6. **Made data/tokenizer defaults relative to the repository root**, so `train_gpt.py` can be launched directly from this candidate directory.

## How to run

From the candidate directory:

```bash
cd candidates/202604020512_annealed-mtp
RUN_ID=annealed_mtp \
SEED=1337 \
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.15 \
MTP_DECAY_THRESHOLD=0.25 \
LATE_QAT_THRESHOLD=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If your dataset or tokenizer live somewhere else, override `DATA_PATH` and `TOKENIZER_PATH`.

Evaluation uses the same built-in export path as the base script: EMA weights are applied, MTP heads are dropped from the exported state dict, the trunk is quantized to mixed int6, and the script prints `final_int6_roundtrip` plus sliding-window metrics.

## Expected risks / tradeoffs

- **Training overhead:** even one MTP head adds extra logits and cross-entropy work, so the model may take fewer optimization steps within 600s.
- **Transfer risk:** better pre-quant next-token features may not translate into better post-quant bpb.
- **Schedule risk:** the annealing threshold is a reasoned heuristic, not a tuned optimum.

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604020512_annealed-mtp/train_gpt.py` | Passed |
| `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604020512_annealed-mtp/train_gpt.py` | Passed |
| CPU smoke via direct module import + random `GPT(...)` forward | Not feasible in this runner: both `python` and `python3` lacked PyTorch (`importlib.util.find_spec("torch") -> None`) |

