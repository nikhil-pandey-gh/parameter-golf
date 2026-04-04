# Shared-Head MTP Fadeout on the 11L EMA + GPTQ-lite Stack

## Hypothesis

Training-only multi-token prediction (MTP) should improve sample efficiency on this repository's strongest no-TTT stack, and fading the auxiliary loss out during warmdown should keep the final model better aligned with the single-step validation objective.

The key repo-specific bet is that this auxiliary supervision is almost free in artifact bytes: the horizon-specific MTP tensors are excluded from export, so the final submission size is still dominated by the base 11-layer model and its mixed int6/int8 checkpoint.

## Why it is promising for this repository

- The challenge is heavily wallclock-constrained, so better supervision density per step matters.
- Recent records already squeezed a lot out of quantization, EMA, XSA, partial RoPE, and optimizer tuning.
- Public record scripts contain dormant MTP support, but the visible recorded configs keep `MTP_NUM_HEADS=0`, so this path is still underexplored here.
- Compared with adding more recurrence or broader architectural changes, train-only MTP is a smaller, lower-risk adaptation of an already-strong stack.

## Prior records and experiments that influenced this candidate

- **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best no-TTT extension point from the repo review;
  - already includes the modern 11L stack: EMA, GPTQ-lite clip search, XSA, BigramHash, partial RoPE, LN scale, and mixed int6/int8 export.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirms partial RoPE + LN scale are durable wins.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - establishes the 11-layer EMA + XSA direction as a strong baseline family.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - shows the current frontier is now dominated by training/eval refinements on top of a strong base, but this candidate intentionally avoids TTT complexity.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - documents a negative naive recurrence result, which pushes this candidate toward auxiliary supervision instead of weight reuse.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737)
  - motivates predicting multiple future tokens from a shared trunk to improve sample efficiency.
- Tianle Cai et al., **"Medusa"** (arXiv:2401.10774)
  - reinforces the usefulness of extra future-token heads without changing the exported backbone itself.
- Zhenzhong Lan et al., **"ALBERT"** (arXiv:1909.11942)
  - was one of the main alternatives considered because factorization / sharing fit tight parameter budgets, but this candidate prefers the lower-risk MTP route first.

## What changed versus the chosen base implementation

Starting from the 2026-03-22 record script, this candidate makes the following focused changes:

1. **Shared-head MTP is enabled by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
   - `MTP_SHARED_HEAD=1`

2. **The main LM head is reused for all MTP horizons**
   - instead of separate full-vocab horizon heads, each future horizon uses the shared output projection;
   - each horizon gets only a tiny training-only affine adapter in hidden space (`scale` + `bias` vectors).

3. **Horizon losses are distance-weighted**
   - the candidate uses harmonic weighting (`1/(k+1)`) so nearer future tokens matter more than farther ones.

4. **The auxiliary objective fades out during warmdown**
   - `MTP_DECAY_START_SCALE=0.35`
   - `MTP_DECAY_END_SCALE=0.05`
   - as the LR multiplier shrinks in warmdown, the MTP loss scales down to zero.

5. **Training-only MTP tensors are excluded from export**
   - all state dict keys containing `mtp_` are dropped before artifact serialization.

6. **Candidate-directory execution works by default**
   - dataset and tokenizer defaults now resolve relative to the repository root via the script path, so the script can be launched from inside this candidate directory.

7. **Late QAT is off by default here**
   - `LATE_QAT_THRESHOLD=0.0`
   - this keeps the candidate focused on the MTP hypothesis instead of relying on the compile-fragile late-QAT path from prior stacks.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202604032355_mtp-fadeout
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful comparison toggles:

```bash
# Disable MTP inside this candidate script
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Approximate the cited 2026-03-22 base more closely
MTP_NUM_HEADS=0 LATE_QAT_THRESHOLD=0.15 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Keep MTP on, but disable the fadeout schedule
MTP_DECAY_START_SCALE=0.0 MTP_DECAY_END_SCALE=0.0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still expects the cached FineWeb dataset and tokenizer described in the repository README. By default it resolves:

- dataset: `<repo>/data/datasets/fineweb10B_sp1024`
- tokenizer: `<repo>/data/tokenizers/fineweb_1024_bpe.model`

## Validation

Commands run in this workspace:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604032355_mtp-fadeout/train_gpt.py
```

Outcome:
- passed successfully.

Attempted smoke check:
- a tiny CPU import/forward smoke was attempted for the candidate GPT path;
- it could not run in this environment because `torch` is not installed (`ModuleNotFoundError: No module named 'torch'`), so only syntax-level validation was feasible here.

## Main expected risks and tradeoffs

- **Step-time risk:** even shared-head MTP adds extra softmax work, so the gain in supervision must outweigh any reduction in total steps.
- **Adapter capacity risk:** vector-scale/vector-bias horizon adapters may be too weak compared with fully separate horizon heads.
- **Objective mismatch risk:** fading the MTP loss late should help, but if the useful signal mostly arrives late, the schedule may turn it off too aggressively.
- **Quantization interaction risk:** this candidate keeps the base export path intact, but MTP may still change the final weight distribution in ways that help or hurt mixed int6/int8 compression.

## Suggested next experiments

1. Sweep `MTP_NUM_HEADS` in `{1, 2, 3}` and `MTP_LOSS_WEIGHT` in `{0.10, 0.15, 0.20}`.
2. Compare `MTP_SHARED_HEAD=1` against the older full-vocab separate-head path.
3. Sweep fadeout thresholds, especially a later shutoff such as `0.25 -> 0.10`.
4. If MTP helps pre-quant quality but not post-quant quality, revisit a properly runtime-controlled QAT path on top of this candidate.
