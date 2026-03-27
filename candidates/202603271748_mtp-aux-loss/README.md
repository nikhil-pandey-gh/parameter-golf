# Candidate: MTP Auxiliary Loss on the 11L EMA + GPTQ-lite Stack

## Hypothesis

Enable a small, training-only multi-token prediction (MTP) auxiliary loss on top of the strongest training-only record stack. The extra future-token supervision should improve sample efficiency under the fixed 10-minute training budget, while keeping the exported artifact unchanged because the auxiliary heads are excluded from the final checkpoint and roundtrip eval path.

## Why this looks promising for this repository

Recent record progress in this repo has come from evaluation, quantization, and a few zero-or-near-zero-byte architectural tweaks: sliding eval, EMA, partial RoPE, LN scaling, GPTQ-lite clip search, and LeakyReLU squared. In contrast, the strongest training-only lineage already carries an export-safe MTP codepath, but the published record logs still report `mtp_num_heads:0`, so the idea was present in code yet never actually tried in the submitted runs.

That makes MTP a good next candidate here:

- it targets **training efficiency**, which still matters under the 600s wallclock cap;
- it adds **no final artifact bytes** because the auxiliary heads are stripped before export;
- it fits the current code with a **small, low-risk change** instead of a broad refactor;
- it avoids repeating dead ends like naive layer recurrence, which already looked poor in the repo's single-GPU exploration.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen base implementation.
  - Supplies the mature 11-layer training-only stack: EMA, GPTQ-lite clip search, warmdown 3500, partial RoPE, LN scaling, XSA on late layers, BigramHash, SmearGate, shared value embeddings, and int6 export.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Confirms that partial RoPE + LN scale were real wins in this lineage.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Reinforces that the mature 11-layer stack still has headroom, but most of that record's extra gain came from evaluation-time TTT and a separate activation change rather than a cleaner training-objective tweak.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Helpful negative evidence: simple depth recurrence was not attractive under fixed wallclock, so this candidate favors a training-objective improvement instead of a heavier architectural bet.

There were no prior `candidates/` directories in the repository when this candidate was created.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737, 2024)
  - Argues that predicting multiple future tokens with independent auxiliary heads improves sample efficiency and downstream capability with little training-time downside.
  - Especially relevant here because the repository is bottlenecked by a short training budget and strict artifact cap.

I also considered parameter-sharing ideas such as ALBERT-style cross-layer sharing, but repo evidence against naive recurrence made MTP the lower-risk first experiment.

## What changed vs. the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

- changed the default `MTP_NUM_HEADS` from `0` to `1`;
- changed the default `MTP_LOSS_WEIGHT` from `0.2` to `0.15`;
- changed the default dataset/tokenizer paths to resolve from the repository root instead of the current working directory;
- added short comments clarifying that:
  - MTP heads are **training-only**, and
  - the final export / roundtrip eval path intentionally drops them.

Everything else stays aligned with the base 11L stack.

## How to run

From the repository root:

```bash
cd candidates/202603271748_mtp-aux-loss

SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to override the candidate defaults explicitly:

```bash
cd candidates/202603271748_mtp-aux-loss

MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.15 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This script expects the same dataset and tokenizer layout as the recent record stacks.
Its default `DATA_PATH` and `TOKENIZER_PATH` resolve from the repository root via `__file__`, so running from the candidate directory works without extra path overrides as long as the standard cache layout exists.

## Expected risks and tradeoffs

- **Extra training compute:** even one auxiliary head adds logits/loss work during training, so the sample-efficiency gain has to beat the throughput loss.
- **Loss interference:** future-token prediction may regularize the trunk well, or it may partially compete with the next-token objective.
- **No inference-time benefit by construction:** the candidate keeps MTP heads out of the final artifact, so gains must transfer into the shared trunk during training.
- **Untuned hyperparameters:** `1` head and `0.15` weight are conservative defaults, not the result of a dedicated sweep.

## Validation

Validation was kept lightweight and repository-aligned:

- `python -m compileall candidates/202603271748_mtp-aux-loss/train_gpt.py`
- `python - <<'PY' ... Path('candidates/202603271748_mtp-aux-loss/train_gpt.py').resolve().parents[2] ... PY` to confirm the new default path logic points at the repository root

Outcome:

- Syntax compilation passed.
- The `__file__`-relative default path logic resolved to `/home/runner/work/parameter-golf/parameter-golf`, so the candidate no longer depends on being launched from the repository root.
- A true CPU smoke run was **not feasible** in this environment because this script hard-requires CUDA during `main()`, depends on `flash_attn_interface` at runtime, and expects the challenge dataset/tokenizer layout used by the record scripts.
