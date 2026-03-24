# Candidate: Warm-started MTP on the 11L EMA + GPTQ-lite stack

## Hypothesis

A **training-only multi-token prediction (MTP) head** can improve sample efficiency on this challenge without increasing final artifact size, because the auxiliary head is discarded before export. The extra twist in this candidate is to **warm-start the MTP head from the tied LM head / embedding weights** instead of zero-initializing it, so the auxiliary objective becomes useful earlier in a fixed 600s training budget.

## Why this is promising for this repository

The current repo frontier is already very strong on the usual levers: sliding-window evaluation, seq2048, 11 layers, 3x MLP, XSA, partial RoPE, EMA, GPTQ-lite clip search, and tight quantization/compression. The records are also already **very close to the 16MB limit**, so ideas that improve the trained trunk **without adding exported parameters** are especially attractive.

Repository review also showed two important constraints:

- **Throughput matters a lot** in the 10-minute regime, so full recurrence / heavy architectural detours are risky.
- Several strong record scripts already contain MTP plumbing, but the actual runs kept **`MTP_NUM_HEADS=0`**, so this path is present but still effectively untested on the best architecture line.

## Prior records that influenced this candidate

This candidate is based directly on:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

The main record/candidate line that informed the choice was:

- `2026-03-17_NaiveBaseline` — establishes the original 9L baseline and main constraints.
- `2026-03-19_SlidingWindowEval` — confirms evaluation/context handling is a first-order effect.
- `2026-03-19_MLP3x_QAT_Int6_SlidingWindow` — shows that deeper/wider models plus stronger compression are the right meta-direction.
- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` — establishes the 11L XSA + EMA line.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` — adds partial RoPE and LN scaling.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` — current best local base implementation.

Two negative-result notes also shaped the choice:

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` reports that **layer recurrence was net-negative** in the fixed wallclock setting.
- The 2026-03-21 record README notes that its **late-QAT path was effectively dead-code-eliminated** under `torch.compile`, so I avoided centering this candidate on that bug alone.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-Token Prediction"** (arXiv:2404.19737)
  - URL: <https://arxiv.org/abs/2404.19737>
  - Borrowed insight: a shared trunk with independent future-token heads can improve **sample efficiency** and encourage stronger induction-style behavior.
- DeepSeek-AI, **"DeepSeek-V3 Technical Report"** (arXiv:2412.19437)
  - URL: <https://arxiv.org/abs/2412.19437>
  - Borrowed insight: MTP remains relevant in a modern high-performing LM recipe and is compatible with otherwise standard autoregressive training.

I considered shared-block recurrence and more aggressive quantization-aware training as well, but local repository evidence made those riskier for a single candidate iteration: recurrence already showed a bad wallclock tradeoff here, and the current QAT path needs a more invasive compile/runtime redesign to be meaningfully tested.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes four deliberate changes:

1. **Enable one MTP head by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`
   - This predicts the token two steps ahead while the normal head still predicts the next token.

2. **Warm-start the MTP head from the main LM projection**
   - Instead of leaving the auxiliary head at zero-init, copy the tied embedding / LM-head weights into it during initialization.
   - The goal is to reduce the amount of the 600s budget spent just teaching the auxiliary head a basic vocabulary geometry.

3. **Keep export size unchanged in spirit**
   - The MTP head is still excluded from the saved/exported state dict exactly like the upstream MTP plumbing, so the final artifact should remain governed by the main model only.

4. **Make the candidate importable and smoke-testable without FlashAttention 3**
   - Add a fallback to `torch.nn.functional.scaled_dot_product_attention` when `flash_attn_interface` is unavailable.
   - This does not change the intended Hopper path, but it makes lightweight CPU validation feasible.

I also adjusted the default dataset/tokenizer paths to resolve relative to the repository root, so the script can be launched from **inside this candidate directory** as requested.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202603241211_warmstarted-mtp
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable the new idea entirely
MTP_NUM_HEADS=0 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Keep MTP but reduce its influence
MTP_LOSS_WEIGHT=0.10 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If your dataset/tokenizer are stored elsewhere, override:

```bash
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks / tradeoffs

- The extra auxiliary logits add training compute, so if the sample-efficiency gain is too small, the lost steps could dominate.
- The MTP literature is strongest on larger models and code-style generative tasks; the gain on this tiny 11L/512d setup may be smaller.
- Warm-starting from the LM head is a repo-specific optimization rather than a paper-replicated detail, so it could regularize too strongly if `MTP_LOSS_WEIGHT` is too high.
- Because the MTP head is not exported, any improvement must come from better trunk representations rather than memorizing extra parameters.

## Validation

The following lightweight checks were run for this candidate:

- `python -m compileall /home/runner/work/parameter-golf/parameter-golf/train_gpt.py /home/runner/work/parameter-golf/parameter-golf/train_gpt_mlx.py /home/runner/work/parameter-golf/parameter-golf/data`
  - **Passed** on this runner.
- `python -m compileall train_gpt.py`
  - **Passed** on this runner.
- CPU import + forward smoke test using a tiny random GPT instance instantiated from this module
  - **Attempted, but blocked by environment**: the runner's `python` does not currently have `torch` installed (`ModuleNotFoundError: No module named 'torch'`), so a real local forward pass could not be executed here without adding heavyweight new infrastructure.
