# Shared-Head MTP on the 11L Parallel-Muon Stack

## Hypothesis

A small **training-only multi-token prediction (MTP)** objective should improve sample efficiency under the repository's fixed 10-minute training budget, while adding **effectively zero artifact bytes** at export time.

The core bet is that this repo is now much more bottlenecked by **training efficiency and quantization-aware optimization** than by missing one more evaluation trick. Recent records have already squeezed a lot out of sliding eval, legal TTT, EMA/SWA, XSA, Partial RoPE, and GPTQ-lite. A low-overhead objective-side gain is one of the few strong remaining orthogonal directions.

## Why this is promising for this repository

The current record progression suggests three durable lessons:

1. The strongest stacks are already highly optimized around the same 11-layer / 512d family, so completely new architectures are risky under the 10-minute wallclock.
2. Quantization and evaluation tricks matter, but the March 22-23 records show the stack is already fairly mature there.
3. The remaining room is likely to come from **better use of the same training tokens**, not just another small export tweak.

That makes MTP attractive here:

- It is **orthogonal** to the existing wins.
- It targets **sample efficiency** directly.
- It can be implemented by adapting the current PyTorch script rather than building new infrastructure.
- With tied embeddings and a 1024-token vocab, the extra training objective can be made very lightweight.

## Prior records and repository evidence that influenced this candidate

This candidate is a direct fork of:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

It also inherits the strongest pre-TTT training stack from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`

Those records established the core recipe this candidate keeps:

- 11 layers at 512d with 8H / 4KV GQA
- MLP3x
- late-layer XSA
- Partial RoPE + LN scaling
- SmearGate + BigramHash + VE128
- EMA / SWA / warmdown-heavy optimization
- int6 export with sliding-window evaluation

Just as important, the repository review pointed away from a few tempting but risky alternatives:

- naive recurrence has repeatedly cost too many steps under the 10-minute budget,
- SwiGLU-style swaps have often lost on wallclock,
- and several recent gains were already evaluation-heavy rather than training-objective-heavy.

## External research that informed it

The main research motivation is **multi-token prediction as a sample-efficiency objective**:

- Fabian Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (2024): <https://arxiv.org/abs/2404.19737>
- Anastasios Gerontopoulos et al., *MuToR* (2025): <https://arxiv.org/abs/2505.10518>

These papers argue that predicting multiple future tokens can improve training efficiency and generative performance without requiring a wholesale architecture rewrite.

I also reviewed recurrence-oriented alternatives:

- Anthony Nguyen and Jimmy Lin, *Intra-Layer Recurrence* (2025): <https://arxiv.org/abs/2505.01855>

That direction is interesting, but it looked less suitable here because the repo's own record history already shows that recurrence-like ideas can burn too much wallclock for this challenge.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps that stack, but makes one targeted change:

### 1. Turn dormant MTP into a real training objective

The March 20-23 record scripts already carry unused `MTP_NUM_HEADS` plumbing, but the records themselves keep it disabled and do not describe it in their READMEs.

This candidate converts that dormant direction into an actual training path by:

- enabling MTP by default in `main()` via `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15`,
- replacing full extra vocab heads with tiny **per-horizon affine adapters** (`mtp_scales`, `mtp_biases`),
- reusing the main tied output projection for auxiliary horizons,
- routing the MTP adapter parameters into the scalar AdamW optimizer,
- and stripping all `mtp_*` tensors before final export so the artifact stays training-objective-free.

In other words, the export path stays clean, but training gets an additional objective.

### 2. Add a FlashAttention fallback for local CPU smoke testing

The base record assumes `flash_attn_interface` is available. This candidate adds a fallback to `torch.nn.functional.scaled_dot_product_attention` when FlashAttention is unavailable.

That does not change the intended GPU path, but it makes the script easier to import and smoke-test in a more ordinary environment.

## Why this differs from the existing records and candidates

There were no prior `candidates/` directories in the repo at the time of this workflow run.

This idea is also **not a repeat** of an existing record:

- no reviewed record README reported running MTP,
- the existing record run commands kept `MTP_NUM_HEADS=0`,
- and this candidate's specific twist is a **shared-head, training-only MTP implementation** with export stripping, not just flipping an old flag.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202604011201_shared-head-mtp
RUN_ID=shared_head_mtp_seed1337 \
SEED=1337 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
TTT_ENABLED=1 \
TTT_FREEZE_BLOCKS=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- `TTT_ENABLED=1` keeps the evaluation path aligned with the strongest current record style.
- If you want to isolate the effect of MTP on the training stack alone, rerun with `TTT_ENABLED=0`.
- If you want a pure ablation against the copied base, compare `MTP_NUM_HEADS=0` versus `MTP_NUM_HEADS=2` while keeping the rest fixed.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202604011201_shared-head-mtp/train_gpt.py
```

Outcome:

- Passed.

Attempted smoke check:

```bash
python - <<'PY'
import importlib.util
print(importlib.util.find_spec('torch'))
PY
```

Outcome:

- The runner does **not** have `torch` installed (`None`), so a CPU forward-pass smoke test was **not feasible in this environment**.
- The candidate still adds a non-FlashAttention fallback specifically so that a small CPU import/forward smoke test should work in a normal repo environment where `torch` is installed.

## Main expected risks and tradeoffs

- The auxiliary loss could hurt final next-token calibration if `MTP_LOSS_WEIGHT` is too high.
- Even lightweight MTP still adds some training compute, so the gain must come from better token efficiency rather than raw throughput.
- The tiny affine adapters may be too weak compared with full independent heads, though that is also what keeps the export cost effectively zero.
- This workflow validated syntax only; it did not run a full GPU training/eval cycle.

## Suggested next experiments

1. Compare `MTP_NUM_HEADS=0`, `1`, `2`, and `3` on the same seed and wallclock.
2. Sweep `MTP_LOSS_WEIGHT` in the `0.05` to `0.20` range.
3. Test whether MTP helps more on the pre-TTT score than the post-TTT score.
4. If the shared-head version helps, try a slightly stronger horizon adapter before moving to full extra vocab heads.
