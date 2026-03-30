# LeakyReLU^2 + LSQ-lite Late QAT on the GPTQ-lite 11L Stack

## Hypothesis

The strongest non-TTT branch in this repository already combines 11 layers, XSA, Partial RoPE, LN scaling, EMA, and GPTQ-lite export, but its QAT story is still weak: one top record explicitly notes that Late QAT never actually activated under `torch.compile`, and the best GPTQ-lite record still relies primarily on post-training clipping search.

This candidate tests a tighter train/export loop:

1. keep the strong 11-layer EMA + GPTQ-lite + XSA + Partial RoPE base,
2. apply the repo-proven `LeakyReLU(0.5)^2` activation swap,
3. replace fixed late STE-style quantization with **LSQ-lite learned per-row clip multipliers** on every `CastedLinear`,
4. initialize those clip multipliers with a lightweight per-row MSE search when Late QAT turns on, and
5. recompile the training graph once at Late QAT activation so the quantization path is actually live under `torch.compile`.

The expectation is not a giant architectural jump, but a smaller quantization gap on an already strong stack.

## Why this is promising for this repository

This repo's best results repeatedly come from compression-aware training rather than from raw architectural novelty alone:

- sliding-window evaluation was a huge free gain,
- EMA and SWA improved quantization robustness,
- GPTQ-lite clip search improved export quality,
- weight decay and wider MLPs paid off when they made low-bit export easier.

That makes a **more faithful low-bit training signal** a high-leverage next step. The strongest direct repo evidence is that Late QAT was already attempted on a top record, but its README says the branch was dead-code-eliminated by `torch.compile`, so the intended idea was never really tested.

## Records and prior candidates that influenced this candidate

There were **no prior `candidates/` directories** in the repository when this candidate was created.

The main record influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the implementation base because it is the strongest non-TTT GPTQ-lite branch,
  - contributes the 11L architecture, EMA, GPTQ-lite export, XSA, Partial RoPE, LN scaling, VE, and the general training recipe.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - important negative evidence: its README states that Late QAT never activated because `torch.compile` constant-folded the class flag.

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributes the `LeakyReLU(0.5)^2` activation change, which that record reports as a meaningful standalone gain.

## External research that informed it

The core research thread is **learned quantizer parameters**, adapted to this repo's existing weight-only low-bit workflow.

- **LSQ** (`https://arxiv.org/abs/1902.08153`)
  - motivates learning quantizer step sizes instead of treating them as fixed heuristics.

- **LSQ+** (`https://arxiv.org/abs/2004.09576`)
  - motivates better quantizer initialization and learnable scale/offset-style quantizer parameters.
  - I borrowed the spirit of its initialization argument by adding a cheap per-row MSE search when Late QAT activates.

- **PACT** (`https://arxiv.org/abs/1805.06085`)
  - reinforces the value of training-time learned clipping parameters instead of hard-coded clipping heuristics.
  - I did **not** implement full activation quantization here because that would be a broader infrastructure change than this candidate needs.

- **GPTQ** (`https://arxiv.org/abs/2210.17323`)
  - supports the repo's existing post-training quantization direction and motivates keeping GPTQ-lite export, but improving the training-side prior.

- **Primer** (`https://arxiv.org/abs/2109.08668`)
  - relevant background for why squared-ReLU-style activation tweaks can be worthwhile in language models; this candidate uses the repo-proven `LeakyReLU^2` variant rather than adding a new attention primitive.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **LeakyReLU^2 MLP**
   - changed the MLP from `relu(x)^2` to `leaky_relu(x, 0.5)^2`.

2. **LSQ-lite learned per-row clip multipliers**
   - each `CastedLinear` now owns a `qat_log_scale` parameter (one value per output row),
   - during Late QAT, these values learn multiplicative clip factors for the per-row int6 scale.

3. **Per-row MSE initialization at QAT activation**
   - when Late QAT turns on, each linear layer runs a lightweight per-row search over a few clip multipliers and initializes `qat_log_scale` from the best reconstruction error.
   - this is intentionally cheap and meant to approximate a small LSQ+/GPTQ-lite bridge.

4. **Real Late QAT under `torch.compile`**
   - instead of only toggling a class flag and hoping the compiled graph notices, the training loop now recompiles once when Late QAT activates.
   - this makes the quantized forward path explicit rather than depending on a stale compiled graph.

5. **Export keeps GPTQ-lite, but now considers the learned QAT clip prior**
   - GPTQ-lite percentile search is still used,
   - but export also evaluates a candidate scale derived from the learned `qat_log_scale` values and keeps it if it lowers reconstruction MSE.

6. **Candidate defaults tuned for the new flow**
   - `LATE_QAT_THRESHOLD` now defaults to `0.20`,
   - `QAT_MSE_INIT=1`, `QAT_CLIP_MIN=0.50`, and `QAT_CLIP_MAX=1.00` were added.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603302030_leaky-lsq-gptq

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_WD=0.04 ADAM_WD=0.04 \
WARMDOWN_ITERS=3500 LATE_QAT_THRESHOLD=0.20 QAT_MSE_INIT=1 \
QAT_CLIP_MIN=0.50 QAT_CLIP_MAX=1.00 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate now resolves its default dataset and tokenizer paths from the repository root, so it can be launched directly from inside the candidate directory. If you want to force QAT on from step 0 for debugging, set `QAT_ENABLED=1`.

## Validation run for this candidate

I ran the following lightweight validation in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603302030_leaky-lsq-gptq/train_gpt.py
```

Outcome:

- `train_gpt.py` compiled successfully
- `train_gpt_mlx.py` compiled successfully
- `data/` Python files compiled successfully
- `candidates/202603302030_leaky-lsq-gptq/train_gpt.py` compiled successfully

I also attempted a minimal CPU smoke test by importing the candidate module with a tiny flash-attention stub and instantiating a toy `GPT`, but the environment does not have `torch` installed for the available `python` interpreter, so that smoke test could not run here:

```text
ModuleNotFoundError: No module named 'torch'
```

## Main expected risks and tradeoffs

- **Mid-training recompile cost**: recompiling once when Late QAT activates may cost some wallclock and therefore some training steps.
- **More quantizer hyperparameters**: learned clip ranges and activation timing introduce extra knobs that may need a small sweep.
- **Potentially modest gain**: this is a quantization-gap candidate, not a radical architecture change; the likely upside is incremental rather than transformational.
- **Export/train mismatch still exists**: export still uses GPTQ-lite search, not a fully learned quantizer end to end. This candidate narrows that gap instead of eliminating it.
- **LeakyReLU^2 may interact with quantization differently than ReLU^2**: it is promising from repo evidence, but the best threshold for Late QAT may shift because the weight distribution changes.

## Suggested next experiments if this helps

1. Sweep `LATE_QAT_THRESHOLD` across `0.12`, `0.15`, `0.20`, `0.25`.
2. Try a tighter learned clip floor such as `QAT_CLIP_MIN=0.625`.
3. Port the same LSQ-lite logic onto the parameter-banked Parallel-Muon stack from the later TTT record.
4. Combine this candidate with the stronger document-isolated evaluation / TTT pipeline if the pre-quantized weights look healthier.
