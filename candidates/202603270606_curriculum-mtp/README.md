# Forward-Curriculum MTP on the 11L GPTQ-lite / EMA stack

## Hypothesis

The strongest non-TTT line in this repository already gets most of its win from a carefully compressed 11-layer stack: XSA, partial RoPE, LN scaling, shared value embeddings, EMA, and GPTQ-lite quantization. A good next step is to improve **training efficiency per token** rather than adding more export-time complexity.

This candidate applies a **forward-curriculum multi-token prediction (MTP)** objective to that stack. The model starts as pure next-token prediction, then gradually turns on extra future-token heads during the first part of training. The hypothesis is that this gives the tiny 11L model denser training signal without destabilizing early optimization, while keeping the final artifact size unchanged because the auxiliary heads are stripped before export.

## Why this is promising for this repository

The records show a clear pattern:

- the best pure model here is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which already squeezed most of the easy architecture/compression gains out of the 11-layer stack;
- the best overall model, `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`, demonstrates that more training/eval signal can still move the frontier;
- none of the shipped records actually turned on the dormant `MTP_NUM_HEADS` path, even though the recent record code already contains training-only MTP heads and excludes them from export.

That makes MTP unusually attractive here: it is orthogonal to the current compression stack, it costs **training compute instead of artifact bytes**, and the repository already has most of the plumbing needed to try it cleanly.

## Prior experiments that influenced this candidate

### Main base implementation

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

This is the direct base for `train_gpt.py`. It provides the strongest pure-model stack in the repo:

- 11 layers, 512d, 8H / 4KV
- 3x MLP
- XSA on the last 4 layers
- Partial RoPE (16/64 dims)
- LN scale
- shared value embeddings on late layers
- EMA + tight SWA support
- GPTQ-lite int6 quantization and zstd compression
- sliding-window evaluation

### Other records used as evidence

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - shows the frontier is still moving through objective/eval changes, not just by changing quantization formats.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - establishes that cheap architectural refinements can matter, but also documents the failed late-QAT toggle under `torch.compile`.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/`
  - confirms eval remains stride-64 sliding window.

There were **no existing `candidates/` folders** in the repository when this candidate was created.

## External research that informed it

### Better & Faster Large Language Models via Multi-token Prediction

- Fabian Gloeckle et al., 2024
- arXiv: [2404.19737](https://arxiv.org/abs/2404.19737)

This paper argues that predicting multiple future tokens from a shared trunk improves sample efficiency and helps induction-head-like behavior. That is directly relevant to a 10-minute training budget where each forward/backward pass has to do as much useful work as possible.

### Pre-Training Curriculum for Multi-Token Prediction in Language Models

- Ansar Aynetdinov, Alan Akbik, 2025
- arXiv: [2505.22757](https://arxiv.org/abs/2505.22757)

This is the key paper for this candidate. It specifically notes that **smaller language models struggle with always-on MTP**, and shows that a **forward curriculum** helps SLMs benefit from MTP while avoiding the early-training instability of jumping straight into the harder objective.

### DeepSeek-V3 Technical Report

- DeepSeek-AI et al., 2024
- arXiv: [2412.19437](https://arxiv.org/abs/2412.19437)

DeepSeek-V3 uses a multi-token prediction objective in a production-scale training recipe, which is a strong signal that the objective is worth taking seriously beyond toy settings.

## What changed versus the chosen base

Starting from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes three targeted changes:

1. **Turns on MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`

2. **Adds a forward curriculum for MTP**
   - new knobs:
     - `MTP_CURRICULUM_START=0.10`
     - `MTP_CURRICULUM_END=0.40`
   - the script converts training progress into per-head weights.
   - head 1 ramps on first, then head 2 ramps on later; early training remains pure next-token prediction.
   - the schedule is wallclock-aware when `MAX_WALLCLOCK_SECONDS` is set, matching the rest of this repository’s time-budgeted training logic.

3. **Keeps MTP training-only**
   - the auxiliary `mtp_heads` are still excluded from the exported state dict before quantization.
   - evaluation reconstructs the model with `mtp_num_heads=0`, so the final artifact size stays aligned with the usual 16MB accounting.

Additionally, the default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root so the script is runnable from inside this candidate directory without extra path juggling.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603270606_curriculum-mtp
```

Default candidate run:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults already enable the MTP curriculum. To make the schedule explicit:

```bash
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
MTP_CURRICULUM_START=0.10 \
MTP_CURRICULUM_END=0.40 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Expected behavior:

- training logs will report `mtp_curriculum:start=... end=...`
- train logs will include current `mtp_head_weights:[...]`
- export will log `export_excluding_mtp_params:...`
- final eval still uses the usual int6 roundtrip and sliding-window metrics

## Main expected risks and tradeoffs

- **Small-model fragility**: the 2025 curriculum paper exists because small models can regress under naive always-on MTP. The schedule here is a reasonable first guess, not a tuned optimum.
- **Extra training FLOPs**: the auxiliary heads are not paid for in artifact bytes, but they still add training compute. If the overhead meaningfully reduces step count inside 600s, the gain could wash out.
- **Objective mismatch risk**: future-token heads may improve general modeling, or they may slightly dilute the next-token objective if the coefficient is too large.
- **No empirical sweep yet**: this candidate is meant to test the core idea cleanly before doing a larger grid over head count, coefficient, and curriculum span.

## Validation

Validation run in this workflow:

```bash
python -m compileall train_gpt.py candidates/202603270606_curriculum-mtp/train_gpt.py
```

Result:

- `python -m compileall` passed for both the root baseline `train_gpt.py` and this candidate `train_gpt.py`.

CPU-only smoke test status:

- not feasible in this workflow environment.
- the runner does not currently have `torch`, `numpy`, `sentencepiece`, or `flash_attn_interface` installed, and the repository checkout does not include local `fineweb_train_*.bin`, `fineweb_val_*.bin`, or SentencePiece `.model` artifacts.
- because the script is CUDA-first and data-dependent, any real startup smoke test here would fail for environment reasons rather than candidate logic.
