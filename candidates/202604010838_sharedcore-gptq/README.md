# Shared-Core GPTQ-lite + LeakyReLU^2

## Hypothesis

The best current non-TTT stacks in this repo are already compute-limited rather than obviously data-limited: they want the 11-layer / 512d / MLP3x recipe, but they also spend most of the artifact budget on distinct block weights. This candidate tests whether we can keep the same **11-layer virtual depth and training-time compute**, while sharing the expensive attention/MLP cores across depth and preserving layer identity with cheap per-layer adapters.

The specific bet is:

- share only the large block matrices across depth,
- keep per-layer `q_gain`, residual mixing, layer scales, skip weights, and a learned depth bias untied,
- keep the strong `2026-03-22` EMA + GPTQ-lite export path,
- add the `LeakyReLU(0.5)^2` activation from the current top record,
- reinvest a small portion of the saved bytes into a larger `BigramHash` default (`3072`).

## Why this is promising for this repository

Repository history suggests two important things:

1. The best leaderboard progress came from stacking cheap wins on top of the stable 11-layer int6 recipe: XSA, Partial RoPE, LN scale, EMA, and GPTQ-lite.

2. Earlier "depth recurrence" attempts were likely hurt by **extra effective depth / extra step cost** under the 10-minute cap, not by the general idea of parameter reuse itself. In particular, `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` explicitly calls looping layers promising but too slow under the wallclock budget.

This candidate addresses that failure mode directly: it does **not** add more virtual layers than the current 11-layer recipe. It keeps the same forward depth, but swaps 11 distinct block cores for 3 shared ones plus tiny per-layer adapters.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the base implementation because it is the cleanest strong non-TTT stack: 11L, XSA4, Partial RoPE, LN scale, VE, EMA, GPTQ-lite, warmdown 3500.

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the `LeakyReLU(0.5)^2` activation change, which the README reports as a meaningful improvement on top of the strong PR #414 stack.

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - important negative result: "depth recurrence" was promising but too expensive when it increased effective depth. This candidate is explicitly designed to avoid that exact failure mode.

- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/`
  - notable because it contained disabled looped-architecture code paths, reinforcing that the repo has considered recurrence before but did not land a strong fixed-budget implementation.

## External research that informed it

- **ALBERT** — parameter sharing can preserve quality while substantially reducing model size and memory footprint: <https://arxiv.org/abs/1909.11942>

- **Universal Transformer** — recurrent/shared-depth computation can add a useful inductive bias without abandoning Transformer-style parallel sequence processing: <https://arxiv.org/abs/1807.03819>

These papers motivated the central design choice here: keep Transformer-style sequence parallelism and fixed-depth compute, but reuse block parameters across depth.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Shared block cores**
   - replaces 11 distinct transformer blocks with `NUM_SHARED_BLOCKS=3` shared cores reused across the 11-layer schedule.
   - each virtual layer keeps its own lightweight adapter state:
     - `q_gain`
     - `attn_scale`
     - `mlp_scale`
     - `resid_mix`
     - `depth_bias`
     - optional DTG gate

2. **Layer identity preserved**
   - each virtual layer gets a learned `depth_bias`, so the shared core is not forced to behave identically at every depth.
   - XSA remains assigned by virtual layer index, so only the deepest virtual layers use it.

3. **LeakyReLU^2 MLP**
   - swaps ReLU^2 for `LeakyReLU(0.5)^2`, following the current top record's activation choice.

4. **Larger default BigramHash**
   - default `BIGRAM_VOCAB_SIZE` increased from `2048` to `3072`.

5. **Candidate-directory usability**
   - default dataset/tokenizer paths now resolve relative to the repository root, so the script can be launched from inside this candidate directory as requested.

6. **CPU-safe attention fallback for smoke imports**
   - if `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA for import-time / non-CUDA use. The main training path still expects CUDA.

7. **Self-describing exports**
   - exported checkpoints now save the architectural config alongside weights so quantized reload/eval does not depend on ambient env vars like `NUM_SHARED_BLOCKS`.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202604010838_sharedcore-gptq

RUN_ID=sharedcore_gptq \
NUM_SHARED_BLOCKS=3 \
MLP_LEAKY_SLOPE=0.5 \
BIGRAM_VOCAB_SIZE=3072 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
NUM_SHARED_BLOCKS=4 torchrun --standalone --nproc_per_node=8 train_gpt.py
NUM_SHARED_BLOCKS=5 torchrun --standalone --nproc_per_node=8 train_gpt.py
BIGRAM_VOCAB_SIZE=2048 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation run in this workflow

Validated successfully:

- `python -m py_compile candidates/202604010838_sharedcore-gptq/train_gpt.py`
- `python -m compileall candidates/202604010838_sharedcore-gptq/train_gpt.py`

Import-based CPU smoke test was **not feasible in this runner** because the environment did not have the repo's Python runtime dependencies installed (`torch`, `numpy`, and `sentencepiece` were missing). The attempted smoke test failed immediately at import time for that reason, before exercising candidate logic.

## Main expected risks or tradeoffs

- **Underfitting from too much sharing**: 3 shared cores may be too aggressive for an 11-layer tiny LM, even with layer-specific adapters.

- **Optimization mismatch**: Muon / EMA / GPTQ-lite were tuned on fully untied blocks; shared cores may want different LR or warmdown behavior.

- **Artifact headroom vs model capacity**: this first candidate only partially reinvests saved bytes. If it compresses comfortably, the next follow-up should probably trade some of that headroom back into larger side channels or a slightly richer adapter.

- **Compile sensitivity**: the architecture is still `torch.compile`-friendly in principle, but shared-core reuse can surface different graph-specialization behavior than the original fully untied stack.
