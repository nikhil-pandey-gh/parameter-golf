# Shared Encoder MLP + Bigram 3072

## Hypothesis

The current best non-TTT stacks in this repository already exploit most of the obvious training, quantization, and evaluation wins. My hypothesis is that the next cheap gain is to **reuse early encoder MLP weights without adding extra recurrent passes**, then spend a small portion of the saved artifact budget on a larger lexical side-channel (`BigramHash(3072)` instead of `2048`).

This aims to capture some of the benefits of parameter sharing and repeated refinement while avoiding the repo's documented failure mode for explicit depth recurrence under a 10-minute wallclock cap.

## Why this is promising for this repository

Two repository patterns point in the same direction:

- The strongest pure training/export base is the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack, which already combines 11 layers, XSA, partial RoPE, LN scaling, VE sharing, EMA, and GPTQ-lite clipping.
- The repo also documents that **actual layer looping / recurrence** was a bad trade in the 10-minute budget because extra passes reduced the number of optimizer steps too much.

That suggests a narrower move: keep the same number of forward passes, but **share selected block weights statically** so the exported artifact shrinks while throughput stays close to the base stack. This candidate shares the first four encoder MLPs in pairs (`0,1` and `2,3`) and uses part of the recovered bit budget to raise the default bigram hash capacity from `2048` to `3072`.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen base implementation.
  - Strongest pure train/export result in the repo: mature 11-layer stack with GPTQ-lite, EMA, XSA, partial RoPE, LN scale, VE sharing, and `BigramHash(2048)`.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Its README shows that increasing `BigramHash` capacity (`2048 -> 3072`) was still a live, positive ablation on a stronger stack.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - Important negative result: explicit depth recurrence looked promising in theory but was too step-expensive in practice.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Another explicit negative result for layer recurrence under a fixed wallclock budget.

There were **no existing runs under `candidates/`** when this candidate was created.

## External research that informed it

- **ALBERT** (`arXiv:1909.11942`)
  - Classic evidence that cross-layer parameter sharing can preserve quality while cutting parameter count substantially.
- **Universal Transformer** (`arXiv:1807.03819`)
  - Motivates repeated application of a shared transition as a useful inductive bias, especially when compute and parameters need to be decoupled.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (`arXiv:2505.01855`)
  - Especially relevant here: it reports that allocating repeated transformation budget to earlier layers is most effective.
- **Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization** (`arXiv:2603.21676`)
  - Reinforces the idea that shared-depth computation can be powerful, but the repo evidence here argues for a static, throughput-safe approximation rather than true extra recurrent passes.
- **AdaPonderLM** (`arXiv:2603.01914`)
  - Useful as a “not now” reference: adaptive depth is interesting, but too invasive for this repo iteration.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate adds:

1. **Paired encoder MLP sharing**
   - New env var: `SHARED_MLP_GROUPS`
   - Default: `0,1;2,3`
   - Blocks in the same group share one `MLP` module, while keeping their own norms, residual mix, and `mlp_scale`.

2. **Larger default bigram table**
   - `BIGRAM_VOCAB_SIZE` default increased from `2048` to `3072`.

3. **Alias-aware quantized export**
   - Shared tensors are deduplicated before int6/int8 export so weight sharing actually reduces compressed artifact bytes instead of only reducing optimizer parameter count.

4. **Runtime fallbacks for local iteration**
   - FlashAttention import is optional.
   - Standard `scaled_dot_product_attention` fallback is used when FlashAttention is unavailable.
   - `SMOKE_TEST=1` path was added for tiny synthetic forward/backward + quantization roundtrip checks on machines that have `torch` installed.

## How to run

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
SHARED_MLP_GROUPS="0,1;2,3" \
BIGRAM_VOCAB_SIZE=3072 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Local synthetic smoke run, if your environment has `torch` installed:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation run for this candidate

Executed in this workflow:

```bash
python -m compileall candidates/202603270853_shared-encoder-mlp/train_gpt.py
```

Outcome:

- Passed.

CPU smoke status:

- Not executed in this workflow environment because the local Python environment did not have the `torch` package installed, so model execution was not feasible without installing the full ML runtime first.

## Main risks and tradeoffs

- **Too much sharing may underfit early feature extraction.**
  - If this hurts quality, the next ablation should reduce sharing to a single pair, e.g. `SHARED_MLP_GROUPS="0,1"`.
- **The saved bits may be better spent elsewhere than `BigramHash(3072)`.**
  - Natural follow-ups are `VE_LAYERS=8,9,10`, a slightly larger `BIGRAM_DIM`, or a different sharing pattern such as `1,2` only.
- **Alias-aware export adds code complexity.**
  - It is necessary here because otherwise shared modules can still be serialized redundantly by key during custom quantized export.
