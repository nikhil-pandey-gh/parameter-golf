# Shared Tail MLP + LeakyReLU^2

## Hypothesis

Full layer recurrence already showed up as a dead end in this repo because extra effective depth cost too many optimizer steps inside the 10-minute wall-clock budget. This candidate keeps the March 22 11-layer core at the same compute depth, but shares only the deepest **three MLPs** so the export artifact pays for one tail MLP instead of three. The saved budget is reinvested into a larger **BigramHash(3072)**, while the MLP activation is upgraded to **LeakyReLU(0.5)^2** from the current best overall record.

## Why this is promising here

- The strongest non-TTT training stack in the repo is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`.
- The current best overall record (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`) reports a real gain from LeakyReLU(0.5)^2.
- Bigger BigramHash already helped in prior records, but the strongest 11-layer stacks stayed tight on bytes.
- Prior recurrence notes in `2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` and `2026-03-18_FP16Embed_WD3600/README.md` argue against looping full layers in a fixed-time regime. This candidate is explicitly a *parameter-sharing* play, not a *more-matmul recurrence* play.

There were no prior `candidates/` directories in the repo when this was added.

## External research that informed it

- **ALBERT** ([arXiv:1909.11942](https://arxiv.org/abs/1909.11942)): cross-layer parameter sharing can preserve model quality while cutting parameter count.
- **Intra-Layer Recurrence in Transformers for Language Modeling** ([arXiv:2505.01855](https://arxiv.org/abs/2505.01855)): selective recurrence/sharing is more promising than indiscriminate whole-block reuse.
- **Thinking Deeper, Not Longer** ([arXiv:2603.21676](https://arxiv.org/abs/2603.21676)): shared-depth transformers need stabilizing structure; this repo already has useful ingredients for that, including LN scaling and zero-initialized projection layers.

The resulting design choice here is conservative: share only the deepest MLP substack, leave attention unique per layer, and avoid increasing effective depth.

## What changed versus the chosen base

Base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Shared tail MLP**: the deepest 3 layers use one shared `shared_tail_mlp` module; attention, norms, residual mixing, and scales stay layer-specific.
2. **LeakyReLU(0.5)^2**: replaces ReLU^2 inside the MLP, carrying over the March 23 activation win.
3. **Bigger BigramHash default**: `BIGRAM_VOCAB_SIZE` default increased from 2048 to 3072.
4. **Shared-parameter-aware export**: the shared MLP weights live once in `state_dict`, so quantization/export preserves the intended artifact savings.
5. **Config plumbing/logging**: added `SHARED_MLP_TAIL` and logging for which layers share the tail MLP.

## Influential prior records

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`

## How to run

From this directory:

```bash
RUN_ID=shared_tail_mlp \
SHARED_MLP_TAIL=3 \
BIGRAM_VOCAB_SIZE=3072 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs to sweep next:

- `SHARED_MLP_TAIL=2,3,4`
- `BIGRAM_VOCAB_SIZE=2048,3072,4096`
- `VE_LAYERS=8,9,10`

Evaluation behavior is inherited from the March 22 base: int6 roundtrip plus sliding-window evaluation, with stride controlled by `EVAL_STRIDE` (default 64).

## Validation

Commands run in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604012246_shared-tail-mlp/train_gpt.py
```

Outcome:

- syntax compilation passed for the root scripts, `data/`, and this candidate script
- a CPU launch smoke test was **not feasible** here because the runtime environment did not have `torch` or `flash_attn_interface`, and this script requires the existing CUDA + FlashAttention path

## Main risks and tradeoffs

- The deepest MLPs may still want more layer-specific specialization than a single shared module allows.
- Sharing only MLPs is safer than sharing full blocks, but it may still lose some of the gain from deeper unique layers.
- This candidate extends the March 22 core, not the March 23 banked + legal-TTT stack. If the idea looks promising on GPU, the next step should be to port it onto the March 23 code path rather than treating this as the final endpoint.
