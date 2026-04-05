# AWQ-lite activation-aware GPTQ clip search

## Hypothesis

The current 11-layer training-only stack is already strong enough that the next cheap gain is more likely to come from **better post-training quantization** than from another architectural rewrite. This candidate keeps the `2026-03-22` stable stack intact and replaces its weight-only GPTQ-lite clip search with an **activation-aware** variant that scores each per-row int6 clip candidate using input-channel energy collected from a short calibration pass.

In short: if the export path already searches over clip percentiles, it should score those candidates with a signal that better matches downstream damage than plain weight MSE.

## Why this is promising here

Three patterns in this repository point at the same bottleneck:

1. `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` argues that quantization loss is larger than many training-side wins.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` improved the best training-only stack with a better clip search, showing the export path still has headroom.
3. The later overall-best run `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` still relies on the same basic low-bit export story, so a cleaner quantizer is plausibly reusable across stronger stacks.

That makes an AWQ-inspired calibration pass a good fit for this repo: it is cheap, local to `train_gpt.py`, and directly attacks a repeatedly observed failure mode.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- **Compression-first motivation:** `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md`
- **Best overall stack to eventually re-stack on top of:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`

## External research that informed it

- **AWQ**: activation statistics identify salient weight channels better than weight-only heuristics, and activation-aware protection improves weight-only PTQ quality without extra inference cost.  
  Paper: [Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- **SignRound / AutoRound**: low-cost post-training optimization of clipping and rounding can outperform naive weight-only rounding with a short calibration phase.  
  Paper: [SignRound: a combined PTQ/QAT-style rounding-and-clipping optimization](https://arxiv.org/abs/2309.05516)
- **Quantization scaling laws**: the best bit/size trade-off is hard to improve with pure weight-only heuristics, which strengthens the case for smarter calibration rather than simply changing the nominal bit width.  
  Paper: [The case for 4-bit precision / bit-level scaling trade-offs](https://arxiv.org/abs/2212.09720)

This candidate deliberately implements the lightest repo-friendly version of that literature: **activation-aware clip scoring**, not full channel rescaling or learned rotations.

## What changed vs the chosen base

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate adds:

1. **A short AWQ-lite calibration pass after EMA and before export.**
2. **Forward-pre hooks on large quantized `CastedLinear` layers** to accumulate per-input-channel second moments from a few training batches.
3. **Activation-aware int6 clip selection**: each row still searches the same GPTQ-lite clip percentiles, but candidates are scored with activation-weighted reconstruction error instead of plain weight MSE.
4. **Master-only export/calibration**, so the extra calibration work is not redundantly repeated on every rank.
5. **Cleanup of the misleading legacy `int8+zlib` size log line** in the copied export block.

The architecture, optimizer stack, EMA/SWA behavior, RoPE/XSA/bigram/value-embedding structure, and artifact format are otherwise left unchanged.

## How to run

From this candidate directory, assuming the repository dataset/tokenizer defaults are available:

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The new export knobs default on, but can be overridden:

```bash
AWQ_LITE_ENABLED=1 \
AWQ_CALIB_STEPS=4 \
AWQ_CALIB_BATCH_TOKENS=131072 \
AWQ_CALIB_SEQ_LEN=1024 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To ablate back to the old behavior:

```bash
AWQ_LITE_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- This is **AWQ-inspired**, not a full AWQ implementation with equivalent channel rescaling, so the gain may be smaller than the paper-level result.
- The calibration metric uses **input-channel energy as a proxy** for output damage; that is cheaper than layer reconstruction but also cruder.
- Export now pays a small extra calibration cost after training.
- The master-only export path assumes the same shared working directory semantics already used by the rest of the distributed script.
- If the existing GPTQ-lite percentile search is already close to optimal on this stack, the measured win may be tiny.

## Validation

Commands run in this environment:

1. `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604051819_awq-lite-gptq/train_gpt.py`  
   **Outcome:** passed.
2. Minimal runtime smoke check for the candidate script  
   **Outcome:** not feasible here because the container does not have PyTorch installed, and this script also requires CUDA plus `flash_attn_interface` at runtime.

