# Hybrid RoPE/NoPE tail on the 11-layer EMA + GPTQ-lite stack

## Hypothesis

Keep the current winning `11L / 512d / XSA / EMA / GPTQ-lite` recipe, but stop applying RoPE in the final `XSA` tail. The hypothesis is that lower and middle layers should keep cheap local positional bias via partial RoPE, while the last few globalizing layers should behave more like `NoPE` attention and integrate information with less positional anchoring.

A second low-risk carry-over from the latest SOTA is `LeakyReLU(0.5)^2` in the MLP, which improves gradient flow at essentially zero artifact cost.

## Why this looks promising in this repository

The record history in `records/` shows a clear pattern:

- fixed partial RoPE helps materially versus full RoPE,
- deep-tail attention changes (`XSA` on the last few layers) keep showing up in top runs,
- heavy architectural changes that increase compute per step, especially recurrence, are risky under the 10-minute budget,
- the current best pre-TTT stack is already close to the artifact limit, so parameter-free or near-parameter-free changes are especially attractive.

This candidate stays inside that sweet spot: it is effectively a positional-encoding redistribution, not a wholesale architecture rewrite.

## Prior repository work that influenced this candidate

Primary base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Most relevant repo signals:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` showed that fixed partial RoPE (`16/64`) is clearly better than full RoPE on this stack.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` showed that `LeakyReLU(0.5)^2` is a strong cheap activation upgrade.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` reported that layer recurrence was a negative result under fixed wall-clock budget, reinforcing the decision to prefer cheap positional changes over deeper compute.

## External research that informed it

- `RoFormer: Enhanced Transformer with Rotary Position Embedding` (`arXiv:2104.09864`) established the RoPE mechanism and its relative-position benefits.
- `The Impact of Positional Encoding on Length Generalization in Transformers` (`arXiv:2305.19466`) found that explicit position encodings are not always optimal and that `NoPE` can outperform standard schemes on some length-generalization settings.
- `Length Generalization of Causal Transformers without Position Encoding` (`arXiv:2404.12224`) argued that `NoPE` can be competitive when attention temperatures are tuned appropriately.
- `Rope to Nope and Back Again: A New Hybrid Attention Strategy` (`arXiv:2501.18795`) explicitly motivates hybrid attention that mixes RoPE-style and NoPE-style behavior instead of committing to only one regime.

This candidate does **not** attempt to reproduce the full hybrid architecture from that paper. Instead, it adapts the core idea to this repository with the smallest plausible code change: keep partial RoPE in earlier layers and disable RoPE in the top `XSA` tail.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

- added `NOPE_LAST_N` support and defaulted it to `4`, so the final four layers skip RoPE entirely,
- kept partial RoPE (`ROPE_DIMS=16`) in the earlier layers,
- switched the MLP from `ReLU^2` to `LeakyReLU(0.5)^2`,
- added a small non-CUDA attention fallback plus a `CPU_SMOKE_TEST=1` mode so the script can be smoke-tested locally without a GPU,
- disabled late-QAT **by default** for this candidate (`LATE_QAT_THRESHOLD=0.0` is recommended) so the positional experiment is easier to interpret.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202604011312_hybrid-rope-nope-tail

NOPE_LAST_N=4 ROPE_DIMS=16 MLP_NEGATIVE_SLOPE=0.5 LATE_QAT_THRESHOLD=0.0 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

A lightweight local smoke test is also available:

```bash
cd candidates/202604011312_hybrid-rope-nope-tail
CPU_SMOKE_TEST=1 python train_gpt.py
```

## Validation

Commands and outcomes recorded during this workflow:

- `python -m compileall candidates/202604011312_hybrid-rope-nope-tail/train_gpt.py`
  - Outcome: succeeded.
- `cd candidates/202604011312_hybrid-rope-nope-tail && CPU_SMOKE_TEST=1 /tmp/gh-aw/agent/pg-venv/bin/python train_gpt.py`
  - Outcome: succeeded with `cpu_smoke:ok loss:6.9386 logits_shape:(2, 16, 1024)`.

## Main risks or tradeoffs

- Disabling RoPE in the final layers may weaken short-range positional discrimination if the lower stack does not preserve enough order information.
- Because `XSA` already changes the tail attention geometry, the `NoPE` tail could interact negatively with that mechanism instead of complementing it.
- The CPU smoke path only checks initialization and one forward pass; it is not a substitute for a real multi-GPU training run.
