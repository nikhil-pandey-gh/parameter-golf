# Paired MLP Sharing on the 11L Static Backbone

## Hypothesis

The strongest non-TTT stack in this repo already squeezes a lot out of quantization, EMA, XSA, partial RoPE, and sliding evaluation. A promising next lever is to **share only the expensive lower-layer MLP matrices across adjacent blocks** while keeping attention, norms, residual mixing, skip weights, and deep-layer specialization untied. That should preserve most of the 11-layer compute path, improve regularization, and reclaim artifact bytes that are otherwise spent on repeated early MLP weights.

## Why this is promising here

- The 2026-03-22 record is the strongest fully static backbone in the repo (`1.1233` mean, `1.1228` best seed) and is the cleanest base for a new training-time idea.
- The 2026-03-23 record shows **LeakyReLU(0.5)^2** is a strong low-risk static gain even before TTT.
- Earlier repo history repeatedly shows that **bigger MLPs / richer token-pair features** matter, but the artifact budget is tight.
- The non-record 1x5090 sweep found that naive `layer recurrence x2` was bad because it burned too much wallclock on extra depth. This candidate avoids that failure mode by **keeping the same logical depth and reusing weights only**, not adding extra unrolled compute.
- Shared lower MLPs also become an export opportunity: if the trainable weights are truly shared, the artifact should not pay for serializing them multiple times.

## Prior repo experiments that influenced this candidate

- **Chosen base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest static 11-layer backbone with GPTQ-lite, EMA, XSA4, partial RoPE/LN scale carry-over, VE, and BigramHash.
- **Activation influence:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - ports the repo's best recent static MLP activation tweak, LeakyReLU(0.5)^2.
- **Token-pair feature influence:** `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - bigger bigram tables helped there, so this candidate raises the default BigramHash table from `2048` to `3072`.
- **Negative-result guardrail:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - its failed recurrence run is exactly why this candidate uses fixed-compute sharing instead of doubling depth applications.

## External research that informed it

- **ALBERT** ([arXiv:1909.11942](https://arxiv.org/abs/1909.11942)) showed that cross-layer parameter sharing can reduce transformer parameter counts substantially while retaining strong performance.
- **Universal Transformer** ([arXiv:1807.03819](https://arxiv.org/abs/1807.03819)) argued for recurrent/shared-depth inductive bias instead of paying full price for every stacked block.
- **Intra-Layer Recurrence in Transformers for Language Modeling** ([arXiv:2505.01855](https://arxiv.org/abs/2505.01855)) found that recurrence targeted at **earlier layers** is especially promising, which motivated sharing the lower MLPs first.
- **Parameter Reduction Improves Vision Transformers** ([arXiv:2512.01059](https://arxiv.org/abs/2512.01059)) reported that **GroupedMLP** sharing between adjacent blocks maintained compute while improving stability, which is the closest direct inspiration for the paired-MLP design here.
- **A Survey on Transformer Compression** ([arXiv:2402.05964](https://arxiv.org/abs/2402.05964)) reinforced that efficient architecture design and quantization should be treated jointly rather than as separate stages.

## What changed vs the chosen base implementation

1. **Paired lower-layer MLP sharing**
   - Layers `0-1`, `2-3`, and `4-5` now share their MLP weights by default (`SHARED_MLP_GROUP_SIZE=2`, `SHARED_MLP_UNTIL=6`).
   - Attention modules, RMSNorms, layer scales, residual mixing, skip weights, and upper-layer MLPs remain layer-specific.
2. **LeakyReLU(0.5)^2 MLP**
   - Ports the static activation gain from the 2026-03-23 record.
3. **Bigger default BigramHash table**
   - Default `BIGRAM_VOCAB_SIZE` is `3072` instead of `2048`.
4. **Alias-aware int6 export with architecture checks**
   - Shared tensors are detected during mixed int6 export and serialized once.
   - The quantized artifact now also stores the sharing/layout metadata needed to fail fast if someone tries to reload it with a non-matching MLP-sharing topology.
5. **Local robustness tweak**
   - `flash_attn_interface` is now optional; local CPU smoke runs fall back to `torch.nn.functional.scaled_dot_product_attention`.

## How to run

From this directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable MLP sharing
SHARED_MLP_UNTIL=0 SHARED_MLP_GROUP_SIZE=1 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Recover relu^2 instead of leaky-relu^2
MLP_NEGATIVE_SLOPE=0.0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script keeps the same evaluation flow as the base record and prints final roundtrip + sliding-window validation metrics from the quantized artifact.

## Main risks / tradeoffs

- Shared lower MLPs may over-regularize the model if those layers still need more layer-specific lexical transformations.
- The candidate deliberately spends its novelty budget on sharing + export deduplication rather than on a larger architectural reallocation, so the first run may underuse some recovered bytes.
- The optional SDPA fallback is only for local smokeability; the intended fast path remains the CUDA + FlashAttention-style training/eval path.

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604091158_paired-mlp-sharing/train_gpt.py train_gpt.py train_gpt_mlx.py data` | Succeeded. |
| `python3 -m venv /tmp/gh-aw/agent/pgolf-venv && . /tmp/gh-aw/agent/pgolf-venv/bin/activate && python -m pip install --quiet numpy sentencepiece torch` | Succeeded in a temporary venv used only for smoke validation. |
| CPU smoke: import the candidate module from file, instantiate a tiny `GPT`, run forward/backward, then run `mixed_quantize_int6` + `dequantize_mixed_int6` | Succeeded with output `{'loss': 3.483412742614746, 'alias_count': 4, 'shared_groups': ((0, 1), (2, 3)), 'mismatch_guard': 'ok'}`. |

What I did **not** validate here:

- A full training run, because this workflow runner is not the target 8xH100 CUDA environment and the candidate script intentionally hard-requires CUDA for the actual training path.
