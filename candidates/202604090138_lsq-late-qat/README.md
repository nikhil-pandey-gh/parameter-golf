# LeakyReLU^2 + true LSQ-style late QAT

## Hypothesis

The clean 11-layer EMA + GPTQ-lite stack from `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is already close to the frontier, but its late-QAT path is still fragile across the repo and at least one closely related record documented that `torch.compile` can constant-fold the QAT flag away entirely. The hypothesis here is that replacing that brittle path with a **real late quantization-aware training phase** using **learned per-row int6 step sizes** will reduce the export-time quantization gap, and that pairing it with the already-proven **LeakyReLU(0.5)^2** activation will improve pre-quant robustness enough to make those learned scales matter.

## Why this is promising here

Repository evidence points to two consistent trends:

1. **Quantization/export quality is still a major bottleneck.** The strongest non-TTT runs are clustered around smarter export paths: fp16 passthrough for sensitive tensors, mixed int5/int6 schemes, EMA/SWA, and GPTQ-lite percentile search.
2. **LeakyReLU(0.5)^2 already helped on the current frontier stack.** The current top record reports a meaningful gain from that activation alone on a very similar 11-layer architecture.

This candidate keeps the strong 2026-03-22 base intact and focuses on the most obvious open gap: **make late QAT actually real, learnable, and export-aligned**, without bringing in TTT or parameter-banking complexity.

## Prior repository influences

- **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best clean non-TTT training/export stack in the repo
  - already uses EMA, GPTQ-lite clip search, partial RoPE, LN scale, XSA, VE, and warmdown 3500
- **Key activation influence:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - reported a clear win from `LeakyReLU(0.5)^2`
- **Key failure mode influence:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - explicitly documents that a class-attribute late-QAT switch could be compiled away

There were **no prior `candidates/` directories** in this checkout, so there was no earlier candidate lineage to account for.

## External research that informed this candidate

- **LSQ: Learned Step Size Quantization** (`https://arxiv.org/abs/1902.08153`)
  - motivates learning the quantizer step size itself instead of keeping it fixed
- **PACT** (`https://arxiv.org/abs/1805.06085`)
  - motivates learnable clipping / quantizer parameters during training
- **GPTQ** (`https://arxiv.org/abs/2210.17323`)
  - motivates initializing the learned late-QAT scales from a good post-training quantizer instead of starting from arbitrary scales
- **AWQ** (`https://arxiv.org/abs/2306.00978`)
  - reinforces the broader lesson that calibration-aware quantization decisions matter disproportionately for low-bit LMs

This implementation is intentionally **LSQ-style, not a full paper-faithful LSQ reproduction**: it adopts learned step sizes and STE-style rounding in the smallest repo-compatible form.

## What changed vs the chosen base

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. Replaces the brittle global late-QAT flag with a **real per-layer learned late-QAT path** in `CastedLinear`.
2. Adds **per-row learned int6 step sizes** for linear weights, initialized from the same GPTQ-lite percentile search used by export.
3. Puts those learned quantizer parameters in their own **zero-weight-decay optimizer group** during the late phase.
4. Reuses the learned late-QAT scales at export time so training and final int6 packing target the same quantizer.
5. Swaps the MLP nonlinearity from `relu^2` to **`leaky_relu(x, 0.5)^2`**.
6. Adds a **FlashAttention fallback** plus a small synthetic `SMOKE_TEST=1` path so the script can at least exercise the model construction/forward path outside the full training environment.

Everything else stays intentionally close to the 2026-03-22 base: 11 layers, XSA on the last 4 layers, partial RoPE, LN scale, EMA, tight SWA, VE on layers 9/10, SmearGate, BigramHash, and the same general optimizer split.

## How to run

From this candidate directory:

```bash
RUN_ID=lsq_late_qat \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

- `MLP_NEGATIVE_SLOPE=0.5` (default)
- `LATE_QAT_THRESHOLD=0.15` (default; enables late QAT when LR scale drops below this)
- `QAT_LR_MULT=0.5` (default; scales the learned quantizer LR relative to `SCALAR_LR`)
- `QAT_ENABLED=1` to start with learned QAT active from step 0 instead of only late
- `SMOKE_TEST=1 python train_gpt.py` for the synthetic forward-pass smoke path

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604090138_lsq-late-qat/train_gpt.py` | **Passed** |
| `SMOKE_TEST=1 python candidates/202604090138_lsq-late-qat/train_gpt.py` | **Blocked in this runner** because the repo Python dependencies were not installed here (`numpy`, `sentencepiece`, `torch` were all missing). The script still includes the smoke path for environments with the repo deps installed. |

## Expected risks / tradeoffs

- **Training speed risk:** this is a truer QAT path than the repo's earlier flag-based attempts, so it may need step-time retuning on real 8xH100 hardware.
- **Optimizer interaction risk:** the learned quantizer scales may help export quality but could also fight EMA/SWA or destabilize the last training phase.
- **Scope risk:** the learned scales are auxiliary training-time parameters and are excluded from the final export artifact, so the benefit must transfer through the weight updates themselves.
- **Unverified throughput:** this candidate was kept intentionally close to the proven 2026-03-22 stack, but it still needs a real GPU run to establish whether the quantization win outweighs any speed tax.
