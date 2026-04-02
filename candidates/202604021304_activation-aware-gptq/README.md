## Candidate: Activation-Aware GPTQ-lite

### Hypothesis

The current non-TTT 11-layer line is already strong at training-time architecture and optimizer choices; one of the biggest remaining losses is the float-to-int6 export gap. A short post-training calibration pass that measures which input channels are actually active, then uses those statistics to steer the existing GPTQ-lite clip search, should reduce roundtrip error without changing the trained model or adding meaningful artifact bytes.

### Why this is promising here

The repo trend is clear:

- the biggest recent non-eval wins came from compression-aware finishing rather than brand-new architecture (`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`);
- embeddings and export quantization are repeatedly called out as the fragile part of the 16 MB budget (`records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`);
- the latest TTT record is stronger overall, but its main gains are evaluation-time and orthogonal to export quality (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`).

That makes activation-aware export logic a good fit for a new candidate: minimal code churn, no new infrastructure, and easy to stack later with LeakyReLU or legal TTT if it works.

### Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the direct base because it is the strongest self-contained compression-focused stack in the repo.
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` provides the same 11L / partial-RoPE / LN-scale backbone that the base run keeps.
3. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` shows that the frontier has moved toward evaluation-time adaptation, but also highlights that export quality is still worth improving because TTT is layered on top of quantized checkpoints rather than replacing them.

### External research that informed it

1. **AWQ** ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)): protect salient weight channels using offline activation statistics rather than weight magnitude alone.
2. **SmoothQuant** ([arXiv:2211.10438](https://arxiv.org/abs/2211.10438)): activation outliers are the core quantization problem, so lightweight offline calibration is often enough to recover most of the gap.
3. **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)): even fixed preconditioning that removes outliers can make low-bit quantization much easier; this candidate uses the same intuition but stays inside the repo's existing scalar-quantization path.

### What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. adds an **AWQ-lite calibration pass** after EMA export preparation;
2. registers forward pre-hooks on each large `CastedLinear` layer and collects per-input-channel RMS over a small number of training batches;
3. changes the int6 GPTQ-lite clip search from plain reconstruction MSE to a blend of plain MSE and **activation-weighted** MSE, so high-usage channels influence the chosen clip percentile more;
4. keeps the artifact format simple: the exported checkpoint is still dequantized back into the same architecture, with no extra runtime modules or new files;
5. adds `CPU_SMOKE_ONLY=1` plus a guarded FlashAttention import so the candidate-specific quantizer can be sanity-checked without CUDA.

### How to run

From this candidate directory:

```bash
RUN_ID=awq_lite_seed1337 \
SEED=1337 \
AWQ_ENABLED=1 \
AWQ_CALIB_BATCHES=16 \
AWQ_CALIB_BATCH_TOKENS=131072 \
AWQ_BLEND=0.75 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults in this file keep the same 11L / XSA4 / partial-RoPE / LN-scale / EMA / GPTQ-lite backbone as the 2026-03-22 base. The new knobs only affect the export-time calibration:

- `AWQ_ENABLED`: turn activation-aware export on or off
- `AWQ_CALIB_BATCHES`: how many calibration batches to scan
- `AWQ_CALIB_BATCH_TOKENS`: calibration batch size in global tokens
- `AWQ_BLEND`: interpolation between plain MSE and activation-weighted MSE during clip search

For a local non-CUDA sanity check:

```bash
CPU_SMOKE_ONLY=1 python train_gpt.py
```

### Validation

Commands run while preparing this candidate:

1. `python -m compileall candidates/202604021304_activation-aware-gptq/train_gpt.py` - passed
2. `python3 -m venv /tmp/gh-aw/agent/pgolf-venv && /tmp/gh-aw/agent/pgolf-venv/bin/pip install numpy sentencepiece torch` - passed
3. `CPU_SMOKE_ONLY=1 /tmp/gh-aw/agent/pgolf-venv/bin/python candidates/202604021304_activation-aware-gptq/train_gpt.py` - passed (`cpu_smoke:ok compressor:zlib tracked_layers:4 awq_blend:0.75 mse:0.000000`)

The smoke path is intentionally limited to candidate-specific quantization and roundtrip serialization. Full training and evaluation still require the normal Parameter Golf CUDA environment with FlashAttention available.

### Main expected risks and tradeoffs

1. This is **AWQ-lite**, not full AWQ: it uses activation statistics to choose better clip thresholds, but it does not add exact scale-folding or mixed-precision salient-channel retention.
2. The calibration pass adds extra export time; if the gain is tiny, the extra complexity may not pay for itself.
3. Activation-weighted clip search could overfit the calibration subset if `AWQ_CALIB_BATCHES` is too small or too domain-specific.
4. If this helps, the natural next experiments are to stack it with the later LeakyReLU/TTT branch, or to extend it toward true scale-folded AWQ or SmoothQuant-style preconditioning.
