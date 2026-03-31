# MTP Auxiliary Heads on the 11L EMA/GPTQ-lite Stack

## Hypothesis

Training-only multi-token prediction (MTP) heads can improve sample efficiency on the strongest non-TTT 11-layer stack in this repository, while preserving the 16MB artifact budget because the auxiliary heads are excluded from export.

The candidate also folds in the cheap activation win from the current top record by switching the MLP from ReLU^2 to LeakyReLU(0.5)^2.

## Why this is promising for this repository

The record history has largely converged on one backbone:

- 11 layers, 512 width, 2048-token training/eval
- SmearGate + BigramHash
- XSA on deep layers
- Partial RoPE + LN scale
- VE128 on late layers
- EMA/Tight SWA
- strong mixed int6 export

At that point, many remaining gains are tiny and expensive. Training-only MTP is attractive here because:

- it adds supervision during the fixed 10-minute training budget,
- it does not require a new inference pipeline,
- and the recent strong record scripts already exclude `mtp_heads` from export, so the extra heads do not count against artifact size.

This is a cleaner next bet than adding more eval-only complexity such as TTT, and lower-risk than heavier quantization surgery such as learned rotations.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the base implementation because it is the strongest pure train/export stack before the final TTT-heavy record.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the LeakyReLU(0.5)^2 activation change, which was ablated there as a real win.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - useful mainly as a cautionary note: its README documents a late-QAT path that was accidentally constant-folded away by `torch.compile`, so this candidate avoids relying on that sort of fragile toggle for its main idea.

## External research that informed the idea

- Fabian Gloeckle et al., _Better & Faster Large Language Models via Multi-token Prediction_, arXiv:2404.19737
  - reports that joint multi-token prediction improves sample efficiency and helps induction-like behavior.
- Somesh Mehra et al., _On multi-token prediction for efficient LLM inference_, arXiv:2502.09419
  - argues that jointly training MTP heads with the backbone is much more promising than trying to retrofit MTP onto a frozen next-token model.
- John Kirchenbauer et al., _Multi-Token Prediction via Self-Distillation_, arXiv:2602.06019
  - further supports MTP as a promising direction for standalone models without requiring a separate verifier/speculator pipeline.

I also considered newer quantization-specific ideas such as SmoothQuant and QuaRot, but they looked like worse fits for this repository's constraints: they require more invasive architecture/export surgery, while this repository already has a mature compression-focused stack and a latent MTP implementation ready to exploit.

## What changed versus the chosen base implementation

Base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS` default changed from `0` to `2`
   - `MTP_LOSS_WEIGHT` default changed from `0.2` to `0.15`
   - the heads remain training-only and are still stripped before export

2. **Switch the MLP activation**
   - `relu^2` -> `LeakyReLU(0.5)^2`

3. **Add a FlashAttention fallback**
   - if `flash_attn_interface` is unavailable, the script falls back to `torch.nn.functional.scaled_dot_product_attention`
   - this is mainly to make the script importable and smoke-testable on CPU; the intended H100 path still uses FlashAttention when available

4. **Add a tiny `SMOKE_TEST=1` mode**
   - runs a small CPU forward/backward pass
   - uses a deterministic shifted-token fixture so the MTP offsets are well defined
   - asserts that the MTP heads receive non-zero gradients
   - verifies that export still excludes `mtp_heads`
   - runs a mixed-int6 quantize/dequantize roundtrip and a logits pass on the exported eval model

## How to run

From this candidate directory:

```bash
RUN_ID=mtp_aux_heads \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate script searches upward for the repository root using stable repo markers, then resolves the default dataset and tokenizer paths from there. That makes the command above work directly from this candidate directory and keeps the same defaults valid if the script is later copied into a `records/...` folder.

The defaults in this candidate already correspond to the intended recipe:

- 11 layers, width 512, 2048 sequence length
- EMA + tight SWA
- XSA on the last 4 layers
- partial RoPE (16 dims) + LN scale
- VE enabled on late layers
- BigramHash + SmearGate
- GPTQ-lite mixed int6 export
- **2 training-only MTP heads with weight 0.15**
- **LeakyReLU(0.5)^2**

If the extra heads are too expensive, the obvious follow-up sweep is:

```bash
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate / validate

Lightweight validation used for this workflow:

```bash
python -m compileall train_gpt.py
```

Outcome:

- success

Minimal CPU smoke validation:

```bash
SMOKE_TEST=1 python train_gpt.py
```

Outcome in this workflow:

- success
- printed:

```text
smoke_test:ok loss:6.3051 logits_shape:(2, 16, 128) excluded_mtp_params:73728 mtp_grad_sums:[20.2365, 20.416]
```

Note: the workflow container did not have the repo's Python runtime deps preinstalled, so I ran the smoke check inside a temporary virtualenv after installing `numpy`, `torch`, `sentencepiece`, and `zstandard`. The candidate script itself does not depend on that temp path.

## Main expected risks and tradeoffs

- **Step-time risk:** even training-only MTP heads increase compute. If the extra output projections reduce step count too much, the sample-efficiency gain may be canceled out.
- **Scale risk:** the strongest MTP results in the literature are often shown on larger models, so the gain may be smaller at this tiny scale.
- **Optimization interaction risk:** MTP changes the representation pressure on the trunk, which may interact nontrivially with EMA, warmdown, and quantization-focused training.
- **Activation interaction risk:** LeakyReLU(0.5)^2 was a win in the final record, but it has not yet been explicitly combined with this exact EMA/GPTQ-lite stack.

## Suggested next experiments if this starts well

1. Sweep `MTP_NUM_HEADS` over `{1, 2, 4}`.
2. Sweep `MTP_LOSS_WEIGHT` over `{0.10, 0.15, 0.20}`.
3. If the throughput hit is small, test whether MTP stacks cleanly with the record's later systems changes.
4. If throughput drops too much, keep the LeakyReLU^2 activation and fall back to a 1-head MTP setup.
