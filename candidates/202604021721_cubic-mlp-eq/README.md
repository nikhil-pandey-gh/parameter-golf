# Cubic MLP Equalization on the LeakyReLU^2 + Legal TTT Stack

## Hypothesis

The current best recipes in this repository are still strongly constrained by post-training weight-only compression. In the LeakyReLU^2 MLP used by the latest record, each hidden channel can be rescaled exactly between the up and down projections:

- `W_up' = S W_up`
- `W_down' = W_down S^-2`

for any positive diagonal scale `S`, because `leaky_relu(sx)^2 = s^2 * leaky_relu(x)^2` for `s > 0`. This candidate uses that degree of freedom to balance MLP hidden-channel ranges before GPTQ-lite int6 export, with the goal of reducing quantization error and improving compressed-model BPB at effectively zero training-time cost.

## Why this is promising here

Repository history points to quantization/export quality as a durable bottleneck:

- the non-record 4-hour run still lagged badly after quantization,
- GPTQ-lite and late-stage compression-aware tuning kept improving the leaderboard,
- the strongest current stack already combines LeakyReLU^2, EMA/SWA, XSA, partial RoPE, and legal TTT, so a low-risk export-side improvement is one of the most plausible remaining levers.

This idea is also unusually compatible with the existing code: the MLPs are already squared-activation MLPs, the export path already unbanks weights before GPTQ-lite, and the transform adds no parameters and no training-time FLOPs.

## Prior records that informed it

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` provided the base stack and showed that LeakyReLU^2 is materially better than relu^2.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` showed that better weight-only export alone can still buy meaningful BPB improvements.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` reinforced that zero-parameter architectural refinements can move the needle when the base stack is already strong.

## External research behind the candidate

- **AWQ** (Activation-aware Weight Quantization, arXiv:2306.00978) argues that equivalent per-channel rescaling can protect the most salient weight channels during weight-only quantization.
- **SmoothQuant** (arXiv:2211.10438) shows that offline scale migration can improve quantization by moving outlier burden across adjacent operations via mathematically equivalent transforms.
- **Data-Free Quantization through Weight Equalization and Bias Correction** (arXiv:1906.04721) demonstrates that channel equalization can materially improve low-bit quantization with no retraining.

This candidate adapts those ideas to the repository's squared MLPs with an exact cubic rescaling rule tailored to `LeakyReLU^2`, instead of introducing a broad calibration or reconstruction pipeline.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes:

1. Added `apply_mlp_cubic_equalization(...)`, which rescales each MLP hidden channel on the unbanked export weights before GPTQ-lite quantization.
2. Added three knobs:
   - `MLP_EQ_ENABLED=1`
   - `MLP_EQ_ALPHA=0.75`
   - `MLP_EQ_CLIP=4.0`
3. Logged equalization scale statistics during export for quick sanity checking.
4. Added a FlashAttention import fallback to PyTorch SDPA so the candidate can be imported and smoke-tested on CPU without requiring the Hopper-only attention kernel.

## How to run or evaluate

Run the same training/eval stack as the current record, with cubic MLP equalization enabled by default:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MLP_EQ_ENABLED=1 MLP_EQ_ALPHA=0.75 MLP_EQ_CLIP=4.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To isolate the export-side effect without legal TTT, rerun with `TTT_ENABLED=0`.

## Main expected risks and tradeoffs

- The transform is function-preserving in float space, but it may still interact badly with GPTQ-lite clip search or the final compressor if the scales become too extreme.
- Improvements, if any, are likely to come mostly from the quantized model, so gains may be modest relative to more invasive architecture changes.
- The method currently targets only MLP hidden channels; it does not try to rebalance attention or embedding outliers.

## Validation

Executed lightweight checks:

1. Syntax check:

   ```bash
   python -m compileall candidates/202604021721_cubic-mlp-eq/train_gpt.py
   ```

   Outcome: **passed**.

2. CPU import/forward smoke test plus equalization invariance check (run from an isolated venv because the base container did not have the repo's Python deps preinstalled):

   ```bash
   source /tmp/gh-aw/agent/pg-venv/bin/activate
   python - <<'PY'
   import importlib.util
   from pathlib import Path
   import torch

   path = Path("candidates/202604021721_cubic-mlp-eq/train_gpt.py")
   spec = importlib.util.spec_from_file_location("candidate_train_gpt", path)
   mod = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(mod)

   kwargs = dict(
       vocab_size=64,
       num_layers=2,
       model_dim=64,
       num_heads=4,
       num_kv_heads=2,
       mlp_mult=2,
       tie_embeddings=True,
       tied_embed_init_std=0.005,
       logit_softcap=30.0,
       rope_base=10000.0,
       qk_gain_init=1.0,
       mtp_num_heads=0,
       mtp_loss_weight=0.0,
       bigram_vocab_size=32,
       bigram_dim=16,
       xsa_last_n=1,
       rope_dims=8,
       ln_scale=True,
       dtg=False,
       ve_enabled=False,
       ve_dim=16,
       ve_layers="",
       gated_attention=False,
       value_residual=False,
   )

   model = mod.GPT(**kwargs).eval()
   with torch.no_grad():
       for p in model.parameters():
           p.copy_(torch.randn_like(p) * 0.02)
   inputs = torch.randint(0, 64, (2, 16), dtype=torch.long)
   logits = model.forward_logits(inputs)

   sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
   unbanked = mod._unbank_state_dict(sd, kwargs["num_layers"])
   eq_state, _ = mod.apply_mlp_cubic_equalization(unbanked, kwargs["num_layers"], alpha=0.75, clip=4.0)
   rebanked = mod._rebank_state_dict(eq_state, kwargs["num_layers"], sd)

   model_eq = mod.GPT(**kwargs).eval()
   model_eq.load_state_dict(rebanked, strict=True)
   logits_eq = model_eq.forward_logits(inputs)
   print((logits - logits_eq).abs().max().item())
   PY
   ```

   Outcome: **passed**. The tiny CPU model produced logits with shape `(2, 16, 64)`. I then reran the same invariance test after replacing the default near-zero-initialized weights with randomized nonzero weights so the transformed MLP path was actually exercised. The equalized model still matched the original float model with `max_abs_diff = 4.47e-8`, confirming the export transform is function-preserving up to floating-point roundoff for the tested configuration.
