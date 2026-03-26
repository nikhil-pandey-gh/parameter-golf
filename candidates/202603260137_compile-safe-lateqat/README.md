# Compile-Safe Late QAT with Learned Clip Gains

## Hypothesis

The strongest March 2026 stack in this repository already looks close to architecturally saturated: 11 layers, 512 width, 3x MLP, BigramHash, XSA, EMA/SWA, partial RoPE, and a careful int6 export path. The most promising remaining gap is therefore **compression-aware training**, not another broad architecture rewrite.

This candidate tests a narrow hypothesis:

> If the best 11-layer banked model is trained with a **compile-safe late int6 fake-quant ramp** plus **learned clip multipliers** on the banked weights that dominate the compressed artifact, then the final GPTQ-lite/int6 export should retain more of the pre-quant model quality and improve post-quant sliding-window BPB.

## Why this is promising for this repository

Repository review showed three converging signals:

1. The modern winning family is the 11-layer, 512-dim, seq2048, int6-compressed stack with XSA/EMA/bigram features, not the older 9-layer baseline.
2. Several strong records improved primarily by making the model **more compression-aware** rather than radically changing the backbone.
3. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly documents that its Late QAT path was **dead-code-eliminated by `torch.compile`** because a class attribute flag was constant-folded. That means one promising idea was never actually tested in the strongest 11-layer family.

This candidate directly targets that gap instead of inventing a less grounded new architecture.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best score in the repo inventory (`val_bpb: 1.1194`)
  - keeps the strongest known training/eval stack: LeakyReLU(0.5)^2, parameter banking, parallel Muon, BigramHash, XSA, partial RoPE, EMA/SWA, GPTQ-lite export, optional legal TTT

Direct quantization/QAT influences:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - showed that GPTQ-lite per-row clip search and EMA are worth real BPB
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - documented the `torch.compile` Late QAT bug that this candidate specifically avoids
- `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`
  - earlier evidence that int6 QAT-style training can help when the export target is int6
- `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/`
  - earlier full-run STE fake-quant evidence that training for quantization robustness matters

There were **no prior experiments under `candidates/`** at the time this candidate was created.

## External research that informed it

This implementation is a lightweight repository-shaped blend of three research directions:

- **LSQ / learned quantizer scales**: learned step sizes can materially improve quantized-model training quality without needing a new training stack.
  - Esser et al., *Learned Step Size Quantization*, ICLR 2020
  - <https://arxiv.org/abs/1902.08153>
- **GPTQ / stronger post-training weight quantization**: one-shot reconstruction-aware quantization motivates keeping the export path strong and explicit rather than relying on naive rounding.
  - Frantar et al., *GPTQ*, ICLR 2023
  - <https://arxiv.org/abs/2210.17323>
- **AWQ / channel protection via scaling**: salient-channel scaling suggests that tiny learned multiplicative adjustments can reduce quantization damage without broad infra changes.
  - Lin et al., *AWQ*, MLSys 2024
  - <https://arxiv.org/abs/2306.00978>

This candidate does **not** implement full LSQ or full AWQ. Instead, it applies their core intuition in the smallest practical way for this repository: learned scalar clip multipliers on the big banked weight tensors plus a late fake-quant ramp that survives compilation.

## What changed versus the chosen base implementation

Base file:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Compile-safe late QAT ramp**
   - Removes reliance on a mutable class-attribute boolean for late QAT activation.
   - `GPT.forward(...)` now accepts an optional `qat_strength` tensor.
   - The training loop ramps `qat_strength` from 0 to 1 after `LATE_QAT_THRESHOLD` is crossed, so `torch.compile` sees the QAT path as an input-dependent graph, not dead code.

2. **Learned clip multipliers on banked weights**
   - Adds tiny fp32 vectors:
     - `qo_qat_log_scale`
     - `kv_qat_log_scale`
     - `mlp_up_qat_log_scale`
     - `mlp_down_qat_log_scale`
   - These learn how aggressively each bank slice should be clipped before int6 fake-quantization.
   - They are intentionally tiny so they barely affect artifact size.

3. **STE fake quantization only on the banked int6-target weights**
   - The training-time fake quant path is applied to the banked Q/K/V/O and MLP up/down weights that dominate the int6 artifact.
   - The forward uses an STE-style rounding surrogate so gradients still flow through the late QAT phase.

4. **Export path alignment**
   - The learned clip multipliers are reused during GPTQ-lite export, so training-time clip preferences influence the final post-training int6 packing instead of disappearing at save time.

5. **Fallbacks for lightweight validation**
   - Adds a `scaled_dot_product_attention` fallback when `flash_attn_interface` is unavailable.
   - Adds `SMOKE_TEST=1` mode to instantiate a tiny CPU model and run one forward/backward pass without dataset/tokenizer setup.
   - Makes `numpy`/`sentencepiece` imports lazy under `SMOKE_TEST=1` so the smoke path does not depend on the full training environment.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603260137_compile-safe-lateqat

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
LATE_QAT_THRESHOLD=0.15 QAT_CLIP_GAIN_RANGE=0.5 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- This keeps the same overall family as the 2026-03-23 record and adds the new QAT path.
- The copied script still supports the base record's optional legal TTT flags if you want to compare with the exact 2026-03-23 evaluation protocol.
- For a minimal local sanity check in an environment that actually has `torch` installed:

```bash
SMOKE_TEST=1 python candidates/202603260137_compile-safe-lateqat/train_gpt.py
```

## Validation run for this candidate

I ran the following low-cost checks in this workflow environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603260137_compile-safe-lateqat/train_gpt.py
python -m py_compile candidates/202603260137_compile-safe-lateqat/train_gpt.py
```

Outcome:

- `compileall` ✅
- `py_compile` ✅

Attempted smoke check:

```bash
SMOKE_TEST=1 python candidates/202603260137_compile-safe-lateqat/train_gpt.py
```

Outcome:

- Could **not** complete in this workflow environment because the runtime does not have `torch` installed, so no Python-level model execution is possible here.
- The candidate was still adjusted so that, once `torch` is available, the smoke path avoids requiring dataset files, tokenizer files, `numpy`, or `sentencepiece` just to validate a tiny CPU forward/backward pass.

## Main expected risks and tradeoffs

- **Extra late-training overhead**: the fake-quant path and learned clip gains are only meant to activate late, but they still add some cost once the ramp starts.
- **Graph split risk**: `torch.compile` will likely compile a non-QAT graph first and a QAT graph after the late ramp begins. That is intentional, but the exact compile cost needs a real GPU run.
- **Approximate AWQ/LSQ only**: this is a small approximation, not a full activation-calibrated AWQ pipeline or a textbook LSQ implementation.
- **Dominant-weight focus**: the candidate only applies learned late-QAT behavior to the banked int6-target matrices, not every small linear in the file.
- **TTT interaction uncertainty**: the copied script still includes optional legal TTT; the main hypothesis here is about post-training quantization quality, not about changing the TTT recipe.

## Suggested next experiments if this helps

- Sweep `QAT_CLIP_GAIN_RANGE` in `{0.25, 0.5, 0.75}`.
- Compare `LATE_QAT_THRESHOLD` at `{0.10, 0.15, 0.20}`.
- Ablate learned clip gains vs compile-safe late QAT ramp alone.
- If the quant gap narrows, try the same mechanism on the 2026-03-22 EMA/GPTQ-lite branch without TTT to isolate training-only gains.
