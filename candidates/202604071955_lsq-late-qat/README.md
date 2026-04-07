# LSQ-Style Late QAT on the 11L GPTQ-lite Stack

## Hypothesis

The current frontier already uses strong **post-training** quantization, but the repository still lacks a working **training-aware learned quantizer** on the best non-banked 11-layer stack. Replacing the dead late-QAT path with **LSQ-style learned per-row int6 scales** should reduce the train-to-int6 roundtrip gap and improve final validation bits-per-byte without changing the broad architecture.

## Why this is promising for this repository

This repository’s strongest non-TTT records already converge on the same pattern:

- 11 transformer layers,
- partial RoPE and LN scaling,
- EMA / warmdown tuning,
- GPTQ-lite or related fixed post-training quantization,
- int6 artifact pressure as the dominant bottleneck.

That makes quantizer learning a natural next move. The key repo-specific reason to try it now is that **`2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly documented that its late-QAT path never actually activated because `torch.compile` constant-folded the class flag**. So the “QAT slot” is still open: the idea was directionally right, but the implementation was inert.

## Which records influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - chosen base implementation,
   - strongest clean non-banked 11-layer stack with GPTQ-lite and EMA already in place.

2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
   - provided the clearest evidence that late QAT is worth revisiting,
   - but also showed the old flag-based implementation was dead code.

3. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - confirms the frontier is now mostly gaining through evaluation-time tricks and careful training polish,
   - which makes a quantizer-learning intervention on the core model a good distinct next bet.

## External research that informed it

1. **Esser et al., 2019/2020 — _Learned Step Size Quantization_**  
   <https://arxiv.org/abs/1902.08153>  
   Main takeaway: learn the quantizer step size itself, not just the weights, and scale its gradient so low-bit training remains stable.

2. **Choi et al., 2018 — _PACT: Parameterized Clipping Activation for Quantized Neural Networks_**  
   <https://arxiv.org/abs/1805.06085>  
   Main takeaway: train clipping/scaling parameters directly instead of relying on fixed hand-set quantizer ranges.

3. **Repository-specific research pass**
   - The external review ranked learned-scale / learned-clip QAT as the best practical next idea for this repo because it is both **novel relative to current records** and **directly aligned with the known weak spot**: fixed PTQ plus an ineffective late-QAT hook.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate changes the quantization path in four focused ways:

1. **Replace the old class-flag late-QAT path with tensor-gated LSQ-style fake quantization**
   - each `CastedLinear` now owns a learned per-row `qat_scale`,
   - fake quantization uses STE rounding plus LSQ-style gradient scaling,
   - late activation now recompiles the training graph after switching QAT on, so pre-QAT steps avoid fake-quant overhead and the compiled graph cannot dead-code-eliminate the real QAT branch.

2. **Reinitialize learned quantizer scales after weight initialization**
   - `qat_scale` is reset from each row’s average absolute weight magnitude after the model’s orthogonal / zero-projection init finishes.

3. **Use learned scales in the final int6 export**
   - when a learned `*.qat_scale` exists for a weight matrix, export quantization uses it instead of falling back to GPTQ-lite percentile search for that matrix.

4. **Keep learned scales out of the shipped raw model state**
   - `*.qat_scale` tensors are excluded from `final_model.pt`,
   - the eval model loads with `strict=False` because those training-only quantizer parameters are not needed after the int6 roundtrip artifact is built.

5. **Synchronize late-QAT activation across ranks**
   - the enable decision is all-reduced before toggling QAT, so every DDP rank switches on the same step before recompiling.

## How to run or evaluate it

The candidate defaults already encode the intended configuration, so from this directory the simplest run is:

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable learned late QAT entirely
SEED=1337 LATE_QAT_THRESHOLD=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Turn on fake quant from step 0 instead of late activation
SEED=1337 QAT_ENABLED=1 LATE_QAT_THRESHOLD=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Validation recorded for this candidate:

- `python -m compileall train_gpt.py train_gpt_mlx.py data`
- `python -m compileall candidates/202604071955_lsq-late-qat/train_gpt.py`

Outcome:

- syntax compilation succeeded for the repository baseline and the new candidate script.
- A minimal CPU runtime smoke test was **not feasible** here because the script is explicitly CUDA-only (`torch.cuda.is_available()` hard requirement) and imports `flash_attn_interface`, so a no-GPU start would fail for environment reasons before exercising the candidate logic.

## Main expected risks or tradeoffs

- **Stability risk:** even with late activation, learned scales can destabilize the last part of training if the step sizes shrink too aggressively.
- **Artifact risk:** while the learned scale tensors are not needed at eval, the training path may still encourage weights that do not compress as cleanly as hoped.
- **Interaction risk:** GPTQ-lite was already a strong post-training quantizer on this stack; replacing part of that logic with learned scales may help, but it may also lose some of GPTQ-lite’s reconstruction advantage on specific matrices.
- **Coverage risk:** this candidate only upgrades `CastedLinear`-backed matrices, which is the right scope for this base stack but not a drop-in solution for the parameter-banked frontier code.
