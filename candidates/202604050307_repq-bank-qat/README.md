# 202604050307 RepQ-style Bank-Aware Late QAT

## Hypothesis

The current best stack in this repo moved the dominant attention/MLP weights into large parameter banks for faster training, but that also means the existing late-QAT path on `CastedLinear` no longer touches most of the model. This candidate restores compression-aware training where it matters: it fake-quantizes the banked weights directly during late warmdown, using trainable per-row clip multipliers and a one-time recompile when QAT turns on so `torch.compile` cannot constant-fold the old non-QAT path.

If the repo's recent gains are already coming from small quantization-alignment wins, then bringing the banked weights into the late-QAT loop should be a higher-value next step than another large architecture change.

## Why this is promising for this repository

Recent records show a clear pattern:

- the best non-TTT stack kept improving through better quantization alignment (`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`);
- the best overall stack switched to parameter banking plus LeakyReLU² and legal score-first TTT (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`);
- an earlier record documented that late QAT could silently no-op under `torch.compile` when toggled after the initial trace (`2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`).

This candidate is aimed directly at that intersection: keep the latest strong recipe, but make late QAT actually cover the re-parameterized bank weights and explicitly recompile once the QAT path is enabled.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - best overall repo result;
   - provides the LeakyReLU² + legal TTT + parameter-bank base.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - strongest non-TTT quantization-focused stack;
   - motivates continuing to attack train/export mismatch instead of reshaping the model again.
3. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
   - explicitly records the `torch.compile` constant-folding late-QAT failure mode;
   - motivated the one-time recompile when QAT activates here.

## External research that informed it

1. **RepQ: Generalizing Quantization-Aware Training for Re-Parametrized Architectures** (Prutianova et al., 2023, arXiv:2311.05317)
   - key takeaway: apply QAT to the differentiable test-time weights of re-parameterized layers, not just the underlying training representation.
2. **Learned Step Size Quantization** (Esser et al., 2020, arXiv:1902.08153)
   - key takeaway: low-bit QAT improves when the quantizer parameters themselves can adapt during training.

This candidate uses a lightweight LSQ-inspired version of that idea: trainable per-row log clip multipliers on the banked weights, while keeping the export path simple and compatible with the repo's existing GPTQ-lite int6 packaging.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added training-only `*_bank_qat_log_scale` parameters for `qo_bank`, `kv_bank`, `mlp_up_bank`, and `mlp_down_bank`.
2. Added bank-aware fake quantization in `GPT.forward()` and `GPT.forward_logits()` so late QAT reaches the dominant attention/MLP weights instead of only the small `CastedLinear` modules.
3. Changed the late-QAT toggle to **recompile once** after enabling QAT, preventing `torch.compile` from freezing the pre-QAT branch.
4. Reused the learned bank-QAT log scales during GPTQ-lite export as row-wise clip multipliers, then stripped those tensors from the serialized artifact so they still do not consume artifact budget.
5. Fixed the default `DATA_PATH` and `TOKENIZER_PATH` so the script works when launched from inside this candidate directory.

Everything else stays intentionally close to the 2026-03-23 base: LeakyReLU² MLPs, XSA on the last 4 layers, partial RoPE, LN scaling, value embeddings, EMA/SWA, GPTQ-lite int6 export, and legal score-first TTT.

## How to run

From the repository root:

```bash
cd candidates/202604050307_repq-bank-qat
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablation:

```bash
cd candidates/202604050307_repq-bank-qat
SEED=1337 BANK_QAT=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults in this script are already set to the intended candidate configuration.

## Validation

Executed in this repo:

1. `python -m compileall candidates/202604050307_repq-bank-qat/train_gpt.py` - passed
2. `python -m compileall train_gpt.py train_gpt_mlx.py data` - passed

A minimal CPU smoke test was **not feasible** in this runner:

- runtime dependencies required by this script (`numpy`, `sentencepiece`, `torch`, `flash_attn_interface`) are not installed here;
- even with those Python packages present, the actual forward path depends on CUDA + FlashAttention rather than a CPU fallback.

## Main risks and tradeoffs

1. **Late recompile cost:** the one-time `torch.compile` rebuild near the QAT threshold may cost a small number of training steps.
2. **Extra late-stage overhead:** fake-quantizing the full parameter banks increases compute during the final warmdown segment.
3. **Approximate train/export match:** export now reuses the learned log-scale multipliers, but it still wraps them around the existing GPTQ-lite percentile search rather than exporting the training quantizer verbatim.
4. **Stability risk:** the base stack is already highly tuned, so even a compression-aware change can lose more from extra overhead than it gains from reduced quantization gap.
