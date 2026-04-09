# Bank-aware Late QAT for Parameter-Banked GPTQ-lite Int6

## Hypothesis

The strongest recent stack already trains an 11-layer parameter-banked model and exports it through GPTQ-lite int6, but its late-QAT path only touches `CastedLinear` modules. The banked attention/MLP matrices dominate both parameter count and artifact size, so adding **real late-stage int6 STE fake quantization to those banked weights** should reduce the deployment-time quantization gap more than another small architecture tweak.

## Why this is promising for this repository

Three repository trends point in the same direction:

1. Earlier **real QAT** record branches helped materially once the project moved to int6-heavy exports (`2026-03-19_MLP3x_QAT_Int6_SlidingWindow`, `2026-03-19_Seq2048_FP16Emb_TunedLR`, `2026-03-19_smeargate_orthoinit_muonwd`).
2. `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly notes that its late-QAT flag was effectively inactive under `torch.compile`, so the repo has prior evidence that the idea was promising but not actually exercised.
3. The current best stack (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`) moved the dominant weights into 3D banks for optimizer efficiency, which made the old `CastedLinear` fake-quant path even less representative of the deployed artifact.

This candidate keeps the proven 11L + parameter-banking + GPTQ-lite path intact and only changes the part most tightly coupled to final artifact quality: whether the big banked matrices ever see int6 noise during training.

## Prior records that influenced this candidate

- **`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`**: strongest overall stack; this candidate branches from its train script.
- **`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`**: clean no-TTT base showing GPTQ-lite + EMA + warmdown tuning are still the best pre-export foundation.
- **`2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`**: key negative lesson that compile-time constant folding can silently disable late QAT.
- **`2026-03-19_MLP3x_QAT_Int6_SlidingWindow`** and related 2026-03-19 QAT runs: evidence that real int6 STE training can improve quantized scores when it actually reaches the exported weights.

## External research that informed it

- **EfficientQAT** ([arXiv:2407.11062](https://arxiv.org/abs/2407.11062)) argues that block-wise QAT is a practical route for LLMs, which maps well to this repo's parameter banks.
- **SiLQ** ([arXiv:2507.16933](https://arxiv.org/abs/2507.16933)) shows that a simple end-to-end QAT phase with tiny extra budget can outperform heavier quantization recipes, supporting a late-only intervention instead of broad infrastructure changes.
- **Scaling Law for Quantization-Aware Training** ([arXiv:2505.14302](https://arxiv.org/abs/2505.14302)) reports that weight quantization error becomes increasingly important with more training tokens, which is directly relevant to this 600-second multi-billion-token setup.
- **QLLM** ([arXiv:2310.08041](https://arxiv.org/abs/2310.08041)) emphasizes that outlier channels are a major PTQ bottleneck; this candidate tries to pre-condition the weight distribution before the existing GPTQ-lite export step.

## What changed vs. the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes:

1. Added `fake_quantize_int6_ste(...)` for cheap row-wise int6 fake quantization of 2D bank slices.
2. Added bank-QAT state to `CausalSelfAttention` and `MLP`, so `q/k/v/out` and `mlp up/down` bank slices can be fake-quantized during training.
3. Added `GPT.set_bank_qat(...)` to toggle banked-weight QAT across the model.
4. When late QAT activates, the script now:
   - enables the legacy `CastedLinear` QAT path,
   - enables **banked** QAT,
   - **recompiles once** so `torch.compile` sees the late-QAT path instead of constant-folding it away.
   - keeps bank QAT **late-only** instead of silently changing the meaning of older `QAT_ENABLED=1` recipes.
5. Kept the existing **GPTQ-lite int6 export path unchanged**, so the candidate tests whether training-side bank QAT improves the same deployment pipeline rather than changing the quantizer itself.

## How to run / evaluate

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
LATE_QAT_RECOMPILE=1 BANK_QAT_ENABLED=1 BANK_QAT_PERCENTILE=1.0 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- `TTT_ENABLED=0` remains the default so the training-side quantization change can be measured cleanly.
- If you want to compare against the best full leaderboard recipe, re-enable the score-first TTT flags from the 2026-03-23 record after validating the pre-TTT effect.

## Main expected risks / tradeoffs

- **One extra `torch.compile` pass** at late-QAT start costs time; if compile overhead is too large, the quality gain may be offset by fewer optimizer steps.
- The training surrogate uses a **single cheap int6 fake-quant pass**, while export still uses **GPTQ-lite percentile search**, so there is still some train/export mismatch.
- If late QAT turns on too early, added quantization noise could hurt the already-strong pre-export model.
- `BANK_QAT_PERCENTILE` is intentionally pinned to `1.0` in this candidate because the compiled fullgraph training path needs a compile-safe surrogate.

## Validation

Commands run in this environment:

- `python -m compileall candidates/202604082351_bank-qat-int6/train_gpt.py` — **passed**
- `python - <<'PY' ... exec(compile(...)) ... PY` module-load smoke — **not feasible here** because the environment did not have the repo's Python dependencies installed (`numpy` import failed first), and the script also imports CUDA-specific `flash_attn_interface`, so a meaningful CPU-only startup check was not available without changing infrastructure
