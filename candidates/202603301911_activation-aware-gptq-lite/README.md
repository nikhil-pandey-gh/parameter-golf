# Activation-Aware GPTQ-lite on the LeakyReLU² + Legal TTT Stack

## Hypothesis

The current frontier in this repository looks more **quantization-limited** than architecture-limited. The best local record already has a strong 11-layer training stack plus legal score-first TTT, but it still relies on simple weight-only clip selection during export.

This candidate tests a minimal, repo-compatible upgrade: keep the same training stack and artifact format, but make post-training quantization **activation-aware** by calibrating on a small number of train batches and choosing per-row clip values using an activation-weighted error proxy instead of plain weight reconstruction error.

## Why this is promising for this repository

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is already very strong on the training side (`val_bpb: 1.1194` post-TTT mean), so the cleanest remaining lever is export quality.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` shows current GPTQ-lite clip search helps, but only by about `-0.0006` BPB, which suggests there is still headroom inside the quantizer itself.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` explicitly calls `tok_emb.weight` the most sensitive tensor. That is exactly the kind of tensor activation-aware calibration should help, especially because tied embeddings also act as the output head.
- The artifact budget is already close to full but still under 16 MB, so a **quality-preserving export tweak** is more attractive than adding new runtime parameters.

At review time there were **no prior `candidates/` directories**, so this is the first candidate iteration in that namespace.

## Prior repository experiments that influenced this candidate

- **Primary base:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Best local record in the repository.
  - Keeps the strongest known training/eval recipe: LeakyReLU², parameter banks, Parallel Muon, GPTQ-lite-style mixed quantization, and legal TTT.
- **Quantization trend:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Demonstrates that post-training clip-search changes can still buy measurable BPB improvement.
- **Embedding sensitivity warning:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - Shows the tied embedding/output matrix is especially fragile under quantization.
- **Dead-end warning:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Reports recurrence hurt under a strict wallclock cap, which pushed this candidate away from new depth/reuse tricks and toward export-side quality.

## External research that informed it

- **AWQ** — Tang et al., _AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration_ (arXiv:2306.00978)  
  Uses activation statistics to identify which quantization errors matter most.
- **OmniQuant** — Shao et al., _OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models_ (arXiv:2308.13137)  
  Argues that small calibration sets can materially improve PTQ quality without changing the final runtime format.
- **SmoothQuant** — Xiao et al., _SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models_ (arXiv:2211.10438)  
  Reinforces the idea that activation outliers are a major reason weight-only quantization underperforms.
- **AdaRound** — Nagel et al., _Up or Down? Adaptive Rounding for Post-Training Quantization_ (arXiv:2004.10568)  
  Motivates using data-aware quantization objectives instead of nearest-rounding or pure weight-MSE heuristics.

This candidate does **not** try to implement the full AWQ / OmniQuant / SmoothQuant stack. It only takes the smallest practical slice that fits this repository: **activation-weighted clip search** with the existing mixed int6/int8 exporter.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Add a calibration pass before export**
   - New env vars:
     - `ACTIVATION_AWARE_QUANT=1`
     - `CALIBRATION_BATCHES=16`
   - After EMA/LAWA application and the post-EMA diagnostic eval, the script replays a few train batches under `torch.inference_mode()`.

2. **Collect activation second moments for the exact linear inputs that will be quantized**
   - Attention Q/K/V inputs
   - Attention output projection inputs
   - MLP up/down projection inputs
   - Final normalized hidden states feeding the tied output head (`tok_emb.weight`)

3. **Replace weight-only clip selection with activation-weighted clip search**
   - For 2D int6/int8 matrices, clip candidates are scored using a diagonal output-error proxy:
     - rows that interact with larger observed activation variance are penalized more heavily
   - This keeps the search cheap and keeps the exporter simple.

4. **Keep the artifact format unchanged**
   - The script still exports the same mixed int6/int8 tensors plus `lzma`.
   - No new runtime dependency or evaluator change is required.
   - Legal TTT remains unchanged.

## How to run

From the repository root:

```bash
cd candidates/202603301911_activation-aware-gptq-lite

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
ACTIVATION_AWARE_QUANT=1 CALIBRATION_BATCHES=16 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The only new knobs relative to the base record are `ACTIVATION_AWARE_QUANT` and `CALIBRATION_BATCHES`.

## Main expected risks and tradeoffs

- **Extra post-training time:** calibration adds extra forward passes after the 600-second training window.
- **Diagonal approximation:** this is not full AWQ or OmniQuant; it ignores cross-feature covariance and only uses second moments.
- **Tied embedding mismatch:** the calibration signal for `tok_emb.weight` comes from output-head activations, which may not fully capture embedding-lookup sensitivity.
- **Calibration overfit:** too few or too many batches could hurt robustness.

## Validation

Commands run locally in this workflow:

```bash
python -m compileall candidates/202603301911_activation-aware-gptq-lite/train_gpt.py
```

Outcome: **passed**.

```bash
python - <<'PY'
from pathlib import Path
path = Path('candidates/202603301911_activation-aware-gptq-lite/train_gpt.py')
text = path.read_text(encoding='utf-8')
checks = {
    'collect_activation_stats': 'def collect_activation_stats(' in text,
    'activation_calibration_log': 'activation_calibration:batches' in text,
    'weighted_quant_call': 'mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, activation_stats=activation_stats)' in text,
    'tok_emb_calibration': '_record_activation(calibration_state, head_name, x)' in text,
}
assert all(checks.values()), checks
print(checks)
PY
```

Outcome: **passed** (`structure_smoke_ok`).

A fuller import/constructor smoke test was **not feasible in this runner** because the repository's normal Python dependencies were not installed here (`numpy`, `sentencepiece`, and `torch` were all absent when checked via `importlib.util.find_spec`), so only syntax and structure-level validation were safe without mutating the shared environment.
