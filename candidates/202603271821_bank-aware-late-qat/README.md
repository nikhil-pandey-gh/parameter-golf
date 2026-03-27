# Bank-Aware Late QAT on the 11L LeakyReLU2 + GPTQ-lite + TTT Stack

## Hypothesis

The current best banked stack already squeezes strong gains from GPTQ-lite post-training quantization, EMA/SWA, and LeakyReLU(0.5)^2, but its late-QAT path still misses the tensors that matter most: the four large parameter banks (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`). A bank-aware STE int6 fake-quant path, enabled only in late warmdown and paired with a one-time `torch.compile` recompile when QAT turns on, should reduce the final GPTQ-lite int6 roundtrip gap without changing the model architecture or increasing artifact size.

## Why this is promising for this repository

Three repository trends point in the same direction:

- Quantization quality is now a primary bottleneck. The 4-hour non-record baseline still degraded heavily at export time, and the best record line keeps winning through better export-aware tricks rather than through raw training loss alone.
- `11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15` showed that better PTQ and earlier late-QAT thresholds helped the strong 11-layer stack.
- `11L Partial RoPE + LN Scale + EMA + XSA4` explicitly documents that the earlier late-QAT implementation was constant-folded away by `torch.compile`, so the advertised late-QAT gain never actually reached training.

This candidate tries to fix the actual bottleneck directly: fake-quantize the bank weights that dominate the compressed artifact, then recompile exactly once when late QAT becomes active so the compiled training graph really includes the QAT path.

## Prior records or candidates that influenced this idea

There were no prior `candidates/` directories when this candidate was created.

The main record influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - chosen as the base implementation because it is the strongest documented record in this checkout and already uses parameter banking + Parallel Muon.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - motivates the export-aware focus: GPTQ-lite and earlier late-QAT thresholds both improved the same 11-layer family.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - provides the key bug note: `torch.compile` constant-folded the QAT flag, so the previous late-QAT path never activated.

## External research that informed it

The candidate is grounded in three primary-source papers:

- **LSQ** (`arXiv:1902.08153`, *Learned Step Size Quantization*): supports the core idea that low-bit performance improves when quantization is brought into training via STE-style updates rather than only applied after training.
- **GPTQ** (`arXiv:2210.17323`, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*): motivates keeping the existing strong PTQ path and improving the model so that the final low-bit export is easier to quantize.
- **Scaling Law for Quantization-Aware Training** (`arXiv:2505.14302`): argues that quantization error becomes increasingly weight-driven as training data grows, which matches this repository's setting where the large bank weights dominate the final artifact.

I also considered shared-depth / recurrent-depth ideas from **ALBERT** (`arXiv:1909.11942`) and the **Universal Transformer** (`arXiv:1807.03819`), but repository evidence here is weak: earlier notes already call recurrence a dead end under the fixed 10-minute wallclock, while quantization fixes have repeatedly converted into leaderboard gains.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in `train_gpt.py` for this candidate:

- Added `BANK_QAT_ENABLED` hyperparameter plumbing.
- Added `fake_quantize_bank_weight_int6(...)`, an STE int6 fake-quant helper for 2D bank slices.
- Applied that fake quantization to the banked attention and MLP weights during training:
  - `q_w`, `k_w`, `v_w`, `out_w`
  - `up_w`, `down_w`
- Added `compile_training_model(...)` and recompile the training model once when late-QAT activates, so `torch.compile` does not keep the stale pre-QAT graph.
- Added `run_attention(...)`, which falls back to `torch.nn.functional.scaled_dot_product_attention` when `flash_attn_interface` is unavailable or when running off-CUDA. This does not change the intended Hopper path, but makes import-level smoke testing less brittle when dependencies are incomplete.

The final PTQ/export path is intentionally unchanged: this candidate still leans on the proven GPTQ-lite int6 export logic after training.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603271821_bank-aware-late-qat
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 BANK_QAT_ENABLED=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
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

Notes:

- By default, this candidate resolves `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so it can be launched directly from inside the candidate directory without extra path overrides.
- Keep `QAT_ENABLED=0` if you want the intended **late** activation behavior. The script will enable both normal `CastedLinear` QAT and the new bank-QAT path when the LR multiplier crosses `LATE_QAT_THRESHOLD`, then recompile once.
- Set `QAT_ENABLED=1` only if you explicitly want fake quantization active from the start.

## Validation commands and outcomes

Validation run in this workflow:

```bash
python -m compileall candidates/202603271821_bank-aware-late-qat/train_gpt.py
```

Outcome: **passed**.

Environment probe:

```bash
python - <<'PY'
import importlib.util
for name in ['torch', 'sentencepiece', 'flash_attn_interface']:
    print(f"{name}={'present' if importlib.util.find_spec(name) else 'missing'}")
PY
```

Outcome in this workflow: `torch=missing`, `sentencepiece=missing`, `flash_attn_interface=missing`.

Attempted smoke test:

```bash
python - <<'PY'
# import candidate module, instantiate a tiny GPT, run one forward/backward pass
PY
```

Outcome: **not feasible in this workflow image** because the required Python runtime dependencies are absent (`ModuleNotFoundError: No module named 'torch'`).

## Main expected risks or tradeoffs

- **Training throughput risk**: bank fake-quantization adds late-phase forward overhead. Recompiling only once at activation time is meant to contain that cost.
- **Quantizer mismatch risk**: training-time bank QAT uses a cheap row-max int6 STE, while final export still uses percentile-searched GPTQ-lite. The hope is that late QAT makes the weights more quantization-friendly in general, but it is not a perfect train/export match.
- **Compile stability risk**: the recompile-on-activation path is intentionally surgical, but it is still a new systems behavior that needs a real GPU run.
- **Incremental upside risk**: this is a quantization-gap fix, not a new modeling family. The likely gain is modest but plausibly meaningful in the current leaderboard regime where sub-millibit improvements matter.
