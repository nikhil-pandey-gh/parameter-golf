# Bank-aware late QAT on LeakyReLU² + Legal TTT + Parallel Muon

## Hypothesis

The current best stack already wins on architecture, training speed, and evaluation, but its late-QAT path only fake-quantizes `CastedLinear` modules. After parameter banking, the exported core attention/MLP weights live in `qo_bank`, `kv_bank`, `mlp_up_bank`, and `mlp_down_bank`, so the scored weights do not see late-stage STE fake quantization at all.

This candidate reuses the exporter's existing `quantize_int6_per_row()` helper inside the forward pass for those banked weights during late QAT. The goal is to reduce the pre-export to post-export quantization gap without increasing artifact size.

## Why it is promising for this repository

The repo's biggest gains after the initial sliding-window eval jump came from compression-aware changes rather than brand-new macro-architectures:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` showed that better per-row clip selection in the int6 exporter was worth about `-0.0006` BPB on its own.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly documented an older late-QAT no-op caused by compile-time constant folding.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the strongest current stack, so it is the right place to test a more exporter-matched fake-quant path.

In short: the repo already proved that post-training quantization details matter, and the current parameter-banked implementation leaves a clear opening to push that idea further.

## Which records or prior candidates influenced it

There were no prior experiments under `candidates/` when this candidate was created.

The main record influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best overall stack to fork
  - provides LeakyReLU(0.5)², parameter banking, parallel Muon, and legal score-first TTT
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - shows that better int6 clipping materially improves the scored artifact
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - documents the older late-QAT failure mode and motivated verifying that the new path really touches the exported weights

## External research that informed it

Primary sources that shaped this candidate:

- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (`https://arxiv.org/abs/1712.05877`)
- Esser et al., "Learned Step Size Quantization" (`https://arxiv.org/abs/1902.08153`)
- Fan et al., "Training with Quantization Noise for Extreme Model Compression" (`https://arxiv.org/abs/2004.07320`)

These papers all point in the same direction for this repo: if the leaderboard score is determined by the compressed/exported artifact, then training should expose the model to that distortion late in training instead of only quantizing after the fact.

I also considered larger structural changes like ALBERT-style sharing (`https://arxiv.org/abs/1909.11942`) and Universal-Transformer-style recurrence (`https://arxiv.org/abs/1807.03819`), but they are broader architectural bets. For this repo, exporter-matched quantization pressure is the tighter fit to the observed winning trend.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added `ste_fake_quantize_int6()` and reused the existing `quantize_int6_per_row()` export helper inside the training forward path.
2. Applied late fake quantization to the banked core weights used by:
   - attention query/key/value/output projections
   - MLP up/down projections
3. Kept `CastedLinear` late QAT, but routed it through the same helper so the fake-quant path is consistent.
4. Increased the default `LATE_QAT_THRESHOLD` from `0.15` to `0.20` so the bank-aware fake-quant path runs for a slightly larger late-training slice.
5. When late QAT activates, the training loop intentionally falls back from the precompiled graph to eager mode. This avoids the older `torch.compile` constant-folding failure mode that previously turned late-QAT branches into no-ops.
6. Split small non-banked matrices (bigram / VE / MTP projections) into an explicit AdamW param group so the copied script's optimizer split matches the code comments and remains correct if auxiliary heads are enabled later.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603301125_bank-aware-qat
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.20 \
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

Like the record it forks, this script expects CUDA, PyTorch with distributed support, and `flash_attn_interface`.
Late QAT activates automatically once the wallclock-aware LR scale falls below `LATE_QAT_THRESHOLD`.

## Main expected risks or tradeoffs

- The forward-pass fake quant now reuses the exporter's 5-candidate per-row clip search, which is more faithful but also heavier than the previous row-max-only `CastedLinear` path.
- Falling back to eager mode when late QAT activates is correctness-first. It should keep the fake-quant branch live, but it may cost some late-stage throughput relative to a fully compiled run.
- If the extra late-stage overhead reduces the total number of training steps too much inside the 600-second cap, the threshold may need to move back toward `0.15`.
- This candidate specifically targets the quantization gap. If the current stack is already close to its quantization floor, the added training noise may not pay for its extra cost.

## Validation

Commands run in this workflow environment:

```bash
python -m compileall train_gpt.py candidates/202603301125_bank-aware-qat/train_gpt.py
python - <<'PY'
import importlib.util
print('torch_spec', importlib.util.find_spec('torch'))
print('flash_attn_interface_spec', importlib.util.find_spec('flash_attn_interface'))
PY
```

Outcomes:

- `python -m compileall ...` passed for both the root baseline and this candidate.
- A runtime smoke test was **not feasible in this runner** because both `torch` and `flash_attn_interface` are unavailable here, and the script also hard-requires CUDA in `main()`.
