# AWQ-lite buffered GPTQ on the LeakyReLU2 + Legal TTT stack

## Hypothesis

The current top stack already looks close to compute-limited on training dynamics, so the best remaining headroom is likely in **weight-only export quality** rather than another large architectural rewrite. The record GPTQ-lite path still chooses a single clip percentile per matrix; replaying a tiny buffer of **already-seen late-train batches** after EMA to pick **per-row int6 clip percentiles with activation-aware error weighting** should preserve salient channels better at the same artifact size.

## Why this is promising for this repository

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` showed that a small post-training quantization improvement can still buy about **-0.0013 BPB** without changing the main training recipe.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the current best stack and still depends on GPTQ-lite int6 export, so a compression-aware tweak can stack directly on the best-known training/eval recipe.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` found that late QAT was dead code under `torch.compile`, which makes a PTQ-side improvement safer than leaning harder on fake quantization.
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/` explicitly identified quantization quality as the dominant bottleneck once the base model is reasonably trained.

## Prior experiments that influenced this candidate

- **Chosen base:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- **Quantization prior art in-repo:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Stability/ablation signal:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **No prior `candidates/` directory existed** when this candidate was created.

## External research that informed it

- **AWQ** — *Activation-aware Weight Quantization for LLM Compression and Acceleration* (`arXiv:2306.00978`): use activation statistics, not just raw weight error, to protect salient channels.
- **SmoothQuant** — *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* (`arXiv:2211.10438`): offline activation statistics can shift quantization decisions without retraining.
- **GPTQ** — *GPTQ: Accurate Post-Training Compression for Generative Pretrained Transformers* (`arXiv:2210.17323`): post-training output-aware quantization is strong enough to matter even late in an optimization stack.

This candidate does **not** implement full AWQ or SmoothQuant. It takes the smallest repository-native slice that fits this codebase: buffered calibration from late-train batches plus activation-weighted clip search during int6 export.

## What changed versus the chosen base

1. Added `ActivationAwareQuantStats`, which accumulates per-input-channel second moments for quantized matrices.
2. Buffered a small number of late-train token batches (`AWQ_BUFFER_*`) once the LR scale falls below `AWQ_COLLECT_THRESHOLD`.
3. Replayed those buffered batches **after EMA** and **before export** to collect calibration statistics, without reopening train shards after training.
4. Changed `quantize_int6_per_row` from matrix-global clip selection to **true per-row candidate selection**.
5. Scored int6 clip candidates with activation-aware weighted reconstruction error when stats are available.
6. Left the architecture, optimizer stack, sliding eval, and legal TTT path otherwise unchanged.

## How to run

From this directory on the normal 8xH100 environment:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
AWQ_ENABLED=1 AWQ_BUFFER_BATCHES=8 AWQ_BUFFER_EVERY=64 \
AWQ_COLLECT_THRESHOLD=0.20 AWQ_MAX_BATCH_SEQS=16 AWQ_ACTIVATION_POWER=1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The AWQ settings above are also the candidate defaults.

## Validation

| Command | Outcome |
|---|---|
| `python -m py_compile candidates/202604021155_awq-lite-gptq/train_gpt.py` | Passed |
| `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604021155_awq-lite-gptq/train_gpt.py` | Passed |
| Minimal CPU smoke run | Not feasible in this workspace: `torch`, `numpy`, and `sentencepiece` are not installed, and no FineWeb shard directory is present under `data/` |

## Main risks and tradeoffs

- The buffered calibration batches may be too small or too narrow, so the activation weighting could overfit to late-train microbatch statistics.
- Protecting activation-salient channels can sometimes make a row's raw weight range less compressible, so some tensors may benefit while others regress.
- This is still the most complex stack in the repo; if the gain is real but fragile, the cleaner 2026-03-22 non-TTT base may be a better long-term home for the same export idea.
