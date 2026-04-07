# LSQ-Inspired Bank-Aware Late QAT

## Hypothesis

The strongest remaining quantization gap in this repository is that the current best TTT stack still relies on post-training int6 export, while the actual high-impact matrices live in parameter banks rather than `CastedLinear` modules. A compile-safe late-QAT path that operates directly on those banked weights, plus learned per-row scale multipliers that feed back into export-time quantization, should reduce the roundtrip gap without changing the rest of the winning recipe.

## Why this is promising here

- The best local record is still the March 23 stack with LeakyReLU^2, parameter banking, legal score-first TTT, and GPTQ-lite int6 export.
- The March 21 record explicitly notes that one late-QAT path did not activate as intended under `torch.compile`.
- The March 22 record shows that export-time quantization details matter even after the architecture is already strong.
- LSQ / LSQ+ style results suggest that learning quantizer scales late in training is one of the most practical ways to improve low-bit robustness without adding large new infrastructure.

## Prior repository evidence

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - Supplies the strongest current stack: LeakyReLU^2, partial RoPE, XSA, VE, EMA/SWA, legal TTT, and parameter banking.
- **Quantization influence:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - Shows that small export-time quantization improvements still move BPB in a strong 11-layer regime.
- **Failure mode addressed:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - Documents a compile-folded late-QAT path that did not affect training.

There were no pre-existing experiments under `candidates/` when this candidate was created.

## External research

- **LSQ** — Esser et al., 2019: https://arxiv.org/abs/1902.08153
- **LSQ+** — Bhalgat et al., 2020: https://arxiv.org/abs/2004.09576
- **QAT overview / practical context** — Nagel et al., 2022: https://arxiv.org/abs/2210.17323

These papers motivate learning quantizer step sizes late in training rather than relying only on static post-training clipping.

## What changed vs the March 23 base

1. **Compile-safe bank QAT**
   - Added a tensor-controlled `qat_strength` path that fake-quantizes the actual banked attention and MLP weights during training.
   - This avoids the compile-fragile class-attribute toggle that previously targeted only `CastedLinear`.

2. **LSQ-inspired learned row scales**
   - Added trainable per-row log-scale tensors for `qo_bank`, `kv_bank`, `mlp_up_bank`, and `mlp_down_bank`.
   - These are optimized with a separate LR and excluded from export artifacts.

3. **Row-wise export selection**
   - Export-time int6 quantization now considers the learned row scales as an additional candidate alongside GPTQ-lite percentile clips.
   - Selection happens per row instead of picking one percentile for the full matrix.

4. **Local fallback attention path**
   - If `flash_attn_interface` is unavailable, the script falls back to `torch.nn.functional.scaled_dot_product_attention`.
   - This is for portability and smoke-import friendliness; the intended fast path on CUDA still uses FlashAttention when available.

## How to run

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
QAT_ENABLED=1 LATE_QAT_THRESHOLD=0.15 QAT_SCALE_LR=0.01 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

| Command | Outcome |
| --- | --- |
| `python -m compileall candidates/202604072251_lsq-bank-late-qat/train_gpt.py` | Passed |
| CPU import/forward smoke test | Not feasible in this workflow environment because the local Python runtime does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`) |

## Main risks / tradeoffs

- Learned row scales add a new optimization surface and may need LR tuning to avoid over- or under-clipping.
- The candidate keeps the base architecture fixed, so gains rely on export robustness rather than a larger modeling change.
- Full H100 behavior still needs an actual run; this workflow only performed lightweight validation.
