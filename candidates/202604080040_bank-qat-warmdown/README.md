# Bank-aware int6 warmdown projection

## Hypothesis

The current best stack already trains a strong fp/bf16 model, but it still leaves a meaningful **post-EMA -> exported int6** gap because its late QAT path never touches the four large parameter banks that dominate the artifact. This candidate adds a **bank-aware late warmdown projection**: once the wallclock-aware LR scale drops below a threshold, the bank tensors are blended toward their per-row int6 dequantized values after each optimizer step.

The goal is to make the final EMA checkpoint live closer to the eventual int6 export manifold without paying the full throughput cost of always-on STE QAT.

## Why this is promising for this repository

- The strongest current base is the 2026-03-23 record stack: 11L / 3x MLP / BigramHash / XSA4 / EMA+SWA / parameter banking / legal TTT.
- That record still shows a sizable non-sliding export penalty: in `train_seed1337.log`, the post-EMA diagnostic is `val_bpb=1.1369` while the final int6 roundtrip is `1.1452`, a gap of about **+0.0083 bpb** before sliding-window and TTT gains are applied.
- Earlier 2026-03-19 QAT-heavy records showed that when block weights really do see low-bit pressure during training, the quant gap can collapse dramatically.
- The 2026-03-21 partial-RoPE record explicitly notes that compiled late QAT was dead-code-eliminated in a prior stack. The 2026-03-23 winner moved even more weight mass into 3D banks, so a bank-aware path is the natural follow-up.

## Prior records that informed this candidate

- **Base implementation**: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **GPTQ-lite + later warmdown inspiration**: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Late-QAT failure mode**: `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Evidence that real QAT helps when it hits block weights**:
  - `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`
  - `records/track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR/`
  - `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`

There were **no prior `candidates/` directories** in the repo at the time this candidate was created.

There was, however, a **prior unmerged GitHub candidate iteration** in the same general space (`#948` / draft PR `#949`, `202604072316_banked-lwc-lateqat`). This candidate intentionally takes a different route: **no learned clipping tensors, no late recompilation, and no export-time reuse of trained clip parameters**. The twist here is a cheaper, zero-extra-export-parameter projection-only warmdown that aims to preserve more of the 03-23 stack's step budget.

## External research that informed it

- **LSQ: Learned Step Size Quantization** — https://arxiv.org/abs/1902.08153
- **BitNet b1.58** — https://arxiv.org/abs/2402.17764

This candidate does **not** implement a full LSQ replica. Instead, it adapts the core lesson for this codebase: expose the trainable high-mass matrices to low-bit structure during training, but do it with a lightweight, compile-safe post-step projection that fits the existing banked trainer.

## What changed versus the chosen base implementation

1. **Bank-aware late warmdown projection**
   - Added `BANK_QAT_ENABLED`, `BANK_QAT_START_SCALE`, and `BANK_QAT_MAX_MIX`.
   - During warmdown, when the wallclock-aware LR scale falls below `BANK_QAT_START_SCALE`, the four parameter banks are blended toward a row-wise int6 dequantized copy after each step.
   - The blend factor ramps from 0 to `BANK_QAT_MAX_MIX`, so the projection pressure grows as training approaches export time.

2. **Targeted scope**
   - The projection currently hits the four bank tensors (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`) because they dominate the exported model bytes and were previously untouched by the old `CastedLinear`-based late-QAT path.
   - Export format and evaluation remain unchanged: GPTQ-lite-style mixed int6 export + lzma roundtrip + sliding eval + optional legal TTT.

3. **Local smoke-test friendliness**
   - Added a FlashAttention import fallback to `torch.nn.functional.scaled_dot_product_attention` so the module can be imported and a tiny CPU forward pass can run even when `flash_attn_interface` is unavailable.
   - On **CUDA**, the candidate still fails fast by default if `flash_attn_interface` is missing. Set `ALLOW_SDPA_FALLBACK=1` only if you intentionally want the slower SDPA fallback for debugging.

4. **Old late QAT disabled by default**
   - `LATE_QAT_THRESHOLD` now defaults to `0.0` in this candidate so the new bank-aware path is the active quantization-aware mechanism instead of the older compiled `CastedLinear` toggle.

## How to run

### Main candidate run

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
BANK_QAT_ENABLED=1 BANK_QAT_START_SCALE=0.25 BANK_QAT_MAX_MIX=1.0 \
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

### Useful ablations

```bash
# Disable the new mechanism
BANK_QAT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Start projection later / earlier
BANK_QAT_START_SCALE=0.15 torchrun --standalone --nproc_per_node=8 train_gpt.py
BANK_QAT_START_SCALE=0.35 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

- `python -m compileall candidates/202604080040_bank-qat-warmdown/train_gpt.py` — **passed**
- Tiny CPU import/forward smoke via `python - <<'PY' ...` — **not runnable in this workflow container** because the available Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
- The SDPA fallback is still included so the same smoke command should work in a normal Parameter Golf runtime where `torch` is available; on CUDA it remains opt-in via `ALLOW_SDPA_FALLBACK=1`.

## Main expected risks / tradeoffs

- The projection uses a **cheap row-max int6 approximation**, not full learned clipping or full GPTQ-lite clip search during training. It may underfit the true export quantizer.
- Only the **bank tensors** get warmdown projection. If embeddings or other non-bank tensors dominate the remaining gap, gains could saturate.
- Projection pressure near the end of training may reduce fp/bf16 quality if `BANK_QAT_START_SCALE` is too high.
- The CPU SDPA fallback is correctness-oriented, not speed-oriented, and should not be used to judge leaderboard timing.
