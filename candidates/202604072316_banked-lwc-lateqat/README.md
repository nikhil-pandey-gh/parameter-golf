# Banked LWC LateQAT

## Hypothesis

The strongest next step is to make late quantization-aware training actually touch the tensors that dominate the exported artifact: the banked attention/MLP weights and the tied embedding. This candidate adds a lightweight learned rowwise clipping path during the late-training phase so the training-time fake-quantizer and the export-time int6/int8 quantizer are finally aligned.

## Why this is promising here

The repo history shows that quantization quality stayed central even after architecture gains stacked up:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` improved by tightening export-time GPTQ-lite clipping.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` explicitly documents that its late-QAT path was effectively dead under `torch.compile`.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/` both point at quantization as a major residual bottleneck.

The current best stack in `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is therefore a strong base, but it still leaves a clear opening: its main banked weights bypass the old `CastedLinear` fake-quant path entirely.

## Prior repo experiments that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Quantization/export inspiration:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Bug/fragility to fix:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Embedding sensitivity:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`

There were no existing `candidates/` folders in this repo when this candidate was created.

## External research

- **LSQ**: learned quantizer step sizes with small training-code changes  
  <https://arxiv.org/abs/1902.08153>
- **PACT**: learned clipping parameters instead of fixed activation ranges  
  <https://arxiv.org/abs/1805.06085>
- **OmniQuant**: learnable weight clipping for low-bit LLM quantization  
  <https://arxiv.org/abs/2308.13137>

This candidate intentionally implements the smallest repo-compatible version of that idea family: learned rowwise clip multipliers anchored to the current row maxima, enabled only late in training and reused directly at export.

## What changed vs. the chosen base

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

1. Adds **learned rowwise clip parameters** for:
   - `qo_bank`
   - `kv_bank`
   - `mlp_up_bank`
   - `mlp_down_bank`
   - tied token embedding weights
2. Applies **bank-aware fake quantization** during training so the main transformer weights no longer bypass late QAT.
3. Enables late QAT in a **compile-safe** way by recompiling once when the late-training threshold is crossed, instead of relying on a class attribute that can be constant-folded away.
4. Reuses the learned rowwise clip multipliers during export quantization instead of falling back to percentile search for those tensors.
5. Keeps the rest of the winning March 23 stack unchanged: LeakyReLU(0.5)^2, XSA on the last 4 layers, Partial RoPE, LN scale, VE128, EMA+tight SWA, parameter banking, legal score-first TTT, and lzma-compressed mixed int6/int8 export.

## How to run

From this candidate directory:

```bash
RUN_ID=banked_lwc_lateqat \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
LEARNED_CLIP_QAT=1 LEARNED_CLIP_THRESHOLD=0.15 \
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

`QAT_ENABLED=1` is also supported if you want to start with learned clipping active from step 0, but the default intent of this candidate is a late-only activation via `LEARNED_CLIP_THRESHOLD`.

## Expected tradeoffs / risks

- **Late-train overhead:** bank-aware fake quantization touches every major matrix once enabled, so the last training phase will be slower than the base run.
- **Approximate LSQ/OmniQuant-lite, not full OmniQuant:** this uses learned rowwise clip multipliers tied to current row maxima rather than a fully separate learned quantizer state.
- **Optimizer interaction risk:** the learned clip tensors live in the scalar/control optimizer path, which is simple and minimal but not separately tuned.
- **Export sensitivity:** if learned clipping overfits the final late-training window, post-lzma artifact quality could regress even if the raw model improves.

## Validation

### Ran successfully

```bash
python -m compileall candidates/202604072316_banked-lwc-lateqat/train_gpt.py
```

Outcome: **passed**

### CPU smoke check feasibility

Attempted a minimal import-based smoke check for `train_gpt.py`, but this runner does not currently have the repo's required Python modules installed (`numpy`, `torch`, `sentencepiece`) and also lacks the CUDA-specific `flash_attn_interface` import used by the record stack.

Outcome: **a meaningful CPU startup smoke test was not feasible in this environment**
