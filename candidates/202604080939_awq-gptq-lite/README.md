# AWQ-lite GPTQ export on the 11L EMA base

## Hypothesis

The strongest remaining headroom on the non-TTT 11-layer stack is still in the export path, not the training architecture. This candidate tests whether an **AWQ-style activation-aware rescaling step** can reduce the remaining int6 quantization gap on top of the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` recipe by protecting activation-salient input channels before the existing GPTQ-lite clip search.

## Why it is promising for this repository

- Recent record progression repeatedly improved by reallocating artifact bytes more intelligently: mixed low-bit export, GPTQ-lite clipping, zstd/lzma choices, FP16-sensitive passthroughs, and EMA/SWA all mattered.
- The `1.1233` base already searches clip percentiles per matrix row, but it is still **weight-only PTQ**. It never consults activation statistics.
- External PTQ work shows that **activation-aware channel scaling** and **outlier isolation** are especially important once models are pushed into aggressive low-bit export regimes.
- The repository review found no prior record or candidate using AWQ-, SpQR-, QuaRot-, or SpinQuant-style activation-aware export.

## Prior experiments that influenced this candidate

- **Chosen base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strong 11L/XSA/partial-RoPE/LN-scale stack
  - GPTQ-lite clip search already present
  - excellent non-TTT score with relatively localized export code
- **Compression-budget evidence:** `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/` and `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`
  - showed that better low-bit allocation can buy real BPB gains under the same artifact cap
- **Latest SOTA context:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - reinforces that targeted deltas still matter on top of the modern 11L recipe, even if this candidate intentionally avoids heavier TTT and parameter-banking changes
- There were **no prior `candidates/` directories** in the repository when this was created.

## External research that informed it

- **AWQ** — Activation-aware Weight Quantization for LLMs (`arXiv:2306.00978`): argues that activation statistics, not just weight magnitudes, identify the most salient channels, and that equivalent rescaling can reduce weight-only low-bit error.
- **SpQR** — Sparse-Quantized Representation (`arXiv:2306.03078`): motivates treating a small number of outlier-heavy channels/weights differently when pushing models toward very low bitwidths.
- **GPTQ** (`arXiv:2210.17323`): the base export path is already in the GPTQ-lite spirit, so AWQ-style scaling is a natural extension rather than a rewrite.

## What changed versus the chosen base

1. Added **AWQ-specific knobs**:
   - `AWQ_ENABLED` (default `1`)
   - `AWQ_ALPHA_CANDIDATES` (default `0.25,0.5,0.75`)
   - `AWQ_SCALE_LIMIT` (default `4.0`)
2. Added **online activation-stat collection inside the timed training loop** for the large int6 `CastedLinear` weights that dominate the export, so the final artifact does not depend on any extra post-cap training-data pass.
3. Extended `quantize_int6_per_row` to search:
   - the original identity path, and
   - several AWQ-style per-input-channel scaling candidates
   before the existing GPTQ-lite percentile sweep, choosing the lowest reconstruction-MSE option per tensor.
4. Stored the chosen AWQ scale vectors inside the export blob and applied the inverse scaling during dequantization before evaluation.
5. Left the **model architecture, optimizer split, EMA path, late-QAT trigger, and evaluation code** otherwise unchanged.

If you want the exact base behavior back, set `AWQ_ENABLED=0`.

## How to run or evaluate it

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
SWA_ENABLED=1 SWA_EVERY=50 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
AWQ_ENABLED=1 AWQ_ALPHA_CANDIDATES=0.25,0.5,0.75 AWQ_SCALE_LIMIT=4.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults already match the intended base stack for most other knobs (`VE_*`, `LOGIT_SOFTCAP`, `TRAIN_SEQ_LEN=2048`, `TRAIN_BATCH_TOKENS=786432`, etc.).

## Main expected risks and tradeoffs

- **Artifact-size risk:** storing fp16 AWQ scale vectors adds bytes, so any roundtrip-quality win still has to survive the 16 MB budget.
- **Training-time overhead risk:** collecting activation statistics inside the timed loop adds some hook overhead during training, so the quantization win has to justify any slowdown.
- **Heuristic risk:** this is an AWQ-inspired approximation, not full AWQ with a broader search or sparse outlier retention, so the gain may be smaller than the papers suggest.
- **Interaction risk:** the base already uses GPTQ-lite clip search; AWQ scaling could help reconstruction error while hurting downstream compression or vice versa.

## Validation

- `python -m compileall candidates/202604080939_awq-gptq-lite/train_gpt.py`
  - **Passed**
- Safe runtime smoke test:
  - **Not feasible in this runner** because the local Python environment did not have the candidate runtime dependencies available (`torch`, `numpy`, `sentencepiece`, and `flash_attn_interface` were all missing), so an import/forward-pass smoke check would not have tested the candidate itself reliably.

## Suggested next experiments if this helps

1. Add **SpQR-style sparse fp16 outlier retention** only for the few tensors where AWQ scaling wins on MSE but misses on compressed size.
2. Try the same export path on the **LeakyReLU²** stack if you want the strongest non-TTT + training delta combination.
3. Revisit the repo’s dormant **MTP auxiliary heads** if the next bottleneck shifts back from export quality to training-time representation learning.
