# Candidate: LeakyReLU^2 + Calibration-Guided Mixed-Precision Export

## Hypothesis

The strongest **no-TTT** stack in this repository is already very close to the 16 MB ceiling, so the next gain is more likely to come from **better byte allocation at export time** than from a larger architectural rewrite.

This candidate combines two ideas:

1. **LeakyReLU(0.5)^2** in the MLP, which already produced a documented improvement in the current repo-best record.
2. A **calibration-guided, budget-aware mixed-precision export** that starts from an aggressive `int5/int6` policy and only upgrades stage groups that measurably reduce calibration loss while staying under the exact artifact budget.

The core bet is that the repo's recent wins already show quantization sensitivity is highly non-uniform, and that a tiny amount of export-time search can buy back more BPB than a fixed one-size-fits-all quantization recipe.

## Why this is promising for this repository

Repo history suggests that the challenge is now dominated by **compression-aware quality retention**, not by discovering a completely different backbone:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed that improving **post-training quantization quality** alone still moved the frontier.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` reported an additional gain from **LeakyReLU(0.5)^2** on top of an already strong stack.
- Earlier records showed that **int5/int6 mixes**, **FP16 embeddings**, and **careful export choices** are all meaningful levers under the 16 MB cap.

This candidate therefore stays on the proven 11-layer XSA/Partial-RoPE/EMA/GPTQ-lite path and changes only the parts most likely to matter next: the MLP activation and the export allocator.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Direct feature donor:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Additional export/byte-budget influences:

- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`

## External research that informed it

This candidate is mainly inspired by recent PTQ work showing that quantization error is **not uniformly distributed** across weights and layers, and that light calibration can be used to spend precision where it matters most:

- **GPTQ** — Frantar et al., 2022: one-shot GPT quantization with better low-bit retention via stronger PTQ methodology.  
  <https://arxiv.org/abs/2210.17323>
- **SmoothQuant** — Xiao et al., 2022/2024: quantization difficulty can be redistributed instead of treated uniformly.  
  <https://arxiv.org/abs/2211.10438>
- **AWQ** — Lin et al., 2023/2024: only a small subset of weights/channels are especially salient, so protecting them disproportionately helps.  
  <https://arxiv.org/abs/2306.00978>
- **QuaRot** — Ashkboos et al., 2024: outlier-aware transformations matter for low-bit LLM PTQ.  
  <https://arxiv.org/abs/2404.00456>
- **SpinQuant** — Liu et al., 2024/2025: learned or selected quantization-friendly transforms meaningfully improve low-bit accuracy, reinforcing the broader lesson that export quality depends on where precision is spent.  
  <https://arxiv.org/abs/2405.16406>

This candidate does **not** implement rotations or activation transforms directly. Instead, it adapts the more repo-compatible lesson: **under a strict artifact budget, treat precision as a scarce resource and allocate it by sensitivity rather than by fixed tensor category alone**.

## What changed versus the chosen base implementation

Starting point: the `2026-03-22` GPTQ-lite record.

Changes in `train_gpt.py`:

1. **MLP activation changed from `relu^2` to `LeakyReLU(0.5)^2`.**
2. Added export controls:
   - `TOTAL_SIZE_BUDGET`
   - `EXPORT_SEARCH_ENABLED`
   - `EXPORT_CALIBRATION_BATCHES`
   - `EXPORT_CALIBRATION_BATCH_SEQS`
3. Replaced the fixed `int6(attn/mlp) + int8(other)` export path with a **stagewise mixed-precision search**:
   - start from a more aggressive base policy,
   - treat early/mid/late attention and MLP groups separately,
   - allow higher precision for embeddings and auxiliary projections,
   - greedily upgrade only groups that improve calibration CE loss while remaining inside the size budget.
4. Renamed the export artifact path to `final_model.intmix.ptz` and kept roundtrip/sliding evaluation intact.
5. Added a hard runtime check that fails if the final export exceeds the configured total-size budget.

## Export policy design

The search groups large tensors into a few budget-relevant buckets:

- `tok_embed`
- `bigram_embed`
- `value_embed`
- `aux_proj`
- `early/mid/late_attn`
- `early/mid/late_mlp`
- `other`

The default starting policy is intentionally aggressive:

- early/mid attention + MLP: `int5`
- late attention + MLP: `int6`
- embeddings: `int8`
- auxiliary projections: `int6`

The script then greedily upgrades one group at a time (for example `int5 -> int6`, or `int8 -> fp16` for selected embedding groups) if the upgrade reduces calibration loss and still fits the exact byte budget.

## How to run

Run from this candidate directory. The script now resolves its default dataset and tokenizer paths relative to the repository root, so `DATA_PATH` and `TOKENIZER_PATH` do not need to be overridden just to launch from here.

```bash
cd candidates/202603282244_leaky-calmix-export

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
TOTAL_SIZE_BUDGET=16000000 EXPORT_SEARCH_ENABLED=1 \
EXPORT_CALIBRATION_BATCHES=6 EXPORT_CALIBRATION_BATCH_SEQS=8 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to compare against a fixed policy without the search, set:

```bash
EXPORT_SEARCH_ENABLED=0
```

## How to evaluate

The script keeps the same built-in evaluation flow as the base record:

- post-training EMA diagnostic eval,
- quantized roundtrip eval,
- sliding-window eval at `EVAL_STRIDE`,
- optional stride-64 eval if requested stride differs.

Look for the final logs:

- `final_intmix_roundtrip_exact`
- `final_intmix_sliding_window_exact`
- `final_intmix_sliding_window_s64_exact`

## Validation run for this candidate

Lightweight validation completed in this workflow:

From the repository root:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603282244_leaky-calmix-export/train_gpt.py
```

From inside the candidate directory itself:

```bash
python -m compileall train_gpt.py
```

Outcome:

- baseline repository Python syntax check from the repo root: **passed**
- candidate `train_gpt.py` syntax check from the repo root: **passed**
- candidate `train_gpt.py` syntax check from inside the candidate directory: **passed**

Safe smoke check status:

- A real CPU/GPU startup smoke test was **not feasible in this runner** because the environment was missing multiple import-time runtime dependencies used by this script, including `torch`, `numpy`, `sentencepiece`, and `flash_attn_interface`. As a result, this workflow stayed on the repo's existing lightweight syntax-validation path rather than attempting heavyweight dependency setup.

## Main expected risks / tradeoffs

1. **Calibration overfitting risk**: the export search only sees a small calibration subset, so it may choose upgrades that do not transfer perfectly to the full validation set.
2. **Extra eval-time overhead**: the search adds a few additional dequantize-and-score passes during export.
3. **Stagewise grouping is coarse**: a tensor-level or channel-level allocator could be better, but this version keeps code and runtime complexity modest.
4. **LeakyReLU^2 may not add linearly** on top of the GPTQ-lite stack; the repo shows it helps on a stronger TTT stack, but exact interaction here is still uncertain.

## Suggested next experiments

1. Compare `EXPORT_SEARCH_ENABLED=1` vs `0` on the same seed to isolate the allocator effect.
2. Test a slightly larger calibration set, e.g. `EXPORT_CALIBRATION_BATCHES=8`.
3. Allow `tok_embed` and `value_embed` to be upgraded independently to see whether FP16 bytes are better spent on one of them first.
4. If this direction is promising, move from **stagewise group search** to **per-tensor search** or a tiny **saliency-protected** variant inspired by AWQ.
