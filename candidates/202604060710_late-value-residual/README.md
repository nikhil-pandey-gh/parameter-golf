# Late Value Residual on the 03-23 stack

## Hypothesis

The current best Parameter Golf models keep winning with *cheap, deep-layer-only information-flow tweaks* (XSA on late layers, partial RoPE, LN scale, LeakyReLU^2, EMA/GPTQ-lite on top). A **ResFormer-style value residual** should fit that pattern: preserve token identity from the first attention layer, but only re-inject it into the **later** blocks where information loss is most likely to matter.

This candidate therefore tests a **late, normalized value residual** on top of the current best stack:

- keep the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` training/eval recipe,
- turn on value residuals by default,
- apply them only from **layer 6 onward**,
- and constrain the learned mix with a softmax so the current layer still dominates early in training.

## Why this is promising for this repo

- The strongest records already show that **small architectural edits with near-zero byte cost** can still move the needle: XSA on the last layers, partial RoPE, LN scale, GPTQ-lite clip search, and LeakyReLU^2 all improved the same 11-layer core stack.
- Repo evidence says **depth recurrence** and **SwiGLU** are poor fits for the 10-minute budget, while **QAT** has both overhead and a previously documented late-QAT compile bug. That makes a cheap modeling tweak more attractive than a more invasive quantization rewrite for the next iteration.
- Value residuals are especially plausible here because the current stack already leans on **value-side tricks** (`VE_ENABLED`, GQA/XSA on late layers, legal TTT at eval). Reusing first-layer values is a natural continuation of that direction.

## Prior repo experiments that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Training-only cleaner predecessor:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Late-layer structural tweaks that worked:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **XSA as the template for “apply only in later layers”:** `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
- **Negative results worth avoiding:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` (SwiGLU and depth recurrence), `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` (recurrence)

There were **no prior `candidates/` directories** in this repo before this candidate.

## External research that informed the choice

- **ResFormer / SVFormer** — <https://arxiv.org/abs/2410.17897>  
  The paper argues that standard hidden-state residuals do not fully preserve token-level information in deep transformers, and reports that adding **value residual connections** can match validation quality with fewer parameters and less training data. That is the direct motivation for reusing first-layer values here.
- **AWQ** — <https://arxiv.org/abs/2306.00978>  
  Useful reminder that low-bit weight quality is still a major bottleneck for small models, but this repo has already explored that direction aggressively with GPTQ-lite and mixed int5/int6 export.
- **GPTQ** — <https://arxiv.org/abs/2210.17323>  
  Reinforced that weight-only quantization is important, but also highlighted that another quantization-only candidate would likely be more incremental than a still-unexplored modeling change.
- **LSQ / PACT** — <https://arxiv.org/abs/1902.08153>, <https://arxiv.org/abs/1805.06085>  
  These were the strongest alternative ideas considered. I deferred them for this candidate because the current best banked script would need a more invasive late-QAT rewrite, while the repo already documents QAT overhead and one late-QAT constant-folding failure.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Self-contained path defaults**: data/tokenizer defaults resolve relative to the candidate file, so the script can be run from inside this candidate directory.
2. **Value residual enabled by default**: `VALUE_RESIDUAL=1`.
3. **Late-only application**: `VALUE_RESIDUAL_START_LAYER=6` so only the deeper half of the 11-layer stack mixes in first-layer values.
4. **Normalized residual mix**: the two learned coefficients are passed through `softmax`, preventing unstable unconstrained scaling.
5. **Current-layer-biased init**: the initial logits favor the current layer's value stream while still injecting a smaller first-layer residual.
6. **First-layer value capture without full-depth mixing**: the script now captures the seed value stream even when early layers are not themselves using the residual.

## How to run

First make sure the repo-level cached dataset and tokenizer exist:

```bash
cd /path/to/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

Then run from this directory:

```bash
cd candidates/202604060710_late-value-residual

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VALUE_RESIDUAL=1 VALUE_RESIDUAL_START_LAYER=6 \
LATE_QAT_THRESHOLD=0.15 \
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

The candidate defaults resolve `DATA_PATH` and `TOKENIZER_PATH` against the repo root, so no extra path flags are needed once that cache exists.

For a cleaner training-only ablation against the 03-22/03-23 family, set `TTT_ENABLED=0`.

## Main risks / tradeoffs

- The first-layer value stream may be **too low-level** for later layers and could fight the existing `VE_ENABLED` path or late-layer XSA.
- `VALUE_RESIDUAL_START_LAYER=6` is a heuristic. The best setting may be later (`7-8`) or even all late decoder layers only.
- The normalized mix is safer than unconstrained coefficients, but it may also cap upside if the model wants a stronger first-layer reuse ratio.
- This change targets **model quality**, not export format, so it may improve pre-quant loss more than final compressed BPB.

## Validation

- `python -m compileall train_gpt.py` from this candidate directory (equivalently `python -m compileall candidates/202604060710_late-value-residual/train_gpt.py` from repo root) — **passed**
- `python -m compileall train_gpt.py train_gpt_mlx.py data` from repo root — **passed**
- Minimal CPU-only runtime smoke test — **not feasible in this runner** because `nvidia-smi` is unavailable here and this script hard-requires a CUDA + `flash_attn_interface` runtime path before training can start
