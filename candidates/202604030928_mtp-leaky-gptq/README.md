# MTP + LeakyReLU² on 2026-03-22 GPTQ-lite/EMA base

## Hypothesis
Enable the already-wired train-only multi-token prediction (MTP) path by default, while swapping the MLP from ReLU² to LeakyReLU(0.5)². The shared-trunk MTP objective should improve sample efficiency during the fixed 10-minute budget, and the LeakyReLU² change already showed a real gain in this repo.

## Why this is promising for this repo
- The selected 2026-03-22 base is already a strong 11-layer stack: GPTQ-lite, EMA, warmdown3500, late QAT, partial RoPE, XSA, VE, BigramHash, and SmearGate.
- The base code already contains a dormant MTP auxiliary-loss path, with export explicitly excluding `mtp_heads`, so train-time-only MTP can be tested without changing the artifact format.
- The 2026-03-23 LeakyReLU² record showed the activation change can produce a meaningful BPB gain with a tiny code diff.
- This keeps the repo’s proven compression path intact: train a better shared trunk, then export the same GPTQ-lite int6 artifact without MTP heads.

## Repo records that influenced this candidate
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` — chosen base implementation and README.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` — evidence that `LeakyReLU(0.5)^2` improved this architecture family.
- Earlier 11-layer XSA / EMA / partial-RoPE records from 2026-03-20 and 2026-03-21 — supporting evidence for preserving the current backbone rather than changing architecture breadth/depth again.

## External research that informed it
- **arXiv:2404.19737** — multi-token prediction improves sample efficiency using a shared trunk with independent future-token heads. That maps closely onto the repo’s existing dormant `mtp_heads` design.
- **arXiv:2412.19437** — DeepSeek-V3 uses a multi-token prediction training objective, adding more evidence that MTP is a credible train-time objective rather than a one-off trick.
- This repo’s quantization path is already GPTQ-lite, which is directionally inspired by GPTQ/AWQ-style post-training quantization literature; this candidate keeps that export path unchanged.

## What changed versus the chosen base
1. **Turned on train-only MTP by default**
   - `MTP_NUM_HEADS=1` by default.
   - `MTP_LOSS_WEIGHT=0.15` by default.
   - Export still excludes `mtp_heads`, preserving the artifact contents and size path.
2. **Changed the MLP activation**
   - From `ReLU^2` to `LeakyReLU(negative_slope=0.5)^2`.
3. **Tiny usability fallback**
   - Added a minimal fallback path for missing `flash_attn_interface`, using PyTorch SDPA so the script can at least import / smoke more gracefully when FlashAttention 3 is unavailable.
4. **Made default dataset/tokenizer paths repo-root relative**
   - This keeps the script runnable from inside this candidate directory without needing to `cd` back to repo root.

## How to run / evaluate
From this candidate directory:

```bash
cd candidates/202604030928_mtp-leaky-gptq
python train_gpt.py
```

Typical explicit run:

```bash
cd candidates/202604030928_mtp-leaky-gptq
SEED=1337 python train_gpt.py
```

If you want to disable MTP for an ablation:

```bash
cd candidates/202604030928_mtp-leaky-gptq
MTP_NUM_HEADS=0 MTP_LOSS_WEIGHT=0 SEED=1337 python train_gpt.py
```

If FlashAttention 3 is unavailable and you only want a lightweight smoke/import path, the script now has a small SDPA fallback, but full training still expects CUDA.

## Main expected risks / tradeoffs
- Even low-weight MTP adds training FLOPs and optimizer state for the auxiliary heads; if wallclock is the bottleneck, step count may drop slightly.
- MTP may help the shared trunk, but the optimum auxiliary weight could differ from `0.15`; too much weight could hurt next-token loss.
- LeakyReLU² improved a nearby record, but the gain may interact differently with this exact GPTQ-lite/EMA stack.
- The SDPA fallback is intended as a precise low-risk fallback, not as the tuned fast path for production runs.

## Validation
Ran successfully in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604030928_mtp-leaky-gptq/train_gpt.py
```

Attempted, but blocked by runner environment:

```bash
python - <<'PY'
mods = ['torch', 'sentencepiece', 'numpy']
for mod in mods:
    try:
        __import__(mod)
        print(f'{mod}: ok')
    except Exception as e:
        print(f'{mod}: missing ({type(e).__name__}: {e})')
PY
```

Observed limitation:
- The default Python on this runner does not have `torch`, `sentencepiece`, or `numpy`, so an import-level CPU smoke test could not run here.
- Full validation still needs CUDA, the repo datasets, tokenizer, and ideally FlashAttention 3 for performance-faithful runs.
- The clean ablation to run next is `MTP_NUM_HEADS=0 MTP_LOSS_WEIGHT=0` against the same seed.
