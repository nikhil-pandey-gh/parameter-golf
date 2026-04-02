# Activation-Aware GPTQ-lite on the 11L EMA Stack

## Hypothesis

The strongest open training stack in this checkout already looks close to an architectural plateau, but its export path is still mostly **weight-error-driven**. This candidate adds a small **activation-aware reparameterization** step before the existing GPTQ-lite int6 export:

1. collect per-channel RMS stats from a few **train-set** calibration batches after EMA,
2. build per-block input scales for the attention and MLP branches,
3. divide the branch activations by those scales at inference time,
4. multiply the matching input columns of `q/k/v` and `mlp.fc` weights by the same scales,
5. run the existing GPTQ-lite percentile search on the reparameterized weights.

Because the transform is exact before quantization, the candidate is trying to **spend a few extra control-vector bytes to reduce quantization loss**, not to change the learned function.

## Why this is promising for this repo

Repository evidence points in a consistent direction:

- early gains came from **evaluation** (`SlidingWindowEval`),
- then from **larger-but-compressible models** (11L, 3x MLP, mixed int5/int6),
- then from **smaller architectural refinements** (XSA, Partial RoPE, LN scale, VE, LeakyReLU^2),
- and the latest non-TTT training gains came from **better export quality** (`GPTQ-lite`, EMA, tighter warmdown).

That makes export-time error a strong remaining bottleneck. Unlike full QAT, this candidate adds **no extra backward pass**, **no extra training objective**, and only a tiny post-training calibration sweep.

## Prior repo work that influenced this candidate

- **Root baseline**: `train_gpt.py` provides the core tokenizer-agnostic BPB setup, Muon/Adam split, and clean export/eval loop.
- **Chosen base implementation**: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
  - strongest pre-TTT training stack in this checkout,
  - already has 11L + XSA4 + Partial RoPE + LN scale + SmearGate + BigramHash + VE + EMA,
  - already uses a GPTQ-lite-style percentile search for int6 export.
- **Current best record**: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - confirms that small deltas still matter once the overall stack is mature, but it spends its extra complexity mostly in eval-time adaptation.

## External research that informed it

- **SmoothQuant** ([arXiv:2211.10438](https://arxiv.org/abs/2211.10438)) shows that activation statistics can be used to migrate quantization difficulty into weights with an exact reparameterization.
- **AWQ** ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)) shows that activation-aware scaling protects salient channels better than weight-only heuristics.
- **GPTQ** ([arXiv:2210.17323](https://arxiv.org/abs/2210.17323)) motivates preserving the existing low-bit export path rather than replacing it wholesale.

This candidate intentionally keeps the idea smaller than full SmoothQuant/AWQ: it adds **per-block attention/MLP input scales** and keeps the repo’s existing GPTQ-lite search instead of introducing a new quantizer.

## What changed vs the chosen base implementation

1. **Candidate-friendly defaults**
   - `DATA_PATH` and `TOKENIZER_PATH` now default relative to the repository root, so the script can be launched from this candidate directory.
   - If `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA instead of failing at import time.
2. **Post-EMA activation calibration**
   - new knobs: `ACT_CALIB_STEPS`, `ACT_SCALE_ALPHA`, `ACT_SCALE_MAX`, `ACT_SCALE_EPS`.
   - default calibration uses 8 train batches.
3. **Per-block export control vectors**
   - each block now carries `attn_in_scale` and `mlp_in_scale`.
4. **Exact branch reparameterization before quantization**
   - scale columns of `c_q`, `c_k`, `c_v`, and `mlp.fc`,
   - divide the matching normalized branch inputs by the stored scale vectors,
   - then quantize the reparameterized model with the existing GPTQ-lite int6 path.
5. **Diagnostic eval after activation-aware scaling**
   - the pre-export diagnostic now runs on the post-EMA, post-scale model.

## How to run

From this candidate directory:

```bash
cd candidates/202604021510_act-aware-gptq-lite
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful export ablations:

```bash
ACT_CALIB_STEPS=4 ACT_SCALE_ALPHA=0.35 ACT_SCALE_MAX=3.0
ACT_CALIB_STEPS=8 ACT_SCALE_ALPHA=0.50 ACT_SCALE_MAX=4.0
ACT_CALIB_STEPS=16 ACT_SCALE_ALPHA=0.65 ACT_SCALE_MAX=6.0
```

If your cached data lives elsewhere, override:

```bash
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model
```

## Validation in this workspace

- `python -m compileall candidates/202604021510_act-aware-gptq-lite/train_gpt.py` -> passed
- A runtime smoke test was **not feasible in this workflow container** because the local Python environment here does not include `torch`, `numpy`, or `sentencepiece`. Even with those packages present, a full script startup would still require cached FineWeb shards, a SentencePiece model, and CUDA.

## Main risks / tradeoffs

- The calibration pass might be too small or too biased, so scales learned from a few train batches may not help the full validation distribution.
- If `ACT_SCALE_ALPHA` or `ACT_SCALE_MAX` is too aggressive, the transform may make weights less compressible even if it reduces some outlier channels.
- This is deliberately narrower than full AWQ/SmoothQuant; if it helps, the next step is a more careful shared-scaling search or mixed-precision allocation on top of these scales.
