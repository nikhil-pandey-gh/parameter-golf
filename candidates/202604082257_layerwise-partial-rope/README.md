# Layerwise Partial RoPE

## Hypothesis

The current best training-only stack already wins with **partial RoPE (16/64 dims)**, while recent research suggests that only a small rotary fraction is needed for stable convergence. This candidate keeps the proven 16-dim partial RoPE in early layers, but **shrinks the deepest XSA tail to 8 dims** so later layers spend more capacity on content mixing and less on positional anchoring.

## Why this is promising here

- The repo's strongest training-only record is `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, built around the 11-layer XSA4 + partial-RoPE + LN-scale stack.
- The earlier `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` record showed that **uniform** partial RoPE was already worth a measurable gain.
- The later `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` record improved further, but only by adding substantial evaluation-time complexity. For a new candidate, the cleaner place to search is still the best training-only stack.
- Recent paper evidence now argues that **depth-agnostic RoPE allocation is probably leaving efficiency on the table**.

## Prior repo influence

| Source | What mattered |
|---|---|
| `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` | Established that partial RoPE itself is a real win in this codebase. |
| `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` | Best training-only base; supplied the implementation fork. |
| `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` | Useful upper bound on what extra eval-time complexity buys, which argues for staying training-side here. |
| `candidates/` | No prior candidates existed when this was created. |

## External research that informed this candidate

1. **Fractional Rotation, Full Potential? Investigating Performance and Convergence of Partial RoPE** ([arXiv:2603.11611](https://arxiv.org/abs/2603.11611)) reports that applying RoPE to only a small fraction of dimensions, around 10%, can preserve convergence while saving substantial RoPE cache cost.
2. **Exclusive Self Attention** ([arXiv:2603.09078](https://arxiv.org/abs/2603.09078)) shows that deeper layers benefit from stronger context-focused attention, which fits the idea of reducing positional bias in the XSA tail instead of uniformly across depth.
3. **SliderQuant: Accurate Post-Training Quantization for LLMs** ([arXiv:2603.25284](https://arxiv.org/abs/2603.25284)) emphasizes that shallow and deep layers have different sensitivities, reinforcing the broader point that one shared setting for every layer is often suboptimal.
4. **Enhanced QKNorm normalization for neural transformers with the Lp norm** ([arXiv:2602.05006](https://arxiv.org/abs/2602.05006)) was reviewed as an alternative path, but this candidate stayed closer to the repo's proven partial-RoPE line.

## What changed versus the chosen base

Base fork: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes:

1. Added **depth-aware RoPE allocation** with two new knobs:
   - `ROPE_TAIL_DIMS` (default: half of `ROPE_DIMS`, floored at 8)
   - `ROPE_TAIL_LAYERS` (default: `XSA_LAST_N`)
2. Default schedule for the stock 11-layer candidate is:
   - layers 0-6: `ROPE_DIMS=16`
   - layers 7-10: `ROPE_TAIL_DIMS=8`
3. Logged the effective per-layer RoPE schedule at startup for reproducibility.
4. Changed the default dataset/tokenizer paths to resolve relative to the repository root, so the script can be launched directly from this candidate directory.

## How to run

From the repository root:

```bash
cd candidates/202604082257_layerwise-partial-rope
RUN_ID=layerwise_partial_rope \
XSA_LAST_N=4 ROPE_DIMS=16 ROPE_TAIL_DIMS=8 ROPE_TAIL_LAYERS=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still performs the same training, EMA/SWA application, mixed int6 export, roundtrip reload, and sliding-window evaluation as the base record.

## Expected risks and tradeoffs

- If the 8-dim tail is too aggressive, deeper layers may lose useful relative-position signal and regress on long-context continuation quality.
- The gain, if any, is likely incremental rather than dramatic; this is a "small-bet on a strong stack" candidate.
- The schedule is tuned to the current XSA-on-last-4-layers design; if `XSA_LAST_N` changes, `ROPE_TAIL_LAYERS` should usually move with it.
- This change does not attack the quantizer directly, so any win depends on better learned representations rather than a smaller export gap alone.

## Validation

### Commands run

From the repository root:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604082257_layerwise-partial-rope/train_gpt.py
```

### Outcomes

- `compileall` succeeded for the repository baseline scripts, `data/`, and this candidate's `train_gpt.py`.
- A true CPU smoke run was **not feasible in this runner**: the environment does not have the candidate's Python runtime dependencies installed (`numpy`, `sentencepiece`, `torch`, `flash_attn_interface`), and the script is designed for the CUDA + FlashAttention challenge environment rather than CPU execution.
