# Candidate: Partial Heavy Sharing on the 11L GPTQ-lite Base

## Hypothesis

The current best non-TTT stack is already close to the 16 MB artifact limit, so the next useful architectural move is not more recurrent depth or a broader rewrite. Instead, this candidate shares the **heavy attention+MLP cores only in early layers**, where representations are more lexical and redundant, then reinvests the saved artifact budget into a larger **BigramHash** side channel.

Concretely, layers **0-1**, **2-3**, and **5-6** share adjacent heavy cores, while layer **4** and layers **7-10** stay unique. That yields **8 unique heavy cores for 11 layer wrappers** with the same depth and avoids sharing across the model's encoder/decoder midpoint.

## Why this is promising here

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` is the strongest clean **non-TTT** base in the repo, so it is the best place to test an architectural compression idea without mixing in TTT or parameter-banked optimizer changes.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` reports that **layer recurrence x2** was actively bad. This candidate avoids repeated extra compute and instead shares parameters across the existing depth.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` reports a helpful **BigramHash 2048 -> 3072** ablation, so this candidate uses the saved bytes to raise the default bigram bucket count to **3072**.

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

- **MobileLLM / MobileLLM-LS** (Liu et al., 2024): <https://arxiv.org/abs/2402.14905>  
  The paper argues that sub-billion LMs are unusually architecture-sensitive and reports gains from **immediate block-wise weight sharing** with no model-size increase.
- **ALBERT** (Lan et al., 2019): <https://arxiv.org/abs/1909.11942>  
  Cross-layer sharing is an established way to lower parameter count while preserving useful depth.

These papers point in the same direction: when parameters are the bottleneck, reuse weights selectively instead of spending bytes uniformly everywhere.

## Chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Why this base:

- It already contains the repo's strongest non-TTT recipe: 11L, XSA, Partial RoPE, LN scaling, EMA, GPTQ-lite export, BigramHash, VE, SmearGate.
- It is much simpler than the 2026-03-23 Parallel-Muon + legal-TTT record, so the effect of sharing is easier to interpret.

## What changed versus the base

1. **Added early-layer heavy-weight sharing**
   - New knobs: `SHARE_HEAVY_UP_TO` (default `7`) and `SHARE_GROUP` (default `2`)
   - Heavy cores are stored once in `shared_heavy_cores`
   - Lightweight per-layer wrappers keep their own:
     - norms
     - residual mix
     - attn/mlp output scales
     - layer-index-based LN damping
   - Sharing is applied separately inside the encoder and early decoder so the U-Net midpoint stays unique
   - Default core map for 11 layers: `[0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7]`

2. **Reinvested bytes into lexical features**
   - `BIGRAM_VOCAB_SIZE` default: `2048 -> 3072`

3. **Left the compute path intact**
   - Same depth
   - Same FlashAttention-based attention path
   - Same EMA + GPTQ-lite export flow
   - Same XSA placement on the last 4 layers, which remain unshared by default

4. **Disabled late-QAT by default**
   - `LATE_QAT_THRESHOLD` now defaults to `0.0`
   - Repo review showed a prior late-QAT path was compile-sensitive and had already been identified as a likely no-op in earlier records, so this candidate isolates the sharing hypothesis instead of inheriting that ambiguity

## How to run

From this directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate defaults already enable the sharing setup:

```bash
BIGRAM_VOCAB_SIZE=3072
SHARE_HEAVY_UP_TO=7
SHARE_GROUP=2
LATE_QAT_THRESHOLD=0.0
```

To turn sharing off and recover a closer baseline from this file:

```bash
SHARE_HEAVY_UP_TO=0 SHARE_GROUP=1 BIGRAM_VOCAB_SIZE=2048 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate

The script keeps the same export/eval flow as the base record:

1. train with EMA
2. apply EMA weights
3. export `final_model.int6.ptz` via GPTQ-lite-style mixed quantization
4. run roundtrip eval
5. run sliding-window eval (default stride 64)

The main comparison to watch is the final sliding-window BPB against the 2026-03-22 base.

## Main risks / tradeoffs

- **Capacity loss in early layers**: adjacent sharing could oversimplify lower-layer feature extraction and cancel out the larger bigram table.
- **No throughput gain**: this is an artifact-budget trade, not a training-speed optimization. FLOPs are roughly unchanged.
- **Shared control within the heavy core**: the attention core, including its internal `q_gain`, is shared for paired layers, so layer individuality is reduced more than a pure matrix-only share.
- **Quantization could dominate anyway**: if the remaining bottleneck is export error rather than parameter allocation, the gain may be small.

## Validation

Ran:

```bash
python -m compileall train_gpt.py
python -m compileall ../../train_gpt.py ../../train_gpt_mlx.py ../../data train_gpt.py
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

Outcomes:

- `compileall` passed for the candidate script.
- The broader low-cost repository `compileall` check also passed.
- A runtime import smoke test was **not feasible in this environment** because the Python environment is missing repository dependencies (`ModuleNotFoundError: numpy`) before the script can even reach the repo's CUDA/FlashAttention runtime requirements.
