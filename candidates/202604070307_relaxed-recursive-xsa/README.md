# Relaxed Recursive XSA

## Hypothesis

The late decoder in the current 11-layer XSA/EMA/GPTQ-lite stack is probably overparameterized relative to the rest of the model. If the last few decoder layers reuse **one shared late block** and recover specialization with **tiny per-depth adapters**, the model may keep most of the late-layer benefit while spending fewer artifact bytes on duplicated weights.

## Why this is promising here

Repository review pointed to a narrow opening:

- The strongest **train-only** base is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`.
- The strongest recent gains are mostly from **quantization/export**, **EMA/SWA**, **XSA**, **partial RoPE**, **LN scale**, and later **TTT**, not from a fresh compact architecture.
- The repo also documents that **naive full layer recurrence** is a bad fit for the 10-minute cap because it reduces optimizer steps too much.

This candidate therefore tries a narrower version of recurrence: **share only the late decoder block**, keep the logical depth unchanged, and use lightweight per-layer adapters so the reused block can still specialize.

## Prior records and experiments that influenced this candidate

- **Chosen base:** `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - best clean pre-TTT stack in the repo
  - already includes XSA, EMA, partial RoPE, LN scale, VE, SmearGate, BigramHash, and GPTQ-lite export
- **`2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`**
  - clarified that partial RoPE + LN scale mattered, while the submitted late-QAT path did not
- **`2026-03-19_SlidingWindowEval`**
  - contains dormant looped/LoRA code paths that showed how to thread shared-block adapters through this codebase
- **`2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`** and **`2026-03-18_FP16Embed_WD3600`**
  - both report that blunt layer recurrence / looping is net-negative under a strict wall-clock budget
- **`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`**
  - its ablation includes a modest gain from a larger BigramHash table, which made that a natural place to reinvest some recovered artifact budget

## External research that informed it

- **ALBERT** (Lan et al., 2019, arXiv:1909.11942): cross-layer parameter sharing can cut parameters heavily without destroying quality.
- **Universal Transformers** (Dehghani et al., 2018, arXiv:1807.03819): recurrent depth is useful when applied selectively and with the right inductive bias.
- **Understanding Parameter Sharing in Transformers** (Lin et al., 2023, arXiv:2306.09380): sharing helps partly because it can improve convergence, not just because it reduces parameter count.
- **Relaxed Recursive Transformers** (Bae et al., 2024/2025, arXiv:2410.20672): shared blocks recover much of their lost flexibility when paired with small depth-wise low-rank adapters.
- **Looped Transformers are Better at Learning Learning Algorithms** (Yang et al., 2023/2024, arXiv:2311.12424): strong evidence that iterative/shared-depth transformers can match standard models with far fewer unique parameters.

## What changed vs the chosen base implementation

1. **Late shared decoder block**
   - The final 3 logical decoder layers now reuse a single shared block.
   - Logical depth stays at 11; only the number of unique late blocks is reduced.

2. **Per-depth lightweight adapters**
   - Each reused late layer gets its own:
     - residual-mix vector,
     - attention scale vector,
     - MLP scale vector,
     - LN rescale scalar,
     - low-rank attention residual adapter,
     - low-rank MLP residual adapter.
   - This is the “relaxed recursive” part of the candidate.

3. **Recovered-byte reinvestment**
   - `BIGRAM_VOCAB_SIZE` now defaults to `3072` instead of `2048`.
   - `VE_LAYERS` now defaults to `8,9,10` so token-identity reinjection covers the full reused late stack.

4. **Run-from-candidate and smoke-test ergonomics**
   - Dataset/tokenizer defaults now resolve from the repository root, so the script can be launched from this candidate directory directly.
   - FlashAttention import is optional; if unavailable, attention falls back to PyTorch SDPA instead of requiring the external kernel package. This keeps CPU-side import/smoke paths possible in a properly provisioned environment.

## How to run

From this directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The key candidate-specific defaults are:

```bash
NUM_LAYERS=11
SHARED_DECODER_LAYERS=3
SHARED_ADAPTER_RANK=16
BIGRAM_VOCAB_SIZE=3072
VE_LAYERS=8,9,10
XSA_LAST_N=4
ROPE_DIMS=16
LN_SCALE=1
```

`XSA_LAST_N` should stay at `0` or at least `SHARED_DECODER_LAYERS`, since the shared late block carries one XSA setting across all of its repeated applications.

You can still override `DATA_PATH` and `TOKENIZER_PATH`, but by default they resolve to the repo-root `data/` paths.

## Expected risks and tradeoffs

- Sharing the last 3 decoder layers may remove too much late-layer specialization even with adapters.
- The adapter path adds extra small parameters that are easy to serialize but may still be too weak or too strong depending on rank.
- Bigger BigramHash may not repay the bytes reclaimed from late-layer sharing.
- This candidate does **not** try to win via more eval compute; it is intentionally focused on train-time architecture and export.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202604070307_relaxed-recursive-xsa/train_gpt.py
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('candidates/202604070307_relaxed-recursive-xsa/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PY
```

Observed outcomes:

- `compileall` **passed**
- the tiny CPU import/forward smoke attempt was **not feasible in this runner environment** because the installed Python environment did not include the repo runtime dependencies (`torch`, `numpy`, `sentencepiece`)

That means syntax was validated here, but a real runtime smoke still needs an environment with the declared repo dependencies installed.
