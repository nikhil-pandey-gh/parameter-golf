# Candidate: Compile-Stable Bank QAT Ramp

## Hypothesis

The current best repository stack already spends most of its budget on the right things: an 11-layer U-Net-like transformer, leaky-ReLU-squared MLPs, efficient evaluation, legal score-first TTT, and artifact-aware int6 export. The remaining gap looks increasingly like a **compression-training mismatch**: the codebase has experimented with late QAT, but the main model weights now live in large 3D parameter banks, while the existing late-QAT path only touches `CastedLinear` layers and previously got constant-folded away under `torch.compile`.

This candidate tests a lightweight fix: **turn on bank-aware int6 STE fake quantization only in late warmdown, ramp its strength from 0 to 1, and explicitly recompile once when the QAT path becomes active** so the compiled graph actually includes it.

## Why this is promising for this repository

Repository evidence points at quantization, not raw architecture, as the most levered remaining axis:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly says its late-QAT path never activated because `torch.compile` constant-folded the flag.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` shows that even tiny quantization/export improvements still move the leaderboard at the current frontier.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` is the strongest current base in this checkout (`val_bpb: 1.1194` mean), so it is the right place to spend the next unit of complexity.

Compared with exporter-only ideas such as AWQ-lite, bank-aware late QAT directly targets the exact mismatch the repo has already uncovered, while still fitting the existing code structure.

## Prior repository work that influenced this candidate

### Chosen base implementation

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps that stack intact:

- 11 layers / 512 dim / 8 heads / 4 KV heads
- parameter banking + Parallel Muon
- Partial RoPE + LN scale + XSA on late layers
- SmearGate + BigramHash + shared value embeddings
- leaky-ReLU(0.5)^2 MLPs
- legal score-first TTT
- int6 + lzma export path

### Earlier records that motivated the change

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - Important because it documents the late-QAT bug clearly.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - Important because the best recent gains are tiny but real quantization/export refinements.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - Important because it is the best current stack and still carries the same late-QAT limitation.

### Prior candidates

There was **no existing `candidates/` directory** in this checkout, so this is the first candidate folder.

## External research that informed the idea

- **LSQ: Learned Step Size Quantization** (`arXiv:1902.08153`, https://arxiv.org/abs/1902.08153)
  - Main takeaway used here: low-bit fake quantization can be introduced directly into training with a straight-through path rather than only relying on post-training export.
- **QDrop** (`arXiv:2203.05740`, https://arxiv.org/abs/2203.05740)
  - Main takeaway used here: abrupt low-bit constraints can be brittle; gentler exposure to quantization noise can improve robustness. That motivates the **0 -> 1 QAT ramp** instead of a hard instantaneous switch.
- **OmniQuant** (`arXiv:2308.13137`, https://arxiv.org/abs/2308.13137)
  - Main takeaway used here: differentiable low-bit calibration remains useful even when PTQ is already strong, especially in the low-bit regime.
- **AWQ** (`arXiv:2306.00978`, https://arxiv.org/abs/2306.00978)
  - Considered as the main candidate, but not chosen for this folder. Activation-aware export still looks promising, yet the repo-specific evidence around the broken/partial late-QAT path made bank-aware QAT the stronger first follow-up.

## What changed versus the base implementation

Relative to `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

1. Adds a shared `fake_quantize_int6_ste(...)` helper for late-training int6 fake quantization.
2. Extends late fake quantization from tiny `CastedLinear` layers to the **actual 3D parameter banks** used by attention and MLP weights.
3. Adds a **bank-QAT mix buffer** that ramps from `0.0` to `1.0` during the late phase instead of applying full fake quantization immediately.
4. **Recompiles once** when late QAT becomes active so `torch.compile` cannot keep using the no-QAT graph.
5. Keeps small `CastedLinear` late QAT synchronized with the same late-phase activation.
6. Adds a FlashAttention fallback path using `scaled_dot_product_attention` so the module can be imported in environments without `flash_attn_interface` once PyTorch is present.

## Expected benefits

- Better alignment between training-time weights and the final int6 export target.
- A repo-specific fix for the already-documented no-op late-QAT path.
- Late-only quantization overhead instead of paying the fake-quant cost for the full 10-minute run.

## Main risks and tradeoffs

- The late recompile and fake-quant math may reduce step count enough to erase any quantization win.
- This is a **lightweight STE approximation**, not full learned-scale LSQ or full OmniQuant-style differentiable clipping.
- If the late phase starts too late, the gain may be negligible; if it starts too early, pre-quant quality may degrade.
- The best observed repo metrics still depend heavily on evaluation protocol and TTT, so any training-side gain here is likely incremental rather than dramatic.

## How to run

From this directory:

```bash
cd candidates/202603291617_bank-qat-ramp
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
QAT_ENABLED=1 BANK_QAT_ENABLED=1 LATE_QAT_THRESHOLD=0.20 BANK_QAT_CLIP_RANGE=31 \
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

Notes:

- The candidate keeps the record stack's intended target: CUDA training, Hopper-friendly attention, and post-training int6 export.
- The new late-QAT path activates automatically once the learning-rate multiplier falls below `LATE_QAT_THRESHOLD`.
- Setting `LATE_QAT_THRESHOLD=0` turns the candidate into an immediate-QAT ablation instead of the default late-ramp behavior.

## Validation

Validation run in this workflow:

- `python -m compileall candidates/202603291617_bank-qat-ramp/train_gpt.py`
  - **Passed**.
- Runtime smoke import attempt:
  - Attempted to import the candidate module and exercise the new helper functions.
  - **Blocked** because the local workflow Python environment does not currently have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
  - As a result, no meaningful runtime smoke test was possible without installing heavyweight dependencies not already present in this workflow environment.

## Files added

- `candidates/202603291617_bank-qat-ramp/train_gpt.py`
- `candidates/202603291617_bank-qat-ramp/README.md`
