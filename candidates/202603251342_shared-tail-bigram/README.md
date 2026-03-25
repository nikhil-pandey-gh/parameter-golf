# Shared Tail Cores + 3K Bigram Budget

## Hypothesis

The current repo frontier has already stacked most of the obvious training and evaluation wins: 11 layers, XSA on the tail, partial RoPE, LN scaling, EMA / SWA, GPTQ-lite-style clip search, and legal score-first TTT. The remaining underexplored axis is **artifact-aware parameter sharing**.

This candidate shares the **last 4 logical transformer layers across 2 reusable bank cores**, while keeping the layer-local controls unique:

- `RMSNorm`s
- residual mixing
- attention / MLP output scales
- skip weights
- XSA placement
- value-embedding scaling

The saved bank bytes are then reallocated into a larger default `BigramHash` table (`3072` buckets instead of `2048`), because prior repo history suggests that bigram-side capacity is one of the more reliable low-cost gains once the 11-layer stack is already strong.

A key detail is that the shared cores are aligned with the existing **XSA tail** (`XSA_LAST_N=4`), so the same shared bank is not reused once with XSA and once without it.

## Why this is promising for this repository

Repo review suggests three stable patterns:

1. Compression-aware design keeps winning. The leaderboard improvements from `2026-03-19` onward repeatedly came from better byte allocation, not just raw pre-quant loss.
2. The best models keep buying extra useful capacity under the same cap: first `MLP3x`, then `10L/11L`, then `BigramHash`, `SmearGate`, `VE128`, and better quantization.
3. True cross-layer sharing does **not** show up in the local record history, even though this challenge is explicitly artifact-limited.

That makes shared tail banks a good next candidate: it is meaningfully different from the existing records, but still adapts the current code instead of requiring a new training stack.

## Prior repo experiments that influenced this candidate

### Most influential base

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - Current repo-best local stack.
  - Contributes the parameter-banked optimizer path, LeakyReLU(0.5)^2 MLP, legal score-first TTT, and the strongest overall evaluation recipe.

### Nearby record lineage

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - Strong static checkpoint recipe with EMA + GPTQ-lite clip search + VE128.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - Established partial RoPE + depth-aware LN scaling as strong zero-parameter improvements.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
  - Showed the importance of late-layer XSA and EMA.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`
  - Important evidence that `BigramHash` and `SmearGate` are worth spending bytes on.

### Dead ends this candidate tries not to repeat

- SwiGLU-style changes that cost too much throughput.
- Pure LR / warmdown tweaks without a new capacity-allocation idea.
- Recurrent/deeper compute-only ideas that trade away too many 10-minute training steps.

## External research that informed it

- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**
  - <https://arxiv.org/abs/1909.11942>
  - Main motivation for cross-layer sharing: keep effective depth while shrinking unique parameter count.
- **Universal Transformer**
  - <https://arxiv.org/abs/1807.03819>
  - Useful framing for treating depth and unique parameters as separate design knobs.

These papers do not map 1:1 onto this repo's training setup, but they support the central hypothesis that late-layer reuse can preserve quality when enough layer-local adaptation remains unique.

## What changed versus the chosen base implementation

Base file: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps that record's overall stack and makes two focused changes:

1. **Shared tail bank mapping**
   - Adds `SHARED_TAIL_LAYERS` and `SHARED_TAIL_CORES`.
   - Default mapping is `4` logical tail layers -> `2` reusable bank cores.
   - The large bank tensors (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`) are allocated only for the unique cores.
   - Forward passes index those banks through `layer_to_core`, while all per-layer control modules remain unique.
   - Quantization/export helpers were updated to unbank/rebank **unique cores**, not logical layers, so shared weights are not serialized multiple times.
   - The quantized artifact now records the sharing layout and XSA tail setting so the reload path can reconstruct the same bank topology explicitly.
   - The code also validates that any shared core stays entirely within either the XSA tail or the non-XSA region.

2. **Larger default bigram budget**
   - `BIGRAM_VOCAB_SIZE` default increases from `2048` to `3072`.
   - This is the concrete byte reallocation enabled by the sharing change.

3. **Candidate-local default paths**
   - Default dataset and tokenizer paths now resolve from the script location back to the repository root.
   - That keeps `cd candidates/202603251342_shared-tail-bigram && torchrun ... train_gpt.py` working without extra path overrides.

## How to run / evaluate

From the repository root:

```bash
cd candidates/202603251342_shared-tail-bigram

BIGRAM_VOCAB_SIZE=3072 \
SHARED_TAIL_LAYERS=4 \
SHARED_TAIL_CORES=2 \
TTT_ENABLED=1 \
TTT_FREEZE_BLOCKS=0 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script keeps the rest of the strongest recent defaults from the chosen base record unless you override them with environment variables.

## Main expected risks / tradeoffs

- **Expressivity risk**: shared late cores may remove too much tail diversity, even with unique norms/scales and skips.
- **TTT interaction risk**: legal TTT now adapts fewer unique bank parameters in the tail; that may regularize adaptation, or it may reduce peak adaptation headroom.
- **Reallocation risk**: the larger bigram table might not fully compensate for the capacity lost by sharing.
- **Placement sensitivity**: sharing the XSA tail is intentionally conservative, but the best tail span / reuse ratio may still need tuning.

## Validation

### Completed on this runner

```bash
python -m compileall candidates/202603251342_shared-tail-bigram/train_gpt.py
```

Outcome: **passed**.

### Attempted but blocked by environment

I attempted a CPU-only import/smoke test with a stubbed `flash_attn_interface` so the candidate could be instantiated without Hopper kernels. That validation could not be completed in this runner because the Python environment does not currently have `torch` installed, even though `torch` is listed in the repository's `requirements.txt`.

```bash
python - <<'PY'
import torch
PY
```

Outcome: **failed with `ModuleNotFoundError: No module named 'torch'`**.

Because the full script also requires CUDA + FlashAttention at runtime, I did not attempt a fake execution path that would misrepresent real startup behavior.
