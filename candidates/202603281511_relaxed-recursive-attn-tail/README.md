# Relaxed Recursive Attention Tail

## Hypothesis

A **shared attention tail** can improve the repo's strongest banked stack by regularizing the deepest decoder layers without paying a large quality penalty, while also freeing artifact bytes for a slightly larger lexical side channel.

Concretely, this candidate keeps the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` training stack, but replaces the last four attention layers with a **2-slot recursive tail repeated twice**. The attention bank slot map is:

- layers `0..6`: unique attention slots
- layers `7..10`: slots `[7, 8, 7, 8]`

MLP banks, per-layer norms, residual scales, `q_gain`, XSA flags, skip structure, VE scales, and the rest of the training recipe remain layer-specific. That makes this a **relaxed** form of recursion rather than a fully tied recurrent block.

## Why it is promising for this repository

The record history points in two directions at once:

- The best models increasingly concentrate useful modeling changes in the **deep tail**: XSA only in late layers, VE in late layers, partial RoPE + LN scaling, EMA/SWA, and evaluation improvements layered on top of a strong 11-layer core.
- The repo has not yet explored **cross-layer parameter sharing** or **mild recursive tails**, even though recent literature suggests they can work well when the shared part is stabilized and the model keeps lightweight per-depth specialization.

This candidate tries the smallest version of that idea that still matters:

- it shares only the **attention banks** in the deepest layers,
- it preserves all the small per-layer control parameters that can specialize the reused attention core,
- it leaves the MLP path unique to avoid over-constraining capacity,
- and it keeps the fast banked optimizer path from the current top stack.

I also increased the default `BIGRAM_VOCAB_SIZE` from `2048` to `3072`. The top record's own ablation reported that a larger BigramHash helped once the rest of the stack was strong, and the shared attention bank reduces serialized model bytes enough to make that trade more attractive.

## Prior repo work that influenced this candidate

No prior `candidates/` tree was populated in this checkout, so this idea is based on the root baseline plus the existing `records/` history.

The most important influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best current stack
  - Parameter Banking + Parallel Muon
  - LeakyReLU(0.5)^2 MLP
  - late-layer XSA, partial RoPE, LN scale, VE, legal TTT
  - especially important: its ablation that `BIGRAM_VOCAB_SIZE 2048 -> 3072` helped further
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest non-TTT base
  - showed the value of EMA + GPTQ-lite clip search + warmdown tuning on the 11-layer stack
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - partial RoPE and LN scale stabilized the deeper model
  - important because modern shared-depth papers also emphasize stability in repeated blocks
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - established that the deepest layers are where extra attention shaping pays off most in this repo

I did **not** reuse an earlier explicit recurrence run from the repo, because the historical notes suggest heavier looping-depth experiments were not competitive. This candidate is intentionally milder: it keeps the same one-pass 11-layer computation graph and only shares the deepest attention weights.

## External research that informed the design

Primary sources reviewed during implementation:

- **ALBERT** (`arXiv:1909.11942`)
  - classic evidence that cross-layer parameter sharing can preserve quality while reducing parameter count
- **Universal Transformer** (`arXiv:1807.03819`)
  - motivates depth recurrence / repeated computation as a useful inductive bias when done carefully
- **Understanding Parameter Sharing in Transformers** (`arXiv:2306.09380`)
  - argues that the benefit of sharing is driven heavily by better convergence, not just raw FLOPs or parameter count
- **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** (`arXiv:2410.20672`)
  - most directly relevant modern result: tie layers, then preserve flexibility with light depth-specific specialization
- **What Matters in Transformers? Not All Attention is Needed** (`arXiv:2406.15786`)
  - motivates sharing or pruning attention more aggressively than MLPs because attention layers often show high redundancy
- **Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization** (`arXiv:2603.21676`)
  - highlights stability ingredients for repeated-depth computation, especially identity-biased recurrence and LayerScale-like damping

The repo already contains analogues of those stabilizers: LN scaling, residual mixing, zero-init projection paths, and late-layer specialization. That makes a relaxed shared tail more plausible here than in the earlier naive recurrence attempts.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Key changes in this candidate:

1. Added `RECURSIVE_ATTN_TAIL_CORE` and `RECURSIVE_ATTN_TAIL_REPEAT` hyperparameters.
2. Introduced `build_recursive_tail_slot_map(...)` to map the deepest layers onto a smaller set of unique attention bank slots.
3. Changed `GPT` so `qo_bank` and `kv_bank` allocate by **unique attention slot count** rather than by full layer count.
4. Left `mlp_up_bank` and `mlp_down_bank` fully unique across all layers.
5. Updated both training forward paths (`forward` and `forward_logits`) to index attention weights through the slot map.
6. Updated the quantization/export pipeline so it serializes **unique attention slots** instead of duplicating tied layers before quantization, and stores the forward-time model config plus slot map inside the quantized artifact metadata so reload does not depend on matching runtime flags by luck.
7. Increased default `BIGRAM_VOCAB_SIZE` to `3072`.
8. Kept legal TTT code available, but the candidate is intended to evaluate the **training-time architectural change first**, so `TTT_ENABLED` remains off by default.

## How to run / evaluate

Run from this candidate directory so paths and logs stay local:

```bash
cd candidates/202603281511_relaxed-recursive-attn-tail
```

A representative 8xH100 command is:

```bash
RUN_ID=rrat_seed1337 \
SEED=1337 \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=3072 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
RECURSIVE_ATTN_TAIL_CORE=2 RECURSIVE_ATTN_TAIL_REPEAT=2 \
TTT_ENABLED=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If the shared tail looks promising in pre-TTT metrics, a natural follow-up is to rerun the same checkpoint recipe with `TTT_ENABLED=1` and the existing legal score-first TTT path.

## Validation

Commands run while preparing this candidate:

- `python -m compileall candidates/202603281511_relaxed-recursive-attn-tail/train_gpt.py`

Outcomes:

- `compileall` completed successfully.
- A focused code review on the candidate directory completed cleanly after fixing artifact-metadata serialization.

CPU-only smoke execution was **not** run. This script intentionally follows the repo's CUDA evaluation path and hard-requires GPU/CUDA plus `flash_attn_interface`, so a meaningful CPU runtime smoke test would require adding a non-repo fallback path. I left that unchanged to keep the candidate minimal and faithful to the record stack.

## Main expected risks / tradeoffs

- Sharing the deepest attention banks may over-regularize the decoder tail and erase some of the gains from XSA + VE.
- The larger BigramHash may help lexical compression, but it could also become a shortcut that does not pay off once the recursive tail is trained.
- The repo's earlier notes on heavier recurrence were negative; this candidate only partially shares the tail, but it still tests the same general axis.
- Because TTT is left off by default here, the first comparison should be against **pre-TTT** quality from the current best stack.

## Suggested next experiments if this underperforms

1. Keep the same slotting logic, but share only the **last two** attention layers instead of the last four.
2. Keep the recursive tail, but revert `BIGRAM_VOCAB_SIZE` to `2048`.
3. Move the slot map to the **MLP tail** instead of the attention tail.
4. If the training-time score is close, re-enable the existing legal TTT path on top of this architecture.
