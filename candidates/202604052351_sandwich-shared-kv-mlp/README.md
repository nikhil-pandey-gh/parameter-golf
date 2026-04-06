# Sandwich-Shared KV/MLP Banks

## Hypothesis

The repo has already pushed hard on quantization, evaluation, and optimizer engineering. A still-underexplored axis is **parameter reuse without extra compute**: share only the expensive **KV + MLP bank rows** in the middle of the 11-layer stack, keep Q/O and the deepest XSA-heavy layers unique, and reinvest the saved artifact/parameter headroom into a larger BigramHash and one more value-embedding injection layer.

The key twist versus the repo's failed layer-recurrence result is that this candidate does **not** add extra forward passes or effective depth. It reuses weights inside the existing 11-layer compute budget, so step count should stay near the current SOTA regime.

## Why this is promising for this repository

Repository evidence points in three directions:

1. **Capacity per byte keeps winning.** Moving from the baseline to 10L/11L stacks, 3x MLPs, BigramHash, XSA, and VE layers repeatedly improved BPB.
2. **Naive recurrence already failed.** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` reports that doubling depth via recurrence hurt badly because it cut the step budget.
3. **Quantization/export now matters almost as much as training.** If sharing is going to help here, it has to survive export. This candidate therefore quantizes the shared bank tensors directly so the artifact keeps the sharing benefit instead of re-expanding it.

That makes selective sandwich sharing a better fit than full recurrence: it attacks the artifact budget, not the wall-clock budget.

## Prior records that influenced this candidate

- **Primary code base:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - keeps the current strongest stack: LeakyReLU(0.5)^2, parameter-banked Parallel Muon, legal score-first TTT, partial RoPE, XSA, VE, GPTQ-lite-style int6 export.
- **Clean training-side ancestor:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - strongest pre-TTT-style training stack and the best evidence for the stable 11-layer recipe.
- **Bigram/XSA/VE lineage:** `records/track_10min_16mb/2026-03-20_*` and `2026-03-21_*`
  - motivated keeping the deep-layer specialization intact while only sharing middle-bank weights.
- **Important dead end to avoid:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`
  - explicit evidence that compute-costly recurrence is the wrong kind of sharing for this challenge.

## External research that informed it

- **ALBERT** ([arXiv:1909.11942](https://arxiv.org/abs/1909.11942)): cross-layer parameter sharing can preserve strong NLP performance while scaling much better under memory constraints.
- **Subformer** ([arXiv:2101.00234](https://arxiv.org/abs/2101.00234)): for generative transformers, **sandwich-style** sharing is stronger than naive blanket sharing.
- **Sparse Universal Transformer** ([arXiv:2310.07096](https://arxiv.org/abs/2310.07096)): shared-parameter transformer depth remains attractive for parameter efficiency, but compute cost must be controlled.
- **Basis Sharing** ([arXiv:2410.03765](https://arxiv.org/abs/2410.03765)): recent LLM-compression work shows that **matrix-type and layer-selection matter**; selective cross-layer sharing can beat plain compression baselines.
- **CommonKV** ([arXiv:2508.16134](https://arxiv.org/abs/2508.16134)): adjacent-layer sharing is particularly plausible for K/V-style representations because neighboring layers are highly similar.

## What changed versus the chosen base implementation

Relative to the `2026-03-23` record script, this candidate:

1. **Adds configurable bank layouts** for Q/O, KV, and MLP bank tensors.
2. **Defaults to sandwich sharing for KV + MLP only**:
   - `QO_BANK_LAYOUT=""` -> all 11 Q/O bank slots remain unique
   - `KV_BANK_LAYOUT="0,1,2,3,3,4,4,5,6,7,8"`
   - `MLP_BANK_LAYOUT="0,1,2,3,3,4,4,5,6,7,8"`
   - layers 3/4 and 5/6 share KV+MLP banks; first 3 and last 4 layers stay unique
3. **Reinvests some of that headroom** by increasing `BIGRAM_VOCAB_SIZE` from `2048` to `4096`.
4. **Expands VE usage** from `9,10` to `8,9,10`.
5. **Quantizes bank tensors directly** during export so shared banks stay shared in the artifact.
6. **Stores topology metadata in the artifact** (bank layouts plus the relevant sharing/lexical knobs) so non-default ablations can be reloaded correctly.
7. **Adds a FlashAttention fallback** to PyTorch SDPA when `flash_attn_interface` is unavailable, which makes local CPU smoke checks possible.

## How to run / evaluate

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

- Full TTT-style eval stack:

```bash
TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

- Override sharing layouts for ablations:

```bash
KV_BANK_LAYOUT=0,1,2,3,4,5,6,7,8,9,10 \
MLP_BANK_LAYOUT=0,1,2,3,4,5,6,7,8,9,10 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

- Disable the larger lexical sidecar if artifact pressure is lower than expected:

```bash
BIGRAM_VOCAB_SIZE=2048 VE_LAYERS=9,10 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks / tradeoffs

- **Over-regularization:** middle-layer sharing may remove too much specialization and hurt pre-TTT quality.
- **Interaction with TTT is uncertain:** shared KV/MLP banks may adapt more globally during score-first TTT, which could help or hurt.
- **Bigger BigramHash may not fully repay its bytes:** 4096 buckets is a hypothesis, not a tuned optimum.
- **Compression behavior still needs real leaderboard hardware data:** the export path now preserves sharing, but the exact lzma artifact gain still needs an 8xH100 run.

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604052351_sandwich-shared-kv-mlp/train_gpt.py` | Passed |
| Temporary-venv CPU smoke test importing `candidates/202604052351_sandwich-shared-kv-mlp/train_gpt.py`, instantiating a 4-layer toy model, and running `forward` + `forward_logits` | Passed (`cpu_smoke_ok`; scalar loss and `(2, 16, 32)` logits shape verified) |
| Temporary-venv metadata-driven export roundtrip: `state_dict -> mixed_quantize_int6 -> dequantize_mixed_int6 -> infer_eval_model_kwargs(...) -> GPT(...) -> load_state_dict` with intentionally wrong fallback args | Passed (`metadata_roundtrip_ok`; restored Bigram/VE/topology metadata and shared-bank counts `4/3/3`) |

The smoke check used a temporary venv in `/tmp/gh-aw/agent/pgolf-venv` because this runner did not have the repo's Python deps installed up front.
