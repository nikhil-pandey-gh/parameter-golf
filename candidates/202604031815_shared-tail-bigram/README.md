# Shared Tail Banks + Bigger Bigram

## Hypothesis

The strongest stack in this repo already concentrates most of its special modeling tricks in the last four layers: XSA, partial RoPE, LN scaling, and late value reinjection. An **ALBERT-style shared tail** should let those late layers keep their 11-step compute path while storing fewer unique late-layer matrices, and the recovered artifact budget can be reallocated into a larger **BigramHash** table that improves short-range token transition modeling.

The key distinction from the repo's earlier failed recurrence experiments is that this candidate **does not add extra depth or extra compute**. It keeps the same 11 logical layers and the same training loop, but only stores 9 unique late-layer bank slices via the logical map:

```python
[0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 8]
```

## Why this is promising here

- The repo's best record (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`) already uses **banked weights**, so late-layer sharing can be implemented surgically without changing the overall training recipe.
- The non-record recurrence exploration in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` found that recurrence failed because it **halved the number of optimizer steps under a fixed wallclock budget**. This candidate avoids that failure mode by sharing storage only.
- The latest best record also reports that a larger bigram table helped; this candidate uses part of the saved tail-bank budget to move **BigramHash from 2048 to 3072 buckets** by default.

## Influential prior work in this repository

- **Primary implementation base:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest overall architecture in the repo
  - banked parameter layout makes storage-efficient sharing straightforward
- **Clean pre-TTT reference:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strong evidence that the 11L/XSA/partial-RoPE/LN-scale/VE path is a good non-TTT base
- **Explicit dead-end to avoid:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - reported that doubling depth with recurrence was worse because compute increased and step count fell
- **Prior candidates:** none existed in `candidates/` when this candidate was created

## External research that informed it

- **ALBERT** — Lan et al., 2019, [arXiv:1909.11942](https://arxiv.org/abs/1909.11942)
  - motivates cross-layer parameter sharing as a way to reduce stored parameters without discarding depth entirely
- **Universal Transformer** — Dehghani et al., 2018, [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)
  - motivates reusing a shared transformation across multiple logical depth steps
- **BitNet / BitNet b1.58** — Ma et al., 2023/2024, [arXiv:2310.11453](https://arxiv.org/abs/2310.11453), [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
  - useful as a reminder that more aggressive compression-aware training exists, but those approaches would require substantially more infrastructure change than this repo-friendly bank-sharing variant

## What changed versus the chosen base

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`:

1. Added **shared tail bank mapping** via `SHARED_TAIL_LAYERS` and `SHARED_TAIL_GROUPS`.
2. Kept all 11 logical blocks, per-layer norms, per-layer residual mixing, per-layer Q gains, and skip structure intact.
3. Reduced the number of stored late bank slices so quantization/export keeps the storage win instead of re-expanding duplicated shared weights.
4. Changed the default **BigramHash** size from **2048 -> 3072**.
5. Changed default data/tokenizer path resolution so the script auto-discovers the repo root when launched from **either this candidate directory or a future `records/...` location**.
6. Saved the shared-tail layout inside exported artifacts so non-default sharing runs stay self-describing.

## How to run or evaluate

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful explicit overrides:

```bash
BIGRAM_VOCAB_SIZE=3072 SHARED_TAIL_LAYERS=4 SHARED_TAIL_GROUPS=2 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script writes the normal training log plus `final_model.pt` and the quantized `final_model.int6.ptz` artifact in the working directory.

## Main risks / tradeoffs

- Sharing the late tail may over-constrain the decoder's deepest layers and cost more quality than the extra bigram capacity buys back.
- The best late-layer sharing pattern is not known yet; alternating two shared groups across the last four layers is only the first plausible setting.
- The candidate keeps the repo's existing quantization/export path, so any remaining post-training quantization gap is still a limiting factor.

## Validation

Executed during candidate creation:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604031815_shared-tail-bigram/train_gpt.py
```

Outcome: **passed**.

Attempted CPU smoke test:

```bash
python - <<'PY'
import torch
PY
```

Outcome: **not feasible in this workflow environment** because the available Python runtime did not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`), so a real model-construction smoke test could not be executed before a GPU run.
