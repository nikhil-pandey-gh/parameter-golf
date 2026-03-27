# 202603270729 Mirror-Shared Banks

## Hypothesis

ALBERT-style parameter sharing is a good fit for this repository, but full recurrent depth reuse is not: prior repo evidence says looping layers burns too much 10-minute training budget, while external research suggests weight sharing can preserve quality when the compute graph stays fixed. This candidate therefore keeps the current 11-layer compute path intact and only **mirror-shares the large MLP banks** across symmetric layers, then spends some of the saved artifact budget on a **larger BigramHash table**.

## Why it is promising for this repository

The current best stack in `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is already near the 16MB cap and gets most of its gains from better eval, quantization, and small architectural tweaks. The repo also contains repeated evidence that full depth recurrence is too slow for this benchmark:

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` reports layer recurrence as the worst result in that sweep.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` explicitly calls depth recurrence promising but too step-hungry for 10 minutes.

This candidate tries the compute-neutral version of that idea: share only the dominant MLP bank weights, keep per-layer norms/scales/residual mixing/XSA placement untied, and keep attention weights layer-specific.

## Which records influenced it

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Supporting evidence:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` for the compression-aware late-stage stack that led into the current best model.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/` and the ablation table in `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`, which both point to bigger hashed bigram features being useful when the byte budget permits.
- The negative recurrence notes in the two READMEs above, which are why this candidate avoids adding any extra layer applications.

There were no prior experiments under `candidates/` when this candidate was created.

## External research that informed it

- **ALBERT** (Lan et al., 2019, arXiv:1909.11942): cross-layer parameter sharing can cut parameters substantially while preserving model quality.
- **Universal Transformer** (Dehghani et al., 2018, arXiv:1807.03819): depth reuse is powerful, but in this repo the wall-clock budget makes extra recurrent applications unattractive.
- **Multi-Token Prediction** (Gloeckle et al., 2024, arXiv:2404.19737): considered as an auxiliary-training add-on, but not enabled here because this implementation would likely add meaningful per-step compute.
- **Multi-Query Attention** (Shazeer, 2019, arXiv:1911.02150): also considered as a byte-saving mechanism, but this candidate stays closer to the current winning stack and changes only the MLP side.

## What changed versus the chosen base implementation

Relative to `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`:

1. Added `SHARED_MIRROR_MLP` (default `1`).
2. Replaced the per-layer MLP banks with a **mirror-sharing map**. For 11 layers, the shared-bank pattern is:

   - `[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]`

   This reduces unique MLP bank slices from 11 to 6 while keeping the 11-block forward graph unchanged.
3. Kept attention banks fully untied to avoid entangling XSA / non-XSA attention behavior across layers.
4. Changed the candidate default `BIGRAM_VOCAB_SIZE` from `2048` to `3072`, using some of the saved artifact budget on a previously helpful feature.
5. Changed the export path so quantization operates on the **shared-bank state dict directly**, which preserves the candidate's byte savings instead of expanding shared weights back to per-layer tensors before compression.
6. Generalized the int6 per-row quantizer/dequantizer to handle the bank tensors directly.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603270729_mirror-shared-banks
TTT_ENABLED=1 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Important notes:

- `SHARED_MIRROR_MLP=1` and `BIGRAM_VOCAB_SIZE=3072` are already the candidate defaults.
- `TTT_ENABLED=1` is still recommended explicitly, following the current best record's evaluation recipe.
- To ablate the new idea while keeping the same script, set `SHARED_MIRROR_MLP=0`.

## Main expected risks or tradeoffs

- The saved bytes may not fully compensate for the reduced MLP diversity; if the current leaderboard is capacity-limited rather than byte-limited, this could regress BPB.
- Mirror-sharing is aligned with the encoder/decoder-like skip structure, but it is still a strong inductive bias and may underfit middle/deep layers.
- The larger bigram table is intentionally low-compute, but it may not be enough to pay back the lost layer-specific MLP capacity.
- This candidate preserves the existing TTT path rather than changing eval protocol, so most upside must come from better compressed training weights.

## Validation

Ran:

```bash
python -m compileall train_gpt.py ../../train_gpt.py ../../train_gpt_mlx.py ../../data
```

Outcome:

- Success. All listed files compiled to bytecode without syntax errors.

Attempted runtime smoke check:

- Wrote a temporary `flash_attn_interface` stub under `/tmp/gh-aw/agent/` and tried to import this candidate plus instantiate a tiny CPU model.
- This environment does not have `torch` installed, so the smoke import failed immediately with `ModuleNotFoundError: No module named 'torch'`.
- Because of that missing dependency, no CPU runtime smoke test was feasible here without introducing new infrastructure.
