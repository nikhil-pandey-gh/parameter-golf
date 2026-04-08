# LSQ-style Late Int6 QAT

## Hypothesis

The strongest next compression-aware candidate in this repo is to keep the proven 11-layer GPTQ-lite/EMA/XSA/partial-RoPE stack, but replace its brittle late-QAT path with a **compile-safe LSQ-style learned-scale int6 curriculum**. The goal is to make the model learn weights that survive the repo's real rowwise int6 export more gracefully, without adding major inference-time infrastructure.

## Why this is promising here

- The repo's biggest wins repeatedly came from **compression-aware changes** rather than from baseline-only hyperparameter tuning: fp16/tied-embed export, int6/int5 export, GPTQ-lite clip search, and QAT-funded width/depth all moved the frontier.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly documents that its late-QAT path was neutralized by `torch.compile` constant-folding, so there is still room for a **correct** late-stage quantization-aware implementation.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the right base because it already combines the strongest non-TTT stack: EMA, GPTQ-lite, XSA4, partial RoPE, LN scaling, VE128, BigramHash, and mixed int6 export.
- No `candidates/` directory existed when this workflow started, so this idea does not overlap an existing candidate branch in-repo.

## Prior repository influences

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Early QAT + int6 evidence:** `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`
- **Late-QAT correctness warning:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Top-stack context:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

## External research that informed this candidate

- **LSQ** — Esser et al., *Learned Step Size Quantization*, 2019: <https://arxiv.org/abs/1902.08153>
- **LSQ+** — Bhalgat et al., *LSQ+: Improving low-bit quantization through learnable offsets and better initialization*, 2020: <https://arxiv.org/abs/2004.09576>
- **BitNet b1.58** — Wang et al., 2024: <https://arxiv.org/abs/2402.17764>

The common theme is that very low-bit training improves when quantization parameters are part of training rather than treated as fixed post-hoc metadata. This candidate takes the smallest repo-compatible slice of that idea: **per-row learned scales for the existing int6-style weight path**, enabled only late in training.

## What changed vs. the chosen base

1. Added a per-row `lsq_scale` parameter to every `CastedLinear`.
2. Replaced the old boolean late-QAT switch with a **tensor-controlled `qat_strength` blend**, so the training graph no longer depends on a Python/class attribute that `torch.compile` can constant-fold away.
3. Swapped the fixed row-max fake-quant path for an **LSQ-style STE formulation**:
   - learned per-row scales,
   - STE rounding,
   - LSQ-style gradient scaling on the scales.
4. Turned late QAT into a **ramp**: when LR scale falls below `LATE_QAT_THRESHOLD`, `qat_strength` increases smoothly from 0 to 1 across warmdown instead of hard-switching.
5. Refreshed LSQ scales from the current weights at the moment late LSQ first activates.
6. Included auxiliary `lsq_scale` parameters in the scalar optimizer groups so the learned scales actually train.
7. Dropped `lsq_scale` tensors from the compressed export path (`final_model.int6.ptz`) while keeping the full-precision local checkpoint strict-loadable.

The export quantizer is intentionally unchanged: the model still exports with the same mixed int6 GPTQ-lite / int8 logic as the base run, while the training-only LSQ scales are filtered out of the compressed artifact.

## How to run

From the repository root:

```bash
cd candidates/202604082022_lsq-late-int6-qat
RUN_ID=lsq_late_int6_qat \
SEED=1337 \
LSQ_ENABLED=1 \
QAT_ENABLED=0 \
LATE_QAT_THRESHOLD=0.15 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This candidate keeps the strong defaults from the 2026-03-22 record stack, including:

- 11 layers, 512 model dim, 8 heads / 4 KV heads
- 2048 train/eval sequence length
- XSA on the last 4 layers
- partial RoPE (`ROPE_DIMS=16`)
- LN scaling, BigramHash, VE128, EMA, and GPTQ-lite export

Evaluation still prints the usual final metrics from the script:

- `final_int6_roundtrip_exact`
- `final_int6_sliding_window_exact`
- `final_int6_sliding_window_s64_exact`

## Validation

| Command | Outcome |
| --- | --- |
| `python -m compileall candidates/202604082022_lsq-late-int6-qat/train_gpt.py` | Passed |
| `python - <<'PY' ... exec_module(train_gpt.py) ... PY` | Not feasible in this runner: the required runtime deps (`numpy`, `torch`, `sentencepiece`) are not installed here, and the script also targets CUDA + direct FlashAttention 3 bindings (`flash_attn_interface`) rather than a CPU path. |

## Main risks and tradeoffs

- **Throughput risk:** even a late-stage LSQ path adds training-time work; if step time rises too much, the extra quant robustness may be offset by fewer optimization steps.
- **Artifact/compat risk:** the training-only LSQ state is now filtered out before export on purpose; if a future iteration starts using LSQ data at inference time, the export/load path would need to change again.
- **Optimization risk:** learned scales can become noisy late in training if the warmdown ramp is too aggressive.
- **Unverified interaction risk:** the idea composes cleanly with GPTQ-lite on paper, but the best threshold and ramp may differ from the base record's `0.15`.
