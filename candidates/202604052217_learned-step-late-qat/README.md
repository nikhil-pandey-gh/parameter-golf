# 202604052217_learned-step-late-qat

## Hypothesis

The current strongest non-TTT stack in this repo already squeezes a lot out of architecture, EMA, and GPTQ-lite export, but its training-time quantization path is still weak: earlier late-QAT code could be constant-folded away under `torch.compile`, and the remaining export gap is still a live bottleneck. This candidate tests whether a **compile-safe, learned-step late int6 QAT endgame** can reduce that gap without changing the artifact format, while also carrying over the **LeakyReLU(0.5)^2** activation that helped the closely related 2026-03-23 record.

## Why this is promising here

1. The repo's winning trend is clearly **compression-aware training + stronger export**, not broad architectural churn.
2. The 2026-03-22 record is already the strongest clean training-only GPTQ-lite stack (`1.1233` mean sliding BPB), so it is the best base for a quantization-focused ablation.
3. The 2026-03-23 record showed **LeakyReLU(0.5)^2** is a low-complexity win on a very similar 11-layer stack.
4. Learned-step QAT is especially attractive for Parameter Golf because it improves the **same int6 artifact** at export time instead of spending bytes on new inference-time modules.

## Prior repo influence

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Activation carry-over:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Cautionary note:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` documents that an earlier late-QAT path was later found to have been dead-code-eliminated under `torch.compile`.
- **Related public candidate iterations:** open LSQ-flavored candidate PRs already existed in the public repo history (not in this checkout), especially **PR #611** (`202604030142_lsq-late-qat`) and **PR #785** (`202604051213_lsq-gptq-int6`).

The checked-out branch used for this run did not contain a local `candidates/` directory, but the public repo already had neighboring LSQ/QAT candidate iterations. This candidate is therefore **not** claiming LSQ-style late QAT as a brand-new direction; the twist is a cleaner, more isolated ablation of that family on the 2026-03-22 base.

## External research

- **EfficientQAT: Efficient Quantization-Aware Training for Large Language Models** (Chen et al., 2024/2025, <https://arxiv.org/abs/2407.11062>) motivates a late quantization-parameter phase, especially learning step sizes instead of only relying on post-training quantization.
- **LR-QAT -- a lightweight and memory-efficient QAT algorithm for LLMs** (Bondarenko et al., 2024, <https://arxiv.org/abs/2406.06385>) motivates lightweight QAT variants that preserve inference efficiency and adapt quantization parameters rather than rebuilding the whole training stack.
- **Learned Step Size Quantization** (Esser et al., 2019, <https://arxiv.org/abs/1902.08153>) is the direct inspiration for the per-row learned step sizes used here.

This candidate intentionally implements a **repo-fit simplification** of those ideas rather than full LR-QAT: it keeps the existing training/export pipeline and adds learned per-row step sizes only to the block attention/MLP matrices that are already exported to int6.

## What changed vs. the chosen base

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Compile-safe late QAT**
   - Replaces the old global `_qat_enabled` switch with per-module learned step sizes (`qat_log_scale`) and a tensor-valued blend factor (`qat_mix`), so the fake-quant path remains visible to `torch.compile`.
   - Applies fake quantization only to the block **attention + MLP** matrices that already export to int6.
   - Initializes late-QAT scales from the existing GPTQ-lite search, then ramps the fake-quant blend in as the LR warmdown scale drops below `LATE_QAT_THRESHOLD`.

2. **Learned-step export**
   - Export still produces the same mixed int6/int8 artifact layout.
   - When late QAT ran, the final int6 export reuses the learned per-row step sizes instead of rerunning percentile search for those matrices.

3. **Separate QAT optimizer group**
   - Learned step sizes train under their own AdamW group (`QAT_LR`, zero weight decay) instead of being mixed into the generic scalar controls.

4. **LeakyReLU(0.5)^2**
   - MLP activation changes from `relu(x)^2` to `leaky_relu(x, 0.5)^2`, following the gain reported by the 2026-03-23 record.

5. **Validation ergonomics**
   - Adds a FlashAttention-3 fallback to PyTorch SDPA when FlashAttention is unavailable.
   - Adds `SMOKE_TEST=1` synthetic mode so the candidate can exercise model creation, one train step, and quantized roundtrip without the FineWeb dataset.

6. **Explicit differences vs. prior LSQ-flavored candidates**
   - Keeps the base stack closer to the 2026-03-22 record instead of also changing BigramHash size or broader repo-root plumbing.
   - Makes `QAT_ENABLED=0` a true ablation that skips the late-QAT path entirely.
   - Excludes `qat_log_scale` from EMA so late-initialized quantization parameters are not averaged with pre-init zeros.
   - Exports the live learned scales after EMA is applied to the model weights, keeping the scale path aligned with the actual late-QAT endgame.

## How to run

From this directory:

```bash
SEED=1337 \
QAT_ENABLED=1 \
QAT_LR=0.01 \
LATE_QAT_THRESHOLD=0.18 \
MLP_LEAKY_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable learned-step late QAT, keep LeakyReLU^2
QAT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Keep QAT but shorten the blend-in window
LATE_QAT_THRESHOLD=0.15 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Synthetic smoke mode:

```bash
SMOKE_TEST=1 \
VOCAB_SIZE=128 NUM_LAYERS=2 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 \
MLP_MULT=2 TRAIN_SEQ_LEN=32 EVAL_SEQ_LEN=32 BIGRAM_VOCAB_SIZE=64 VE_ENABLED=0 \
python train_gpt.py
```

## Validation

| Command | Outcome |
| --- | --- |
| `python -m compileall candidates/202604052217_learned-step-late-qat/train_gpt.py` | Passed on this workflow runner. |
| `SMOKE_TEST=1 ... python candidates/202604052217_learned-step-late-qat/train_gpt.py` | Could not run on this workflow runner because the environment does not have `torch` installed. The candidate now includes a dataset-free synthetic smoke path for environments where PyTorch is available. |

## Expected risks / tradeoffs

- The learned-step path may reduce the quantization gap but still lose on total BPB if fake-quant overhead lowers step throughput too much.
- LeakyReLU(0.5)^2 was strong on the 2026-03-23 stack, but it has not yet been isolated on the exact 2026-03-22 GPTQ-lite base.
- Reusing learned scales at export improves train/export alignment, but GPTQ-lite percentile search may still beat them on some layers if the late-QAT window is too short.
- The candidate is designed to be minimal and inference-compatible, not a full LR-QAT reproduction; if it shows promise, the next step would be a more explicit quant-parameter-only endgame or small low-rank residual adapters.
