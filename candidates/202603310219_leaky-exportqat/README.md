# Candidate: LeakyReLU^2 + compile-safe export QAT

## Hypothesis

The strongest non-TTT stack in this repo is already the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` recipe, but two nearby gaps remain:

1. `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` reports a real gain from `LeakyReLU(0.5)^2`, even before its legal TTT tail.
2. `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly documents that its late-QAT flag was dead-code-eliminated under `torch.compile`.

This candidate combines those observations. It ports the proven `LeakyReLU(0.5)^2` activation into the best non-TTT 11-layer stack, then replaces the brittle late-QAT toggle with a compile-safe path that refreshes per-row int6 clip buffers and recompiles once when late QAT starts.

## Why this is promising for this repository

This repo's records consistently show that the last few points come from improving the **final quantized artifact**, not just the pre-quantized checkpoint. Sliding eval, mixed int6/int8 export, GPTQ-lite clip search, EMA, and warmdown tuning all point in that direction.

That makes a real export-aware late-QAT pass attractive here. The candidate keeps the strong 11-layer XSA/EMA/partial-RoPE/GPTQ-lite base, adds the later activation win, and tries to narrow the remaining train-to-export mismatch instead of introducing a broader architectural rewrite.

## Prior experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen base implementation.
  - Best non-TTT stack in this repo snapshot.
  - Established GPTQ-lite clip search + EMA + warmdown3500 as a strong export-oriented baseline.

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Contributed the `LeakyReLU(0.5)^2` MLP activation.
  - Its README reports a meaningful ablation win from the activation itself, independent of TTT.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Contributed the key failure mode to fix.
  - Its README notes that the late-QAT branch was constant-folded away under `torch.compile`.

- `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`
  - Reinforced that compression-aware training can matter materially in this challenge.

- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`
  - Reinforced that quantization quality is often a larger bottleneck than raw train loss.

## External research that informed it

- **LSQ** — Esser et al., *Learned Step Size Quantization* ([arXiv:1902.08153](https://arxiv.org/abs/1902.08153))
  - Motivates exposing low-bit quantization noise during training instead of only at export time.

- **AWQ** — Lin et al., *Activation-aware Weight Quantization* ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978))
  - Motivates protecting salient channels and treating clipping/scaling as part of the export problem, not just a generic post-processing detail.

- **Switch EMA** — Li et al. ([arXiv:2402.09240](https://arxiv.org/abs/2402.09240))
  - Reinforces the repo trend that weight averaging is a cheap, high-value component for the final exported checkpoint.

This candidate does **not** implement full LSQ or AWQ. Instead, it takes the minimal repo-compatible lesson from those papers: make low-bit exposure real during warmdown, and align the fake-quant scales with the same rowwise clip family used at export.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **LeakyReLU(0.5)^2 MLP**
   - Replaces `relu(x)^2` with `leaky_relu(x, 0.5)^2`.

2. **Compile-safe late QAT**
   - Keeps the existing late-QAT idea, but now:
     - refreshes per-row int6 clip buffers first,
     - flips the QAT flag,
     - recompiles the training graph once so the fake-quant branch is actually present in the compiled model.

3. **Export-aware clip refresh**
   - Each large `CastedLinear` owns a per-row clip buffer.
   - Late-QAT refresh computes the best row clip from the same percentile family used by the GPTQ-lite export search.
   - Those clips are periodically refreshed during warmdown.

4. **Shared int6 clip helper**
   - The final export quantizer now reuses the same row-clip search helper as the late-QAT refresh path, reducing train/export mismatch.

5. **Candidate-directory defaults**
   - Default `DATA_PATH` and `TOKENIZER_PATH` are resolved relative to the repository root, so the script can be launched from inside this candidate directory.

## What stays the same

This candidate intentionally keeps the rest of the 2026-03-22 stack unchanged:

- 11 layers, 512 hidden size, 8 heads / 4 KV heads
- 3x MLP width
- XSA on the last 4 layers
- partial RoPE (16/64) + layerwise LN scaling
- BigramHash + SmearGate
- shared value embeddings
- EMA export path
- GPTQ-lite int6 export
- warmdown3500 and the same optimizer split

## How to run

From this candidate directory:

```bash
cd candidates/202603310219_leaky-exportqat
RUN_ID=leaky_exportqat_seed1337 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If your dataset or tokenizer live elsewhere, override them explicitly:

```bash
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs for ablations:

- `LEAKY_RELU_SLOPE` (default `0.5`)
- `LATE_QAT_THRESHOLD` (default `0.15`)
- `QAT_REFRESH_EVERY` (default `200`)
- `QAT_CLIP_PERCENTILES` (default `0.9990,0.9995,0.9999,0.99999,1.0`)

## Expected risks / tradeoffs

- The one-time recompilation when late QAT starts adds overhead during warmdown.
- Periodic clip refresh performs extra quantile work, so it may trade a small amount of step throughput for a smaller export gap.
- `LeakyReLU(0.5)^2` was validated on the later parameter-banked stack; the exact gain may differ on this older non-TTT code path.
- This is still a conservative candidate: it does not add TTT, parameter banking, or a larger bigram table, so it is aimed at a cleaner non-TTT export win rather than the absolute highest-risk frontier idea.

## Validation

Lightweight validation run in this workspace:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603310219_leaky-exportqat/train_gpt.py
```

Outcome:

- `train_gpt.py` compiled successfully
- `train_gpt_mlx.py` compiled successfully
- `data/` helpers compiled successfully
- `candidates/202603310219_leaky-exportqat/train_gpt.py` compiled successfully

Runtime smoke-test status:

- A real local run was **not feasible** in this workspace because the cached FineWeb shards are absent, the SentencePiece model file is absent, and `flash_attn_interface` is not installed here.
- Concretely, local checks showed:
  - no `data/tokenizers/fineweb_1024_bpe.model`
  - zero `fineweb_train_*.bin` shards
  - zero `fineweb_val_*.bin` shards
  - `flash_attn_interface` unavailable

That means a meaningful CPU-only startup test would fail before reaching the candidate logic itself, so this candidate was validated with syntax-only checks in this environment.
