# 202603301532_sandwich-mlp-share

## Hypothesis

The repo has already extracted most of the easy gains from better quantization, sliding-window evaluation, EMA/SWA, partial RoPE, and late-layer attention tweaks. What it has *not* really tested is **compute-neutral parameter sharing**: keeping the same 11 logical layers and same per-step FLOPs, while sharing only the heaviest MLP weights across mirrored encoder/decoder positions and keeping layer-specific norms, attention weights, skip paths, and learned residual scales unique.

My hypothesis is that this can help in two ways:

- it reduces export bytes without paying the throughput penalty that killed prior recurrence experiments,
- it adds a useful inductive bias instead of simply making the model smaller everywhere.

## Why this is promising for this repository

Two repository trends point in this direction:

- The strongest training-side stacks are already clustered around the same 11-layer family with partial RoPE, XSA, EMA, BigramHash, and aggressive mixed quantization.
- The negative results on "recurrence" were mostly about **extra compute in a fixed 10-minute budget**, not about sharing weights at fixed compute.

This candidate intentionally keeps the strong 11-layer recipe and only changes the parameter allocation inside the MLP path.

## Prior records that influenced this candidate

There were no prior `candidates/` directories in the repo when this was created, so the comparison set is the root baseline plus `records/`.

The main local influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - used as the base implementation because it is a strong pre-TTT 11-layer stack with EMA, partial RoPE, XSA, BigramHash, VE, GPTQ-lite-style clipping, and compact export.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirmed that partial RoPE + LN scale are genuine gains while the attempted late QAT path was compiled away.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the proven `LeakyReLU(0.5)^2` activation change and the observation that a larger `BIGRAM_VOCAB_SIZE` (3072 in the ablation table) is a low-cost lever.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - both reported that naive layer recurrence hurt under a strict wall-clock cap, which is why this candidate avoids extra recurrent passes.

## External research that informed it

The design is most directly motivated by these papers:

- **ALBERT** (`arXiv:1909.11942`): showed that cross-layer parameter sharing can preserve quality while reducing model size.
- **Subformer** (`arXiv:2101.00234`): specifically studied *generative* transformers and found that sandwich-style sharing can outperform naive cross-layer tying.
- **MobiLlama** (`arXiv:2402.16840`): used careful parameter sharing as part of a small-language-model design focused on reduced deployment cost.
- **On Expressive Power of Looped Transformers** (`arXiv:2410.01405`): argues that shared-loop architectures benefit from step-specific scaling, which fits this repo's habit of keeping small layer-specific control tensors even when large weights are shared.
- **Parameter Reduction Improves Vision Transformers** (`arXiv:2512.01059`): while in vision, it is a useful recent signal that **MLP sharing specifically** can improve stability at fixed compute.

## What changed versus the base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate makes three targeted changes:

1. **Mirrored MLP sharing**
   - The heavy MLP weights are moved out of each block into a shared bank.
   - For 11 layers, the default sharing map is:
     - `[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]`
   - That means the model still executes 11 logical layers, but only stores 6 unique MLP modules by default.
   - Attention weights remain untied.
   - Layer-specific control tensors (`attn_scale`, `mlp_scale`, `resid_mix`, skip weights, VE scales, etc.) remain unique.

2. **LeakyReLU(0.5)^2 activation**
   - Replaces the base `relu^2` MLP nonlinearity with the `LeakyReLU(0.5)^2` variant that helped the current best record.

3. **Slightly larger BigramHash default**
   - `BIGRAM_VOCAB_SIZE` now defaults to `3072` instead of `2048`.
   - This is a small, low-compute reinvestment of some saved byte budget rather than a broad architecture expansion.

Additional usability change:

- The script now resolves dataset and tokenizer defaults from the repository root so it can be launched **from inside this candidate directory** without manually rewriting paths.

## How to run

From the repository root:

```bash
cd candidates/202603301532_sandwich-mlp-share
RUN_ID=sandwich-mlp-share SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable sharing to compare against the same script without the new idea
SHARE_MLP=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Recover the old activation if you want to isolate sharing from LeakyReLU^2
MLP_NEGATIVE_SLOPE=0.0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected tradeoffs / risks

- Mirrored MLP sharing may be too restrictive if encoder and decoder halves truly need different feedforward transforms.
- The byte savings may not convert into better BPB if the current frontier is mostly compute-limited rather than artifact-limited.
- The candidate intentionally keeps the reinvestment modest; if this direction works, the next follow-up should probably test whether the freed bytes are best spent on a larger bigram table, larger VE, or a slightly wider untied lexical side path.
- This does **not** add legal TTT. It is meant as a training-side architectural probe on top of the strong pre-TTT family.

## Validation

I ran the following lightweight checks in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603301532_sandwich-mlp-share/train_gpt.py
```

Outcome:

- `train_gpt.py`, `train_gpt_mlx.py`, the `data/` helpers, and this candidate's `train_gpt.py` all compiled successfully.

I also probed the local Python environment:

```bash
python - <<'PY'
import importlib.util
for name in ('torch', 'flash_attn_interface', 'sentencepiece', 'zstandard'):
    spec = importlib.util.find_spec(name)
    print(f'{name}:', 'found' if spec else 'missing')
PY
```

Observed result:

- `torch`: missing
- `flash_attn_interface`: missing
- `sentencepiece`: missing
- `zstandard`: found

Because the container is missing the training/runtime dependencies and the script hard-requires CUDA + FlashAttention at runtime, I did **not** run a local CPU smoke test beyond syntax validation.
