# Basis-Shared Late MLP + Wider Token Features

## Hypothesis

The strongest non-TTT stack in this repository already looks quantization-limited rather than optimization-limited. My hypothesis is that the late decoder MLP output projections are redundant enough to **share one projection bank across the last four layers**, while each layer keeps a **small low-rank residual adapter** to recover layer-specific behavior. If that works, the model should become easier to compress under the 16MB cap while still preserving late-layer capacity.

I pair that with two cheap reinvestments that fit the repo's winning trends:

- **larger BigramHash** by default (`2048 -> 4096`) to reduce collisions,
- **broader shared value-embedding coverage** (`VE_LAYERS=7,8,9,10` instead of `9,10`),
- plus **LeakyReLU(0.5)^2** as the MLP nonlinearity, carried over from the latest high-performing TTT record because it was one of the cleanest non-infrastructure gains.

## Why this is promising for this repository

Repository review showed three persistent patterns:

- The challenge is usually **artifact/quantization constrained**, not purely training-step constrained.
- Small architectural priors like **BigramHash**, **shared value embeddings**, **Partial RoPE**, **LN scale**, and **LeakyReLU^2** stack well when they do not add much runtime complexity.
- **Naive layer recurrence** was a dead end locally, but that does **not** rule out softer cross-layer sharing. It suggests the repo wants **partial sharing with per-layer freedom**, not hard tying.

This candidate follows that evidence: it uses a compression-friendly form of cross-layer sharing only where redundancy is likely highest (late MLP projections), and it keeps per-layer adapters so it is not just another recurrence/tied-weights experiment.

## Prior records that influenced it

Primary base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Most important local influences:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - best clean no-TTT quantization/export stack in the repo snapshot
  - contributed the 11L / XSA4 / EMA / GPTQ-lite / warmdown3500 foundation
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - established Partial RoPE + LN scale as worthwhile zero/near-zero-cost additions
- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
  - showed EMA and late-layer XSA are part of the winning deep-stack recipe
- `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`
  - reinforced that BigramHash and token-local inductive bias help this repo
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - supplied the cleanest recent activation improvement: LeakyReLU(0.5)^2
- `2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`
  - useful negative evidence: hard recurrence / strong weight reuse can backfire under fixed wallclock

## External research that informed it

Primary source:

- **Wang et al., "Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression"** (`arXiv:2410.03765`)
  - argues that cross-layer sharing can outperform simpler tying/SVD baselines when layers reuse a shared basis but keep unique coefficients.

Additional research considered during selection:

- **Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs"** (`arXiv:2404.00456`)
- **Liu et al., "SpinQuant: LLM quantization with learned rotations"** (`arXiv:2405.16406`)

Those rotation-based PTQ papers reinforced the same high-level conclusion seen in the repo: **quantization outliers are the bottleneck**. I did not implement a QuaRot/SpinQuant-style residual-stream rotation here because it would require broader model surgery and loader/export changes than felt appropriate for a minimal candidate directory. Basis-sharing was the cleaner adaptation to this codebase.

## What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate adds:

1. **Late-layer basis sharing for MLP output projections**
   - new env knobs: `BASIS_SHARE_LAYERS` and `BASIS_SHARE_RANK`
   - default shared layers: `7,8,9,10`
   - implementation: one shared `shared_mlp_proj_weight` plus one low-rank residual adapter per shared layer

2. **LeakyReLU(0.5)^2 MLP activation by default**
   - new env knob: `ACTIVATION_NEGATIVE_SLOPE`
   - default: `0.5`

3. **Token-feature reinvestment**
   - `BIGRAM_VOCAB_SIZE` default raised from `2048` to `4096`
   - `VE_LAYERS` default expanded from `9,10` to `7,8,9,10`

4. **Practical runability improvements for candidate-only validation**
   - default `DATA_PATH` and `TOKENIZER_PATH` are resolved from repository root so the script works when launched from this candidate directory
   - FlashAttention import now has a fallback to PyTorch SDPA when FlashAttention is unavailable
   - `SMOKE_TEST=1` path was added for dependency-available forward-pass checks without dataset/GPU setup

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603260427_basis-share-late-mlp
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides for ablations:

```bash
# Turn basis sharing off
BASIS_SHARE_LAYERS= BASIS_SHARE_RANK=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Keep basis sharing but reduce residual freedom
BASIS_SHARE_RANK=16 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Revert to relu^2 while keeping the sharing idea
ACTIVATION_NEGATIVE_SLOPE=0.0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If dependencies are installed locally, the lightweight forward smoke test is:

```bash
cd candidates/202603260427_basis-share-late-mlp
SMOKE_TEST=1 python train_gpt.py
```

## Main expected risks and tradeoffs

- **Over-sharing risk**: if late MLP projections are less redundant than expected, the shared bank could hurt expressivity even with low-rank residuals.
- **Quantization-path mismatch risk**: the shared projection uses the same fake-int6 helper as `CastedLinear`, but it is still a custom path rather than the exact original linear module path.
- **Budget reinvestment risk**: the larger BigramHash may not pay for itself if the late-layer sharing does not sufficiently offset its artifact cost.
- **Ablation ambiguity**: this candidate intentionally combines one novel idea (basis-sharing) with one recent clean win (LeakyReLU^2) and small token-feature reinvestments, so the first follow-up experiment should isolate which part is carrying the gain.

## Validation run in this workflow

Executed in this environment:

```bash
python -m compileall candidates/202603260427_basis-share-late-mlp/train_gpt.py
```

Outcome:

- **Passed**.

Attempted dependency check for a CPU smoke run:

```bash
python - <<'PY'
mods = ['torch', 'sentencepiece']
for mod in mods:
    try:
        __import__(mod)
        print(f'{mod}:ok')
    except Exception as e:
        print(f'{mod}:missing:{e.__class__.__name__}:{e}')
PY
```

Outcome:

- `torch` missing in the current container, so the forward smoke test could not be executed here
- `sentencepiece` also missing in the current container, so the full training/eval path would still need dependencies installed

So a real `SMOKE_TEST=1 python train_gpt.py` forward pass was **not feasible in this workflow container** because `torch` is absent, and full training/eval would additionally require `sentencepiece`.
