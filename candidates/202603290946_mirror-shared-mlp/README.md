# Candidate: Mirror-Shared MLP U-Net

## Hypothesis

The cleanest unexplored parameter-budget lever in this repository is **cross-layer MLP sharing that keeps compute fixed**.

Prior records show that the best 10-minute runs consistently benefit from larger MLP capacity, richer token-identity features such as `BigramHash` and `ValueEmbedding`, and stronger post-training compression. At the same time, a prior non-record exploration reported that naive **layer recurrence** hurt badly because it increased sequential compute and therefore cut the number of optimizer steps in a fixed wall-clock budget.

This candidate takes a different route:

- keep the **same 11 forward passes** as the strong 2026-03-22 stack,
- keep **attention, norms, residual scales, and skip structure layer-specific**,
- share only the **heavy MLP weights** across mirrored encoder/decoder layers in the existing U-Net layout,
- then spend some of the recovered artifact budget on slightly stronger cheap features (`BIGRAM_VOCAB_SIZE=3072`, `VE_DIM=192`) and preserve the savings at export time with alias-aware serialization.

The intended upside is better score-per-byte, not fewer FLOPs.

## Why this is promising for this repository

Repository evidence points to three stable trends:

1. **Artifact-aware design matters**. The strongest runs improved both model quality and how the final artifact is compressed.
2. **Bigger MLP-side capacity helps**. The leaderboard repeatedly rewards `MLP 3x`, better activations, and cheap token-identity side channels.
3. **Pure recurrence was tested in the wrong regime**. The negative result came from reusing layers while also paying extra sequential depth under the same wall-clock cap.

This candidate keeps the proven 11-layer schedule and instead uses parameter sharing to attack the **serialized byte budget**, which is the binding constraint for this challenge.

## Prior records that influenced this candidate

Primary local base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest "clean" base before TTT / parameter-banking complexity
  - preserves the repository's current best non-TTT recipe: 11 layers, XSA on deep layers, partial RoPE, EMA, GPTQ-lite clip search, late QAT, BigramHash, and ValueEmbedding

Additional influences:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - supplied the `LeakyReLU(0.5)^2` activation change, which was unusually strong for its implementation cost
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - important negative result: naive layer recurrence degraded performance because it reduced training steps in the fixed wall-clock regime
- earlier 10-minute records from 2026-03-19 through 2026-03-21
  - established the recurring value of MLP widening, mixed quantization, SmearGate, BigramHash, XSA, partial RoPE, and EMA/SWA-like averaging

## External research that informed it

This candidate is grounded in a mix of classic parameter-sharing ideas and newer small-model lessons:

- **ALBERT** (`arXiv:1909.11942`): strong evidence that cross-layer parameter sharing can reduce parameter count substantially without collapsing performance
- **Universal Transformer** (`arXiv:1807.03819`): iterative/refinement-style reuse suggests that repeated transformations can still be expressive when the surrounding state changes across depth
- **MiniCPM** (`arXiv:2404.06395`): emphasizes that small-model performance depends heavily on careful efficiency tradeoffs, not just raw scale
- **BitNet b1.58** (`arXiv:2402.17764`): reinforces that byte-efficient weights and quantization-aware modeling can change the optimal design space for compact LMs
- **SmolLM2** (`arXiv:2502.02737`): recent evidence that small models benefit from deliberate capacity budgeting and specialized design decisions rather than naive downsizing

These papers do **not** imply that full recurrence is automatically good here. Instead, they motivate a narrower claim: if parameters are scarce, **shared transformations plus carefully preserved per-layer context** can be a worthwhile direction.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Mirrored MLP sharing**
   - added `SHARE_MLP_MIRROR=1` by default
   - skip-paired encoder/decoder layers share MLP weights according to the existing U-Net pairing, while the final unpaired decoder block keeps its own MLP
   - attention stacks, norms, residual controls, and skip weights remain layer-specific

2. **Alias-aware export path**
   - detects duplicate tensors in the shared-MLP `state_dict`
   - stores each shared tensor only once in the quantized artifact
   - reconstructs alias keys on load so evaluation still uses a normal `state_dict`

3. **LeakyReLU(0.5)^2 MLP activation**
   - swapped the base `relu^2` MLP for the stronger cheap activation from the 2026-03-23 record

4. **Slight reinvestment of saved bytes**
   - `BIGRAM_VOCAB_SIZE`: `2048 -> 3072`
   - `VE_DIM`: `128 -> 192`

5. **Optional FlashAttention fallback**
   - if `flash_attn_interface` is unavailable, attention falls back to `torch.nn.functional.scaled_dot_product_attention`, allowing PyTorch to use its best available SDPA backend
   - training still expects the repository's usual CUDA environment

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603290946_mirror-shared-mlp
RUN_ID=mirror_shared_mlp_seed1337 \
SEED=1337 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable only the new sharing idea
SHARE_MLP_MIRROR=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Revert the new activation while keeping sharing
MLP_NEGATIVE_SLOPE=0.0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Spend fewer recovered bytes on side features
BIGRAM_VOCAB_SIZE=2048 VE_DIM=128 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- **Over-sharing risk**: mirrored MLPs may remove too much depth-specific capacity, especially if later layers need more specialized feed-forward transforms than earlier ones.
- **Optimization coupling**: shared MLP weights receive gradients from two logical depths, which may help regularize or may create destructive interference.
- **Export-only benefit risk**: if sharing mostly helps artifact size but not train-time quality, the larger side features may not fully use the recovered budget.
- **Quantization interaction**: alias-aware export is intentionally simple and should be verified on a real GPU run to confirm there are no quality regressions after round-trip load.

## Validation

Commands run in this environment:

```bash
python -m compileall train_gpt.py
python -m compileall ../../train_gpt.py ../../train_gpt_mlx.py ../../data train_gpt.py
python - <<'PY'
try:
    import torch
    print('torch:available')
except Exception as exc:
    print(f'torch:missing:{exc.__class__.__name__}:{exc}')
PY
```

Observed outcomes:

- `python -m compileall train_gpt.py` ✅ passed
- `python -m compileall ../../train_gpt.py ../../train_gpt_mlx.py ../../data train_gpt.py` ✅ passed
- `import torch` ❌ failed in this workflow environment with `ModuleNotFoundError: No module named 'torch'`

Because this environment does not have PyTorch installed, and the repository's trainer is CUDA-first (`train_gpt.py` raises `RuntimeError("CUDA is required")`), I could not run a real startup smoke test here without introducing new infrastructure. The intended next validation step is a minimal GPU launch in the repository's normal training environment.
