# Shared-Core Reuse

## Hypothesis

The repo's quantization and eval stack is already heavily optimized, so the strongest underexplored lever is **depth-wise weight sharing**: reuse a small bank of transformer cores across a deeper effective stack, while keeping per-layer norms, residual scales, skip weights, `q_gain`, XSA flags, and value-embedding scales unique. This should add a useful recurrent inductive bias, regularize the tiny model, and preserve 16MB headroom by exporting the shared banks directly instead of materializing per-layer copies.

## Why this is promising here

- The current leaderboard is saturated with quantization, EMA/SWA, XSA, Partial RoPE, BigramHash, and TTT variants, but not cross-layer sharing.
- The best recent code already stores the large matrices in contiguous parameter banks, so sharing cores is a small refactor rather than a new infrastructure project.
- This challenge is bottlenecked by **artifact bytes per useful unit of depth**. Sharing the large Q/K/V/O and MLP matrices is a direct way to buy more effective depth per byte.

## Prior runs that informed this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` — strongest clean pre-TTT stack: EMA, GPTQ-lite, warmdown tuning, XSA, Partial RoPE, LN scale, VE, SmearGate, and BigramHash.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` — strongest overall stack and the best implementation substrate because it already banks the large weight matrices and uses LeakyReLU(0.5)^2.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` — evidence that Partial RoPE + LN scale are real wins.

## External research

- **ALBERT** (Lan et al., 2019): cross-layer parameter sharing reduces memory and improves scale efficiency. <https://arxiv.org/abs/1909.11942>
- **Universal Transformer** (Dehghani et al., 2018): recurrent depth reuse can add a helpful inductive bias while keeping the Transformer structure. <https://arxiv.org/abs/1807.03819>
- **Subformer** (Reid et al., 2021): sandwich-style sharing works better than naive layer tying in generative Transformers. <https://arxiv.org/abs/2101.00234>

## What changed versus the base implementation

Starting from the banked-matrix `2026-03-23` stack, this candidate:

1. Uses **12 effective layers** backed by **6 shared cores** (`NUM_LAYERS=12`, `NUM_SHARED_CORES=6`).
2. Applies **sandwich-style sharing** with mirrored outer layers and a repeated deepest core in the center when needed.
3. Keeps per-depth adaptation in the existing lightweight layer-specific parameters instead of introducing new large adapters.
4. **Preserves shared banks through quantization/export**, so the compressed artifact can benefit from the actual sharing structure.
5. Reinvests some saved capacity into a larger **BigramHash(3072)** side channel and enables value embeddings on the late 4 layers by default.
6. Resolves default dataset/tokenizer paths from the **repo root**, so the script can be run directly from inside this candidate directory.

## How to run

From this candidate directory:

```bash
cd candidates/202604061849_shared-core-reuse
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable sharing while keeping the rest of the candidate defaults.
NUM_SHARED_CORES=12 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Revert the lexical side-channel expansion.
BIGRAM_VOCAB_SIZE=2048 VE_LAYERS=9,10 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

- `python -m compileall train_gpt.py train_gpt_mlx.py candidates/202604061849_shared-core-reuse/train_gpt.py` — **passed**
- `python - <<'PY' ... import candidate ... PY` — **not feasible in this workflow env** because the repository's Python dependencies are not installed here (`ModuleNotFoundError: numpy`) before any candidate-specific code runs
- A minimal CPU-only start check was therefore **not feasible** in this environment; the inherited training path is also CUDA/FlashAttention-oriented once dependencies are present

## Main risks / tradeoffs

- Sharing the large banks may over-regularize the model and reduce genuinely useful layer specialization.
- 12 effective layers may lower the number of training steps reached in 600 seconds, so the `num_layers` / `num_shared_cores` ratio may need retuning.
- Direct bank quantization is a new export path in this candidate and could change compression behavior relative to the per-layer baseline.
- The interaction between shared cores and optional TTT/eval-time adaptation is untested.
