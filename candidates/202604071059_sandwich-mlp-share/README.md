# Sandwich-shared early MLP + LeakyReLU² + BigramHash3072

## Hypothesis

The strongest training-side stack in this repo already has most of the obvious wins: 11 layers, 3x MLPs, Partial RoPE, deep-only XSA, EMA, GPTQ-lite export, VE128, and BigramHash. The underexplored lever is **cross-layer parameter sharing**. My hypothesis is that the **earliest MLPs are more redundant than the later attention-heavy layers**, so pairwise-sharing the first four MLPs should:

1. act as a mild structural regularizer,
2. reduce the serialized artifact size,
3. let us spend the recovered bytes on a larger **BigramHash** table,
4. preserve most of the later-layer specialization that seems crucial in recent records.

I also fold in **LeakyReLU(0.5)^2**, since the current SOTA stack found that to be a cheap, reliable activation win.

## Why this is promising for this repository

- **Repo evidence says early easy wins are mostly exhausted.** Sliding eval, better warmdown, fp16-sensitive export handling, MLP3x, SmearGate/BigramHash, XSA, Partial RoPE, EMA, GPTQ-lite, and legal TTT are already well mined.
- **No reviewed record or prior candidate tried cross-layer sharing / recurrent-depth style reuse** in the training stack, even though the challenge description explicitly invites that class of ideas.
- **Recent records are right up against the 16 MB cap**, so a parameter-allocation trick that saves model bytes is unusually valuable here.
- **BigramHash has already proven useful in this repo**, so spending recovered bytes on more lexical buckets is a concrete way to convert sharing into capacity instead of just shrinking the checkpoint.

## Records and prior experiments that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen because it is the strongest clean training-side base in the repo review: 11L, Partial RoPE, LN scale, XSA-last-4, VE128, BigramHash, EMA, GPTQ-lite export.
- **Activation change borrowed from:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - That run reported a meaningful gain from **LeakyReLU(0.5)^2**.
- **Earlier architectural lineage:**  
  - `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`
  - `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/`
  - These established that MLP3x, BigramHash, SmearGate, and quantization-aware training/export were strong directions.

## External research that informed it

- **Subformer: Exploring Weight Sharing for Parameter Efficiency in Generative Transformers** ([arXiv:2101.00234](https://arxiv.org/abs/2101.00234))
  - Motivates **sandwich-style sharing** specifically for generative transformers and reports that it can outperform a standard Transformer with fewer parameters.
- **Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression** ([arXiv:2410.03765](https://arxiv.org/abs/2410.03765))
  - Reinforces the idea that **cross-layer sharing can be a compression-aware design tool**, not just a post-hoc model shrink step.
- **Intra-Layer Recurrence in Transformers for Language Modeling** ([arXiv:2505.01855](https://arxiv.org/abs/2505.01855))
  - Suggests that **earlier layers are a particularly promising place to reuse computation/parameters**.
- **ALBERT** ([arXiv:1909.11942](https://arxiv.org/abs/1909.11942))
  - Classic evidence that cross-layer sharing can improve parameter efficiency without simply collapsing model quality.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **LeakyReLU² MLP**
   - Changed the MLP activation from `relu(x)^2` to `leaky_relu(x, 0.5)^2`.
2. **Sandwich-style early MLP sharing**
   - The first four logical blocks share MLPs pairwise:
     - `blocks.0.mlp` == `blocks.1.mlp`
     - `blocks.2.mlp` == `blocks.3.mlp`
   - Per-layer norms, scales, residual mixes, and all attention weights remain distinct.
3. **Reinvested bytes into lexical capacity**
   - Increased the default `BIGRAM_VOCAB_SIZE` from **2048** to **3072**.
4. **Alias-aware quantized export**
   - Added alias tracking so shared MLP weights are quantized once and referenced by name aliases instead of being redundantly serialized.
5. **Run-from-candidate-directory defaults**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to this script, so running from this candidate folder works without extra path overrides.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202604071059_sandwich-mlp-share
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable the sharing idea
SHARED_MLP_LAYERS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Reduce the extra lexical spend
BIGRAM_VOCAB_SIZE=2048 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script inherits the base implementation's final int6+zstd roundtrip and sliding-window evaluation path.

## Validation

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604071059_sandwich-mlp-share/train_gpt.py`
  - **Passed**
- `python -m compileall candidates/202604071059_sandwich-mlp-share/train_gpt.py`
  - **Passed** after fixing the default data/tokenizer paths to resolve from the candidate directory
- `python - <<'PY' ... importlib.util.find_spec('torch') ...`
  - Reported **`torch_available=no`**

### CPU smoke test status

A minimal CPU-only smoke test was **not feasible in this workflow runner**:

- the runner Python environment does not currently have **PyTorch** installed,
- and the script's actual train/eval path expects **CUDA + FlashAttention**.

So this candidate is syntax-validated here, but still needs a real GPU smoke run to confirm startup and artifact behavior end-to-end.

## Main expected risks / tradeoffs

- **Over-sharing risk:** the first four MLPs may still need more layer-specific capacity than the sharing scheme allows.
- **Byte reallocation risk:** a larger BigramHash table only helps if the recovered artifact bytes translate into better lexical bias rather than worse compressibility.
- **Export-path novelty:** the alias-aware quantization path is new logic and should be checked with a real trained checkpoint to confirm the serialized artifact behaves exactly as intended.
