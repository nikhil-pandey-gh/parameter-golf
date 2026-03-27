# Relaxed Recursive MLP Sharing

## Hypothesis

The strongest non-TTT stack in this repo already looks quantization-aware and attention-efficient, but it still spends a large fraction of its artifact budget on per-layer MLP weights.

This candidate tests whether a **relaxed recursive upper stack** can improve parameter efficiency without repeating the repo's earlier failed "naive recurrence" idea:

- keep **attention unique per layer**,
- share only the **upper-stack MLP cores**,
- add tiny **per-layer low-rank adapters on the reused layers** so repeated layers are not forced to be identical,
- spend the saved bytes on a **12th logical layer** and a larger **BigramHash(4096)** table,
- cherry-pick **LeakyReLU(0.5)^2** from the latest record as an orthogonal training-side gain.

## Why this looks promising here

Repo history suggests three important things:

1. The best clean training base is the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack, which already combines the strongest non-TTT ideas in one readable script.

2. Early "depth recurrence" attempts were negative in this repo, but they were effectively **full-block looping** under a tight 10-minute budget. This candidate revisits reuse in a much narrower form: **MLP sharing only**, with unique attention and per-layer adapters.

3. Recent records repeatedly show that extra capacity is good when it is bought cheaply: 11 layers, wider MLPs, bigger bigram tables, and better export all helped.

The new bet is that **shared MLP cores plus cheap adapters** can give some of the convergence/regularization benefits described in the recursive-transformer literature while staying much closer to the repo's proven architecture than a full recursive transformer.

## Prior records that shaped this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the main base implementation.

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the `LeakyReLU(0.5)^2` MLP activation, which appears to help even before TTT.

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - reinforced that embedding/output quality is unusually sensitive and that not every compactness trick is worth step-time cost.

- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - motivated keeping the bigram + SmearGate prior and spending savings on compact extra capacity.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - useful cautionary evidence: one late-QAT path was later found ineffective, so this candidate keeps the sharing hypothesis separate and disables late QAT by default.

## External research that informed it

- **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** (`arXiv:2410.20672`)
  - motivates layer tying with small depth-specific low-rank adapters instead of hard, fully identical recurrence.

- **Understanding Parameter Sharing in Transformers** (`arXiv:2306.09380`)
  - argues that parameter sharing can help not only by changing model complexity, but also by improving convergence.

- **FiPS: Learning Parameter Sharing with Tensor Decompositions and Sparsity** (`arXiv:2411.09816`)
  - specifically highlights MLP modules as strong targets for sharing/compression in transformers and LLMs.

I also considered Cross-Layer Attention / cross-layer KV-sharing (`arXiv:2405.12981`), but for this repo the **MLP weights dominate bytes more than KV projections do**, so shared MLPs looked like the higher-leverage first experiment.

## What changed versus the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

- increases the default model depth from **11 to 12 logical layers**,
- changes the default bigram table from **2048 to 4096** buckets,
- replaces ReLU^2 with **LeakyReLU(0.5)^2** in the main MLP,
- introduces a **shared MLP bank** controlled by `SHARED_MLP_PATTERN`,
- adds tiny **per-layer low-rank MLP adapters only on reused layers**, controlled by `MLP_ADAPTER_RANK`,
- defaults to an upper-stack sharing pattern:
  - `SHARED_MLP_PATTERN=0,1,2,3,4,5,6,7,4,5,6,7`
  - meaning the top 4 logical layers reuse the MLP cores from layers 4-7,
  - and only those reused top 4 logical layers receive the low-rank adapter path by default,
- keeps the rest of the strong base intact:
  - partial RoPE,
  - LN scale,
  - XSA on the deepest layers,
  - EMA + tight SWA,
  - GPTQ-lite int6/int8 export,
  - shared value embeddings,
  - sliding-window evaluation.

## How to run

From this candidate directory:

```bash
RUN_ID=relaxed_recursive_mlp \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults already encode the candidate recipe. If you change `NUM_LAYERS`, update `SHARED_MLP_PATTERN` and `VE_LAYERS` to stay consistent.

The script validates this explicitly:

- `SHARED_MLP_PATTERN` must have exactly `NUM_LAYERS` entries and use contiguous shared-core ids starting at `0`.
- `VE_LAYERS` entries must be unique and lie within `[0, NUM_LAYERS)`.

## How to evaluate

The script preserves the existing record-style flow:

- train with the wallclock cap,
- export mixed int6/int8 weights,
- reload the quantized roundtrip,
- report standard and sliding-window validation BPB.

Useful lightweight check:

```bash
python -m compileall train_gpt.py
```

## Validation run in this workflow

Successful:

```bash
python -m compileall candidates/202603271332_relaxed-recursive-mlp/train_gpt.py
```

Also re-ran the repo's lightweight syntax sweep style:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603271332_relaxed-recursive-mlp/train_gpt.py
```

Attempted but not feasible here:

- I tried to run a stronger CPU-only smoke test by injecting a temporary stub for `flash_attn_interface` and instantiating a tiny model.
- That failed immediately because the workflow Python environment does **not** have `torch` installed (`torch_available=False`), so a true import/forward smoke test was not possible in this container.

## Main risks / tradeoffs

- **Step-time risk:** a 12th logical layer may cost more training steps than the shared-MLP regularization repays.

- **Under-sharing vs over-sharing risk:** the chosen sharing pattern is intentionally conservative, but it may still remove too much top-layer feedforward diversity.

- **Quantization uncertainty:** late QAT is disabled by default here, so export quality depends on the existing GPTQ-lite path and smoother weights from EMA/SWA.

- **Pattern coupling:** this recipe is most coherent at the default depth. If you change layer count without updating the sharing map, the script will raise an error rather than guessing.
