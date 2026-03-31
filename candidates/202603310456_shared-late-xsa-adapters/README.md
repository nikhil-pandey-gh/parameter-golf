# Shared Late XSA Core + Adapters

## Hypothesis

The strongest non-TTT line in this repository already concentrates most of its modeling tricks in the deepest layers: XSA on the last 4 layers, shared value embeddings in late layers, partial RoPE, and EMA/GPTQ-lite export tuning. My hypothesis is that those deepest layers are redundant enough to **share one late transformer core across all 4 late steps**, while preserving layer-specific behavior with **tiny per-layer rank-8 adapters** and distinct normalization/control parameters.

If that works, the model should keep most of the effective depth of the 11-layer stack while paying for far fewer late-layer weights in the compressed artifact. The recovered bytes can then be partially reinvested in cheap lexical capacity, here via a larger `BigramHash` table and value embedding injection on all shared late steps.

## Why this is promising for this repository

The repo history points to a very specific winning family:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the strongest clean training/export stack in-tree without TTT.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` shows that zero-parameter late-layer refinements still matter.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` reports that increasing `BigramHash` size helped its stack, even though its main gains came from TTT and activation changes.

The common pattern is that the best stacks now look like: strong 11-layer core, aggressive late-layer tricks, careful export, and very little evidence that simply training longer or adding generic complexity is enough. That makes **late-layer sharing with a lightweight specialization path** a better fit than another broad architecture rewrite.

There were **no prior `candidates/` directories** in this repository when this candidate was created, so the historical review was based entirely on `records/`.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Key influences:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - best clean non-TTT stack in-tree
  - GPTQ-lite export, EMA, 11L XSA4, VE128, BigramHash, partial RoPE, LN scale
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - explicit warning that the late-QAT branch in that lineage was dead-code-eliminated by `torch.compile`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - evidence that larger `BigramHash` capacity can still help in the strong family

## External research that informed it

This candidate is mainly grounded in five papers:

- **ALBERT** — Lan et al., 2019, `arXiv:1909.11942`
  - established cross-layer parameter sharing as a practical way to reduce transformer parameter count.
- **Universal Transformer** — Dehghani et al., 2018, `arXiv:1807.03819`
  - motivates reusing the same transformation across depth rather than insisting on fully separate layers.
- **Head-wise Shareable Attention for Large Language Models** — Cao et al., 2024, `arXiv:2402.11819`
  - shows that fine-grained sharing can preserve quality better than coarse all-or-nothing layer sharing.
- **Cross-layer Attention Sharing for Pre-trained Large Language Models (LISA)** — Mu et al., 2024, `arXiv:2408.01890`
  - explicitly argues that late-layer attention patterns are highly redundant and that low-rank corrections are a good way to recover layer-specific differences.
- **LoRA** — Hu et al., 2021, `arXiv:2106.09685`
  - strong evidence that small low-rank specialization paths can recover a surprising amount of transformer capacity.

The combined takeaway is a good match for this challenge: **share the expensive part, keep cheap per-layer specialization, and spend recovered bytes where they buy the most BPB**.

## What changed versus the chosen base implementation

Base chosen: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate changes that base in five main ways:

1. **Shared late core**
   - Layers `7,8,9,10` are no longer four independent full blocks.
   - They now reuse one shared `late_shared_core` containing the expensive attention + MLP weights.

2. **Per-layer rank-8 adapters on shared late steps**
   - Each shared late step keeps its own RMSNorm/control tensors and gets a tiny bottleneck adapter (`down -> SiLU -> up`) after the shared block output.
   - This is the main mechanism for recovering step-specific behavior without reintroducing full per-layer weights.

3. **Layer-specific late control still preserved**
   - Every shared late step still has its own `attn_scale`, `mlp_scale`, `resid_mix`, LN scaling factor, and optional XSA flag.
   - This follows the repository trend that small late-layer control tensors are very cheap and often matter.

4. **More lexical-side capacity**
   - `BIGRAM_VOCAB_SIZE` default is raised from `2048` to `3072`.
   - `VE_LAYERS` now defaults to `7,8,9,10` instead of only `9,10`, aligning value-conditioning with the shared late stack.

5. **Safer defaults for this candidate directory**
   - The script now resolves dataset and tokenizer defaults from the repository root, so it can be launched directly from this candidate directory.
   - `LATE_QAT_THRESHOLD` defaults to `0.0` here, because the prior lineage documented a `torch.compile` dead-code issue for that path and this candidate is not intended to rely on it.

## How to run or evaluate it

From the repository root, after downloading the usual FineWeb shards and tokenizer:

```bash
cd candidates/202603310456_shared-late-xsa-adapters
RUN_ID=shared_late_xsa torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
SHARED_LATE_LAYERS=4 \
SHARED_LATE_ADAPTER_RANK=8 \
BIGRAM_VOCAB_SIZE=3072 \
VE_LAYERS=7,8,9,10 \
RUN_ID=shared_late_xsa \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script keeps the base stack's built-in export + evaluation flow:

- trains under the normal wallclock cap,
- exports `final_model.int6.ptz`,
- round-trips the compressed artifact,
- prints standard validation BPB and sliding-window BPB.

## Validation run for this candidate

I ran the following lightweight checks in this workflow:

```bash
python -m compileall candidates/202603310456_shared-late-xsa-adapters/train_gpt.py
```

Outcome:

- **Passed**.

I also attempted a minimal CPU smoke test by stubbing `flash_attn_interface` and importing the candidate module for a tiny forward pass, but that was **not feasible in this runner** because the environment does not have PyTorch installed:

```text
ModuleNotFoundError: No module named 'torch'
```

So, in this workflow, runtime validation could only go as far as syntax compilation.

## Main expected risks and tradeoffs

- **Over-sharing risk**: sharing all 4 deepest blocks may remove too much capacity if those layers are still doing materially different work in this repository's 11-layer regime.
- **Shared q/k/v bias risk**: this version shares the full late attention core, not just a subset of heads or projections.
- **Adapter rank may be wrong**: rank 8 is intentionally conservative; rank 4 may over-regularize, while rank 16 or 32 may spend too many bytes for too little gain.
- **Export gap may shift**: the model is smaller in its late stack, but the changed parameter distribution may interact differently with the existing GPTQ-lite/int6 export path.
- **Late QAT is intentionally disabled by default**: this avoids inheriting a known bug from the prior lineage, but it also means this candidate is not trying to shrink the train/export gap with fake quantization.

## Suggested next experiments if this looks promising

1. Sweep `SHARED_LATE_ADAPTER_RANK` across `4, 8, 16`.
2. Try `SHARED_LATE_LAYERS=3` as a less aggressive version.
3. If artifact bytes remain very comfortable, increase `VE_DIM` or `BIGRAM_VOCAB_SIZE` further.
4. If the shared-core idea looks good, try sharing only late attention while keeping late MLPs separate.
