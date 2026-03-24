# Candidate: paired top-MLP sharing + BigramHash(8192)

## Hypothesis

The current best 11-layer stack likely has enough redundancy in its **last two decoder-side MLPs** that they can share one set of MLP weights without losing much quality, as long as each layer keeps its own norms, residual mix, layer scale, attention block, and skip path.

If that sharing works, the saved artifact budget can be reinvested into a **larger local lexical module** by increasing `BigramHash` from `2048` to `8192` buckets. The expected result is a better balance between:

- deep semantic refinement from the proven 11-layer/XSA/Partial-RoPE stack, and
- stronger short-range lexical modeling from a richer hashed bigram table.

## Why this is promising for this repository

Repository review points to two strong patterns:

1. The best current recipe is the 11-layer EMA + GPTQ-lite stack at `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which already combines the strongest known ingredients in this repo.
2. Earlier records repeatedly showed that **BigramHash helps**, and that **larger hash tables help more** when the artifact budget allows it. In particular, `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md` reports a gain from scaling BigramHash up to `10240` buckets.

This candidate tests whether a **small amount of architectural sharing** can fund a bigger bigram table without abandoning the current best architecture family.

## Prior records that influenced this candidate

### Chosen base

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest current result in the repo,
  - already includes 11L, XSA-on-last-4, Partial RoPE, LN scaling, EMA, GPTQ-lite export, SmearGate, and shared value embeddings.

### Specific prior evidence used here

- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - shows that increasing `BigramHash` capacity is worthwhile when bytes permit.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - establishes SmearGate + BigramHash as part of the winning compact-model family.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - already uses **shared value embeddings** on the top layers, which suggests the late stack is a sensible place to try slightly more sharing.

## External research that informed it

- **ALBERT** ([arXiv:1909.11942](https://arxiv.org/abs/1909.11942)) showed that **cross-layer parameter sharing** can reduce memory while preserving strong language-model quality.
- **Universal Transformer** ([arXiv:1807.03819](https://arxiv.org/abs/1807.03819)) argued that repeated/shared computation across depth can be effective when used carefully.
- **ShishuLM** ([arXiv:2510.13860](https://arxiv.org/abs/2510.13860)) specifically motivated trying **paired weight sharing** in small language models, reporting lower memory use and faster execution in SLM settings.
- **Parameter-Efficient Quality Estimation via Frozen Recursive Models** ([arXiv:2603.14593](https://arxiv.org/abs/2603.14593)) is a useful caution: broad recursive reuse can hurt, so this candidate keeps sharing **narrow and local** rather than making the whole stack recurrent.

## What changed versus the chosen base implementation

This candidate starts from the `2026-03-22` record and makes three deliberate changes:

1. **Paired top-MLP sharing**
   - Added `SHARED_MLP_LAYERS`, defaulting to `9,10`.
   - Layers 9 and 10 now reuse a single `shared_top_mlp` module.
   - Those layers still keep their own:
     - `mlp_norm`,
     - `mlp_scale`,
     - attention weights,
     - residual mix,
     - skip connections,
     - value-embedding scale application.

2. **Larger BigramHash by default**
   - Increased default `BIGRAM_VOCAB_SIZE` from `2048` to `8192`.
   - This is the main budget reallocation target for the saved top-layer MLP bytes.

3. **Portability fallback for attention**
   - If `flash_attn_interface` is available, the candidate keeps using FlashAttention 3.
   - Otherwise it falls back to PyTorch `scaled_dot_product_attention`, which makes the script more robust for local environments.

## Why this differs from the existing records and prior candidates

There were **no pre-existing `candidates/` directories** in this repository when this candidate was created.

Relative to the records, this candidate is distinct because it is the first one here to explicitly test **paired top-layer MLP sharing** on the current best 11-layer architecture family. The repo has already explored:

- deeper models,
- wider MLPs,
- XSA,
- Partial RoPE,
- EMA/SWA,
- mixed quantization,
- shared value embeddings,
- bigram features,
- sliding eval,
- TTT,

but not this specific **“share a tiny amount of late depth, spend the bytes on richer local lexical capacity”** tradeoff.

## Files added

- `candidates/202603242207_paired-mlp-bigram8192/train_gpt.py`
- `candidates/202603242207_paired-mlp-bigram8192/README.md`

## How to run or evaluate

From the repository root:

```bash
RUN_ID=paired_mlp_bigram8192 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
BIGRAM_VOCAB_SIZE=8192 \
SHARED_MLP_LAYERS=9,10 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
  candidates/202603242207_paired-mlp-bigram8192/train_gpt.py
```

From inside the candidate directory:

```bash
RUN_ID=paired_mlp_bigram8192 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
BIGRAM_VOCAB_SIZE=8192 \
SHARED_MLP_LAYERS=9,10 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation run for this candidate

I ran the lowest-cost checks available in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603242207_paired-mlp-bigram8192/train_gpt.py
```

Outcome:

- baseline repo syntax check: **passed**
- candidate syntax check: **passed**

A minimal CPU runtime smoke test was **not feasible** here because the environment does not have `torch`, `numpy`, or `sentencepiece` installed, and the task environment does not allow fetching them during this run. The candidate does, however, include a FlashAttention fallback so that missing `flash_attn_interface` alone no longer blocks import/runtime.

## Main expected risks or tradeoffs

- **Under-sharing risk:** sharing only layers 9/10 may save too few useful bytes to fully justify the larger bigram table.
- **Over-sharing risk:** the top two MLPs may still be specialized enough that tying them hurts more than the larger hash table helps.
- **Compression risk:** the 8192-bucket bigram embedding may compress worse than expected, reducing the byte advantage from sharing.
- **Interaction risk:** the current best stack already uses top-layer value embeddings; paired MLP sharing could either complement that or create redundant late-layer behavior.

## Suggested next experiments if this works only partially

1. Keep sharing but reduce the hash table to `6144` or `4096` if artifact margin is tighter than expected.
2. Move the sharing pair earlier (`8,9`) if the very top layer needs more specialization.
3. Extend the same idea to paired **attention output projections** instead of MLPs.
4. Combine this with a verified compile-safe late-QAT mechanism, since prior repo evidence suggests that space is still not fully explored.
