# Shared Early MLPs on the 11L GPTQ-lite Stack

## Hypothesis

The strongest non-TTT stack in this repo already spends most of its artifact budget on repeated 3x MLP weights. Sharing only the **early MLP sublayers** should preserve the proven 11-layer compute graph, late XSA/Partial-RoPE behavior, and per-layer residual controls while freeing bytes for slightly richer lexical/value side features.

The key twist is that this is **not** the same as the repo's earlier negative recurrence experiments. Those runs added extra looped depth and lost training steps under the fixed wallclock. This candidate keeps the same 11 logical layers and the same number of forward passes; it only reuses the early MLP weights.

## Why this is promising for this repo

Recent winning records have converged on a stable recipe:

- 11 layers + 3x MLP + int6 mixed export
- XSA on the deepest layers
- Partial RoPE + LN scaling
- EMA/GPTQ-lite style post-training smoothing and clip search
- cheap lexical inductive bias from BigramHash / SmearGate / VE

That makes **parameter reuse inside the already-good stack** one of the few remaining underexplored levers. The saved bytes are reinvested conservatively by increasing the default BigramHash table and moving VE to three late layers instead of two.

## Prior records and experiments that informed this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- **Partial RoPE + LN scale:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
- **XSA + EMA:** `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`
- **Lexical/value side features:** `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md`
- **Negative evidence against extra looped depth under fixed wallclock:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`

## External research that informed it

- **ALBERT** (Lan et al., 2019, https://arxiv.org/abs/1909.11942): parameter sharing can preserve performance while cutting model size.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (Nguyen & Lin, 2025, https://arxiv.org/abs/2505.01855): selective recurrence works better than blanket recurrence, with the best allocations skewed toward earlier layers.
- **Thinking Deeper, Not Longer** (Chen, 2026, https://arxiv.org/abs/2603.21676): shared-weight depth can work when stabilized carefully rather than applied indiscriminately.
- **Universal YOCO for Efficient Depth Scaling** (Sun et al., 2026, https://arxiv.org/abs/2604.01220): partial recursion restricted to shallow efficient layers is a better efficiency/capability trade than global looping.
- **Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence** (McLeish et al., 2025, https://arxiv.org/abs/2511.07384): retrofitting selective recurrence into an already-strong stack can preserve performance better than naive post-hoc looping.

The implementation here takes the most repo-compatible slice of those ideas: **share only shallow MLP weights, keep all attention blocks and the late stack unique, and avoid adding any extra recurrent steps.**

## What changed versus the chosen base

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. The first `SHARED_MLP_PREFIX_LAYERS=4` logical layers share MLP weights in pairs with `SHARED_MLP_GROUP_SIZE=2`, producing the mapping `[0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8]`.
2. Attention blocks, per-layer `mlp_scale`, residual mixing, skip weights, XSA, and late-layer structure stay unique.
3. The shared MLP pool is exported only once, and quantization classifies `shared_mlps.*` as MLP weights so they still use the int6 path.
4. Default `BIGRAM_VOCAB_SIZE` increases from `2048` to `3072`.
5. Default `VE_LAYERS` changes from `9,10` to `8,9,10`.
6. Default dataset/tokenizer paths resolve relative to the repository root, so the script can be run directly from this candidate directory without rewriting paths.

## How to run / evaluate

From the repository root:

```bash
cd candidates/202604022245_shared-early-mlps
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
SEED=1337 \
SHARED_MLP_PREFIX_LAYERS=4 \
SHARED_MLP_GROUP_SIZE=2 \
BIGRAM_VOCAB_SIZE=3072 \
VE_LAYERS=8,9,10 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still accepts the same `DATA_PATH`, `TOKENIZER_PATH`, `XSA_LAST_N`, `ROPE_DIMS`, `WARMDOWN_ITERS`, and quantization-related overrides as the base record.

## Main risks / tradeoffs

- Sharing shallow MLPs may create optimization interference between paired layers even though the attention paths remain unique.
- The saved bytes may not convert into better BPB without retuning learning rates, warmdown, or the Bigram/VE budget.
- Earlier VE plus a larger BigramHash table may improve lexical modeling, but could also shift the quantization/compression balance in ways that need real 8xH100 validation.
- If the best use of recurrence in this challenge truly requires extra test-time compute rather than artifact savings, this candidate may underperform despite being more parameter-efficient.

## Validation run in this workflow

| Command | Outcome |
| --- | --- |
| `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604022245_shared-early-mlps/train_gpt.py` | Succeeded. |
| Stubbed CPU smoke import/forward for `candidates/202604022245_shared-early-mlps/train_gpt.py` | Attempted, but not feasible in this container because `torch` is not installed (`ModuleNotFoundError: No module named 'torch'`). |
