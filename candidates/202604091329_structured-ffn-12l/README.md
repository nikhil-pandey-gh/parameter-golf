# Structured FFN 12L

## Hypothesis

The current non-TTT record stack spends a large share of its artifact budget and training FLOPs inside the 3x feed-forward blocks. Replacing those dense FFN projections with low-rank factorizations should preserve most of the useful channel mixing while freeing enough compute and bytes to make a **12-layer** variant practical inside the same 10-minute budget.

## Why this is promising here

- The repo's strongest non-TTT runs all converged on **11 layers + 3x MLPs + aggressive quantization/export tuning**, which implies the FFN is now one of the main remaining cost centers.
- Prior records already showed that **more effective depth** and **larger MLP capacity** mattered a lot, while most recent iteration has focused on attention/XSA, EMA, quantization, and eval tricks rather than FFN structure.
- This candidate keeps the proven attention/export stack from the best non-TTT recipe and changes only the part that looks most overparameterized for the artifact budget.

## Prior repo evidence that informed this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`  
  Best non-TTT stack in this checkout: 11L, XSA, partial RoPE, LN scale, EMA, GPTQ-lite, VE, SmearGate, BigramHash.
- **FFN/MLP importance:** `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/` and `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`  
  Both show that 3x MLP width was a major contributor to the jump from the earlier baseline regime.
- **Depth still helps:** the repo trend from 9L to 11L improved materially once optimization/export were tuned.
- **Cheap token-pair priors still help:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` reports another gain from a larger bigram hash table, so this candidate nudges `BIGRAM_VOCAB_SIZE` upward while keeping the rest of the base stack stable.
- **Prior candidates:** no `candidates/` directory existed when this candidate was created.

## External research

- **Wei et al., "Structured Feedforward Layers for Efficient Parameterization of Large Language Models"** ([arXiv:2406.16450](https://arxiv.org/abs/2406.16450))  
  Shows that structured / low-rank FFNs can improve the parameter-FLOP tradeoff when training from scratch.
- **Wei et al., short version** ([arXiv:2407.09835](https://arxiv.org/abs/2407.09835))  
  Reports that low-rank FFNs can deliver meaningful FFN speedups while retaining good language-model scaling behavior.

## What changed vs the chosen base

Starting point: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate keeps the attention, optimizer, EMA, GPTQ-lite quantization, partial RoPE, LN scale, VE, XSA, SmearGate, and export logic from that base, then makes four targeted changes:

1. **Structured FFN:** each dense MLP matrix is factorized into two linear projections with `FFN_RANK=256`.
2. **One extra layer:** defaults move from **11 layers to 12 layers** to spend some of the saved FFN compute on additional depth.
3. **Slightly larger BigramHash prior:** `BIGRAM_VOCAB_SIZE` default moves from `2048` to `3072`.
4. **VE layer placement updated for 12L:** defaults move from `VE_LAYERS=9,10` to `VE_LAYERS=10,11`.

## How to run

From this candidate directory:

```bash
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate bakes in these defaults:

- `NUM_LAYERS=12`
- `MLP_MULT=3.0`
- `FFN_RANK=256`
- `BIGRAM_VOCAB_SIZE=3072`
- `VE_LAYERS=10,11`

The simplest follow-up sweep is:

```bash
for rank in 192 256 320; do
  FFN_RANK=$rank SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Validation

- `python -m compileall candidates/202604091329_structured-ffn-12l/train_gpt.py` - passed
- Minimal CPU smoke test was **not feasible in this runner** because the available Python environment does not currently provide `torch` or `sentencepiece`, so even an import-only execution path would fail before model startup.

## Main risks and tradeoffs

- If `FFN_RANK=256` is too aggressive, the FFN may become the bottleneck and erase the gain from adding the 12th layer.
- Low-rank FFNs can have worse training dynamics from scratch than dense FFNs; this is the main reason the candidate stays close to the best known optimizer/EMA stack.
- The extra layer could spend the saved compute in the wrong place; if training becomes slower than expected, the next sweep should try `FFN_RANK=192` or revert to 11 layers.
