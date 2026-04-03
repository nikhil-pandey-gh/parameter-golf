# Mirror-Shared MLP Bank

## Hypothesis

The repo's best training-only stacks already squeeze a lot out of evaluation, quantization, and small attention tweaks, but they still pay for a full unique 3x MLP in every transformer block. This candidate tests whether **mirroring the MLP bank across the 11-layer U-Net stack** can cut unique artifact bytes enough to improve compression efficiency without paying extra training compute, while a cheap **per-layer MLP input gain** preserves depth-specific behavior.

I also spend part of the recovered artifact budget on a **larger BigramHash table (3072 vs 2048)**, since the current top record reports a further gain from increasing bigram capacity and the compute cost is negligible compared with widening or deepening the transformer.

## Why this looks promising here

From the repo history:

- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` showed that **3x MLP** was one of the biggest architecture-side wins once int6 compression made it affordable.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`, `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`, and `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` established the strongest pre-TTT stack around **11 layers + MLP3x + XSA + Partial RoPE + LN scale + EMA/SWA + GPTQ-lite/int6 export**.
- The current #1 record, `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, reports another gain from pushing **BigramHash** upward, reinforcing that low-cost local-context capacity is still useful.
- The explore pass over `records/` found **no prior parameter-sharing / recurrent-depth candidate in this repo**, even though parameter reuse is explicitly in-scope for this challenge.

That makes cross-layer sharing one of the clearest untested spaces that still fits the repository's constraints.

## External research that informed this candidate

- **ALBERT** (Lan et al., arXiv:1909.11942) showed that **cross-layer parameter sharing** can dramatically shrink language-model parameter counts while preserving useful performance.
- **Universal Transformers** (Dehghani et al., arXiv:1807.03819) argued that reusing transformation blocks across depth introduces a useful recurrent inductive bias rather than acting only as a compression trick.
- **MatFormer** (Devvrit et al., arXiv:2310.07707) motivated focusing on the **FFN/MLP path** as a good place to build elastic capacity.
- **Basis Sharing** (Wang et al., arXiv:2410.03765) is the closest recent match to this repo's objective: it shows that **cross-layer shared bases** are effective specifically when compression ratios matter.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (Nguyen & Lin, arXiv:2505.01855) reinforced that **targeted reuse across depth** is still a live research direction for compact LMs.

This candidate is a deliberately small, code-local version of those ideas: no new training pipeline, just mirrored sharing on the largest repeated submodule in the strongest non-TTT record stack.

## Chosen base implementation

This starts from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

I chose that base because it is the strongest **pre-TTT** training stack already present in the repo, while still being simpler and easier to adapt than the March 23 record's extra evaluation and parameter-banking machinery.

## What changed versus the base

1. **Mirrored shared MLP bank**
   - Instead of 11 unique block-local MLP modules, the candidate builds **6 unique MLP banks** for the 11 logical layers.
   - Logical layers map to banks with:
     - `[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]`
   - This mirrors the repo's existing encoder/decoder-style skip layout and keeps the center layer unique.

2. **Per-layer MLP input gain**
   - Each block gets a learned vector `mlp_input_scale`.
   - The shared bank therefore sees depth-specific normalized inputs even when weights are reused.

3. **Larger BigramHash**
   - Default `BIGRAM_VOCAB_SIZE` increased from `2048` to `3072`.
   - The goal is to reallocate a little of the saved artifact budget into cheap local-context capacity instead of extra per-step compute.

Everything else is intentionally kept close to the March 22 stack:

- 11 layers, 512 dim, 8 heads / 4 KV heads
- 3x MLP width
- XSA on the last 4 layers
- Partial RoPE (`16/64`) + LN scale
- SmearGate + VE128 on layers `9,10`
- EMA + tight SWA
- GPTQ-lite style int6 export with zstd-22 fallback logic inherited from the base

## How to run

From this candidate directory:

```bash
RUN_ID=shared_mlp_bank \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable the mirrored shared bank and its per-layer input gain
SHARED_MLP=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Restore the March 22 bigram size
BIGRAM_VOCAB_SIZE=2048 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Commands run for this candidate in this workflow:

```bash
python -m compileall candidates/202604030949_shared-mlp-bank/train_gpt.py
```

Outcome:

- `compileall` succeeded.

CPU-only runtime smoke was **not** feasible in this environment because this script intentionally inherits the March 22 CUDA-only stack:

- it hard-requires `torch.cuda.is_available()`,
- it imports `flash_attn_interface`,
- and it expects the challenge dataset / tokenizer layout used by the repo's GPU path.

## Main risks / tradeoffs

1. **Too much sharing could undercut MLP diversity.** The mirrored bank halves unique MLP parameters aggressively, so this could regularize well or simply remove too much depth-specific capacity.
2. **The gain may come mostly from export size, not validation quality.** If sharing only shrinks bytes but weakens the fp16/bf16 model, the net BPB may not improve.
3. **The bigger bigram table may not offset lost unique MLP capacity.** It is cheap, but still only a small reallocation compared with what full MLP sharing removes.

## Suggested next experiments

If this idea is directionally good but too aggressive, the next things to try would be:

1. share only the **MLP up-projection (`fc`)** across mirrored layers while keeping per-layer down-projections unique,
2. share only the deepest half of the MLP stack instead of all mirrored pairs,
3. use the saved bytes for a slightly larger **VE** or a bigger **BigramHash** sweep (`3072` vs `4096`) rather than full bank sharing.
