# Candidate: Relaxed Shared Tail MLP

## Hypothesis

The strongest non-TTT stack in this repository already gets most of its quality from the 11-layer / 3x-MLP architecture, but those deep MLPs are also some of the most byte-heavy tensors in the artifact. This candidate shares the **last four MLPs** through a single GPT-level MLP bank, then restores per-layer specialization with **rank-16 depth-specific low-rank adapters**. The saved parameter budget is partially reinvested in a slightly larger **BigramHash** table (`2048 -> 3072`).

The goal is to keep the winning late-stack structure intact (XSA, partial RoPE, LN scaling, value embeddings, EMA, and mixed int6 per-row export) while testing whether **partial parameter sharing** is a better use of the 16MB budget than four fully independent deep-tail MLPs.

## Why this is promising here

- The repository review showed that **MLP 3x expansion** was one of the biggest early gains and stayed in the best later stacks.
- The best non-TTT record, `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, already has the strongest architecture/export recipe that does **not** depend on eval-time TTT.
- A prior negative result on `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` found that **full layer recurrence** hurt badly under a fixed wall-clock budget. This candidate avoids that failure mode by sharing **only the deep MLP tail**, while keeping attention, skips, norms, and all earlier layers distinct.
- The top record on `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` reported a further gain from **BigramHash 2048 -> 3072**, so this candidate uses some of the recovered budget for that safer, already-observed direction.

There were **no prior `candidates/` directories** in the repository at the time this candidate was created.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - chosen as the base implementation because it is the strongest non-TTT record and already packages the repository's best late-stage training/export stack.
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
   - kept the partial-RoPE + LN-scale recipe that appears to be robust and parameter-free.
3. `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
   - confirmed that XSA on the deepest layers and EMA belong in the base stack.
4. `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
   - reinforced that wider MLPs plus n-gram features are worthwhile uses of parameter budget.
5. `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
   - important negative result: naive recurrence / repeated layers can lose more from reduced training efficiency than they gain from parameter reuse.

## External research that informed the choice

1. **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** ([arXiv:1909.11942](https://arxiv.org/abs/1909.11942))
   - motivates cross-layer parameter reduction as a way to improve parameter efficiency without giving up model depth.
2. **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** ([arXiv:2410.20672](https://arxiv.org/abs/2410.20672))
   - the closest match to this candidate: it argues that layer tying works better when small **layer-wise LoRA adapters** restore depth-specific flexibility.
3. **Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression** ([arXiv:2410.03765](https://arxiv.org/abs/2410.03765))
   - supports the narrower claim that **byte-heavy weights can be shared across layers** with limited quality loss under strong compression pressure.
4. **Coupled Query-Key Dynamics for Attention** ([arXiv:2604.01683](https://arxiv.org/abs/2604.01683))
   - reviewed but **not chosen** here because the paper reports gains on domain-coherent corpora but degradation on heterogeneous web text, which makes it a weaker fit for FineWeb.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. Added `SharedTailMLPBank`, a single shared MLP used by the last `SHARED_MLP_LAST_N` layers (default `4`).
2. Added per-layer low-rank adapters on both the MLP up-projection and down-projection paths (`SHARED_MLP_RANK=16` by default).
3. Removed the per-layer standalone MLP parameters from those deepest shared layers so the shared weights appear only once in the exported state dict.
4. Increased the default `BIGRAM_VOCAB_SIZE` from `2048` to `3072`.
5. Left the rest of the March 22 stack intact: XSA-on-tail, partial RoPE, LN scaling, value embeddings, smear gate, EMA, warmdown3500, and percentile-searched mixed int6 export.

## How to run

From the repository root:

```bash
cd candidates/202604051430_relaxed-shared-tail-mlp
SEED=1337 \
SHARED_MLP_LAST_N=4 \
SHARED_MLP_RANK=16 \
BIGRAM_VOCAB_SIZE=3072 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate keeps the base script's default assumptions about the FineWeb cache and tokenizer layout.

## Validation

Executed locally in this container:

```bash
python -m compileall candidates/202604051430_relaxed-shared-tail-mlp/train_gpt.py
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604051430_relaxed-shared-tail-mlp/train_gpt.py
git --no-pager diff --check
```

Outcomes:

- `compileall` on the candidate script: **passed**
- broader repository Python syntax check: **passed**
- diff whitespace check: **passed**
- minimal CPU-only smoke run: **not feasible in this container**, because `torch`, `sentencepiece`, and `flash_attn_interface` are not installed here

## Expected risks / tradeoffs

- Sharing the deep MLP tail may overconstrain late-layer specialization even with rank-16 adapters.
- The rank-16 adapters may be too small to fully recover four distinct 3x-MLP layers, or too large to be compute-neutral in practice.
- The larger BigramHash table may not repay the saved bytes as well as simply keeping more tensors high-precision.
- Real value depends on an actual 8xH100 training run; this repository environment only supports syntax-level validation for the candidate.
