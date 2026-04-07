# Shared Early-MLP + Bigger Bigram

## Hypothesis

Selective **early-layer MLP sharing** can recover artifact budget without paying the throughput penalty that hurt prior layer-recurrence experiments here. If those saved bytes are reinvested into a slightly larger lexical path, the 11-layer XSA/EMA/GPTQ-lite stack may keep most of its modeling power while improving compression friendliness.

This candidate keeps the **same 11 logical layer applications** as the 2026-03-22 pre-TTT record. It does **not** add recurrent extra passes. Instead, it shares only the first two encoder-side MLP pairs (`0,1` and `2,3`), then spends part of the recovered budget on a larger `BigramHash` table and adopts the portable LeakyReLU(0.5)^2 activation from the current best record.

## Why it is promising here

- Repo evidence says **full extra recurrence is bad** under the 10-minute cap because it burns steps, but it does **not** rule out sharing within a fixed logical depth. The failed 1x5090 recurrence run added extra layer applications; this candidate does not.
- Repo evidence also says **bigger lexical shortcuts help**: BigramHash scaling repeatedly improved results, including 2048 -> 3072 in the current SOTA stack and 8192 -> 10240 in the 10-layer int5/int6 line.
- The top record showed **LeakyReLU(0.5)^2** is a cheap, portable MLP upgrade with a measurable gain and no SwiGLU-style throughput hit.
- External research supports the sharing direction:
  - **Intra-Layer Recurrence (ILR)** reports that recurrence is most effective when concentrated in **earlier layers**, which motivates sharing the early MLP path instead of the deepest blocks or the whole network. ([arXiv:2505.01855](https://arxiv.org/abs/2505.01855))
  - **ALBERT** showed that cross-layer parameter sharing can preserve quality while substantially reducing unique parameters. ([arXiv:1909.11942](https://arxiv.org/abs/1909.11942))

## Prior work that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Portable activation win:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Bigger bigram trend:** `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
- **Negative caution against extra recurrence / slow activations:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` and `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`

## What changed vs the chosen base

1. **Early MLP sharing:** new `SHARED_MLP_GROUPS` support, defaulting to `0,1;2,3` when `NUM_LAYERS >= 4`. Only the MLP modules are shared; attention, norms, residual controls, skips, and late layers stay unique. The int6 export path also dedupes those shared aliases before compression and restores them on load.
2. **Bigger lexical path:** `BIGRAM_VOCAB_SIZE` default increased from `2048` to `3072`.
3. **Activation update:** ReLU^2 -> **LeakyReLU(0.5)^2**.

Everything else stays on the cleaner 2026-03-22 training/export stack: 11 layers, XSA on the last 4 layers, partial RoPE, LN scaling, VE on layers 9-10, EMA, GPTQ-lite int6 export, and sliding-window eval.

## How to run

From this candidate directory:

```bash
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
BIGRAM_VOCAB_SIZE=4096 \
SHARED_MLP_GROUPS='0,1;2,3' \
VE_LAYERS=9,10 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script is self-contained and expects the same dataset/tokenizer layout as the main repo README.

## Main risks and tradeoffs

- **Shared MLP underfitting:** even partial sharing can remove too much layer-specific capacity.
- **Artifact savings may not translate to BPB:** zstd can exploit repeated tensors, but the lexical reinvestment may still be too small to repay the expressivity loss.
- **Bigram 3072 may not dominate 2048 on this exact stack:** prior gains were positive, but not on this exact configuration.
- **No local end-to-end CUDA validation here:** the real path still depends on CUDA + Hopper FlashAttention runtime.

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604071437_shared-mlp-bigram/train_gpt.py
```

Outcome: **passed**.

Attempted additional smoke validation:

```bash
python - <<'PY'
# import candidate module, stub flash_attn_interface, instantiate a tiny CPU model,
# and run a forward + quantize/dequantize roundtrip smoke test
PY
```

Outcome: **not feasible in this runner** because the local Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`). A real end-to-end run also needs the CUDA/FlashAttention stack that the challenge environment provides.

## Research references

- Anthony Nguyen et al., **Intra-Layer Recurrence** (Canadian AI 2025): [arXiv:2505.01855](https://arxiv.org/abs/2505.01855)
- Zhenzhong Lan et al., **ALBERT**: [arXiv:1909.11942](https://arxiv.org/abs/1909.11942)
