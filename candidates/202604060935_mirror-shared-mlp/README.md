# Mirror-shared MLP + LeakyReLU² + Bigram3072

## Hypothesis

The strongest training-only stack in this repo already spends most of its bytes in 11 layers of attention + 3x MLP weights. This candidate tests whether **sharing only the expensive MLP weights across U-Net mirror pairs**, while keeping each layer's norms/residual scales/attention weights distinct, is a better parameter-budget trade than fully independent FFNs. The saved bytes are reinvested in a larger **BigramHash (2048 -> 3072)** and paired with **LeakyReLU(0.5)^2** plus a per-layer hidden scaling vector so the shared FFNs can still specialize.

## Why it is promising here

Repository evidence points in two directions:

1. The best pure training/export stack is the 11-layer EMA + GPTQ-lite + partial-RoPE/XSA family, especially `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`.
2. The repo has **not** really explored ALBERT/MobileLLM-style sharing at fixed logical depth. The closest local negative result was a 1x5090 **recurrence x2** experiment, which added extra layer passes and lost badly. This candidate deliberately keeps the **same number of forward passes** and only shares heavy FFN weights.

That makes mirror-shared FFNs a clean unexplored seam: lower artifact bytes without paying the recurrence throughput penalty that already failed elsewhere.

## Prior records and candidates that influenced it

- **`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`**: chosen base implementation and the strongest training-only local recipe.
- **`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`**: provided the LeakyReLU(0.5)^2 MLP swap and evidence that bigger BigramHash settings can help.
- **`2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`**: justified keeping partial RoPE + LN scaling intact.
- **`2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`** and related 3/20 runs: reinforced that SmearGate + BigramHash are still worth paying for.
- **`2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`**: useful negative result showing naive depth recurrence is not the right reuse strategy.

There were no prior `candidates/` directories in this checkout when this experiment was created.

## External research

- **ALBERT** (Lan et al., 2019): cross-layer parameter sharing can preserve quality while cutting parameter count. <https://arxiv.org/abs/1909.11942>
- **MobileLLM / MobileLLM-LS** (Liu et al., 2024): immediate block-wise weight sharing improved sub-billion models with only marginal latency overhead. <https://arxiv.org/abs/2402.14905>
- **(IA)³ / T-Few** (Liu et al., 2022): tiny learned activation scaling vectors are a strong low-parameter way to specialize shared weights. <https://arxiv.org/abs/2205.05638>
- **Universal Transformer** (Dehghani et al., 2018): recurrence/weight reuse can be a helpful inductive bias, but here it is adapted in the safer fixed-depth form. <https://arxiv.org/abs/1807.03819>

## What changed vs the chosen base

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Mirror-shared MLP bank**: 11 logical layers now map to 6 unique FFNs with share ids `[0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 5]`.
2. **Layer-local specialization preserved**: each layer keeps its own RMSNorms, residual mix, attention block, output scales, and a new `mlp_hidden_scale` vector for IA3-style FFN modulation.
3. **LeakyReLU(0.5)^2** replaces ReLU² inside the shared FFNs.
4. **BigramHash default increased to 3072 buckets**.
5. **Repo-root auto-discovery**: dataset/tokenizer defaults walk upward from the script until they find the repository root markers, so the file can be run from this candidate directory without rewriting the default paths.
6. **FlashAttention fallback**: if `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA so the module can still be imported or smoke-tested in more environments.

## How to run

From this candidate directory:

```bash
cd candidates/202604060935_mirror-shared-mlp
RUN_ID=mirror_shared_mlp \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

By default the script resolves:

- `DATA_PATH` to `../../data/datasets/fineweb10B_sp1024`
- `TOKENIZER_PATH` to `../../data/tokenizers/fineweb_1024_bpe.model`

Useful ablations:

```bash
MLP_SHARE_PATTERN=none BIGRAM_VOCAB_SIZE=2048 torchrun --standalone --nproc_per_node=8 train_gpt.py
MLP_SHARE_PATTERN=unet_mirror BIGRAM_VOCAB_SIZE=3072 LEAKY_RELU_SLOPE=0.5 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Commands run in this workflow:

1. `python -m compileall candidates/202604060935_mirror-shared-mlp/train_gpt.py`  
   **Outcome:** passed.
2. `python3 - <<'PY' ... importlib.util.find_spec('torch'/'numpy'/'sentencepiece') ... PY`  
   **Outcome:** this container does not have the repo runtime installed (`torch`, `numpy`, and `sentencepiece` all resolved to `None`), so a true CPU forward-pass smoke test was **not feasible** here without installing new runtime dependencies.

## Main risks / tradeoffs

- Sharing FFNs may remove too much layer-specific capacity even with `mlp_hidden_scale`.
- The larger bigram table may not repay the expressivity lost from FFN tying.
- This candidate intentionally leaves attention weights unshared; if the idea works, the next sweep should compare MLP-only sharing vs sharing the deepest attention blocks too.
- The novelty here is the FFN-sharing trade, not a rework of the inherited EMA/QAT/export path, so the biggest uncertainty is whether mirror sharing gives enough byte headroom to matter under the current quantization recipe.
