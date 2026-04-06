# 2-head MTP on the 11L EMA + GPTQ-lite base

## Hypothesis

Turn on the already-plumbed multi-token-prediction (MTP) auxiliary heads in the strongest non-TTT stack so the model learns more per training token, while keeping the exported artifact unchanged by dropping the MTP heads before serialization.

## Why this looks promising here

- The best non-TTT record in this repo is `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, which already has a very strong compression/eval stack.
- Recent record code already carries MTP support, but the published logs and commands keep it off (`mtp_num_heads:0` / `MTP_NUM_HEADS=0`).
- The repository's biggest wins have come from better sample efficiency and better compression-aware training, not from adding broad new infrastructure.

## Prior repository work that influenced this candidate

- **Chosen base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Architecture stack carried forward:** `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
- **EMA + XSA trend:** `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
- **Repo-wide negative result that pushed me away from recurrence / block reuse instead:** `track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090` reports a bad wallclock tradeoff for layer recurrence.

## External research that informed the choice

- **Multi-Token Prediction** ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737)): auxiliary future-token heads improved sample efficiency on language modeling tasks with no inference-artifact cost when treated as training-only heads.
- **GPTQ** ([arXiv:2210.17323](https://arxiv.org/abs/2210.17323)) and **AWQ** ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)): they reinforce that the current repo is already exploiting the right quantization axis, so the next clean bet is improving training efficiency rather than replacing the export stack again.

## What changed vs. the chosen base

1. **MTP defaults are enabled**: `MTP_NUM_HEADS` now defaults to `2` and `MTP_LOSS_WEIGHT` to `0.15`.
2. **Run-from-candidate-dir defaults**: `DATA_PATH` and `TOKENIZER_PATH` now resolve from the repository root based on `__file__`, so the script can be launched from this candidate directory directly.
3. **FlashAttention fallback**: if `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA. This does not change the intended H100 path, but it makes import/forward smoke tests possible in less specialized environments that still have PyTorch installed.

Everything else intentionally stays aligned with the `2026-03-22` base: 11 layers, XSA on the last 4 layers, Partial RoPE (16/64), LN scaling, VE128 on layers 9-10, EMA, GPTQ-lite int6 export, and the existing warmdown/QAT schedule.

## How to run

From the repository root:

```bash
cd candidates/202604061756_mtp2-gptq-lite
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate defaults already enable the MTP auxiliary loss. To override them explicitly:

```bash
cd candidates/202604061756_mtp2-gptq-lite
SEED=1337 MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If your dataset/tokenizer live elsewhere, set `DATA_PATH` and `TOKENIZER_PATH` explicitly.

## Evaluation / expected behavior

- The script still exports without the MTP heads, so the saved artifact budget remains focused on the main model.
- The candidate should be judged exactly like the base stack: use the final `final_int6_sliding_window` / `final_int8_zlib_roundtrip_exact` lines.

## Main risks and tradeoffs

- The auxiliary heads may reduce training throughput enough to erase the sample-efficiency gain.
- Tiny models may prefer a smaller MTP weight or only one extra head; `0.15` / `2 heads` is a reasoned first bet, not a tuned optimum.
- Better training loss does not guarantee better post-quant BPB; warmdown length and late-QAT timing may need retuning once MTP is on.

## Validation run here

1. `python -m compileall candidates/202604061756_mtp2-gptq-lite/train_gpt.py` — **passed**
2. Tiny CPU import/forward smoke — **not feasible in this environment** because the local Python runtime does not have `torch` installed. The candidate now includes an SDPA fallback so the same smoke should work anywhere PyTorch is available, even without FlashAttention.
