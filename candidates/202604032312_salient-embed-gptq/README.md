# Salient Embed GPTQ-lite

## Hypothesis

This candidate targets the bottleneck that showed up most consistently in the repository review: **post-training quantization damage, especially on the tied embedding/output matrix**. The hypothesis is that a small amount of saliency-aware mixed precision on `tok_emb.weight` can recover most of the benefit of full-fp16 embeddings without paying the full artifact-size cost.

Concretely, the script:

1. uses a GPTQ-lite style multi-percentile per-row clip search for all 2D float tensors,
2. computes rowwise reconstruction error for `tok_emb.weight`, and
3. keeps only the highest-error embedding rows in fp16 while quantizing the rest to int8.

## Why this is promising here

Repository evidence points in the same direction:

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` argues that the tied embedding is the most quantization-sensitive tensor because it serves as both input embedding and output head.
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` identifies quantization penalty as larger than many training-side gains.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` shows that GPTQ-lite clip search still buys measurable BPB gains even on a very strong stack.
- The review found **no existing `candidates/` directory**, so there was no prior candidate iteration to duplicate or avoid.

This candidate is deliberately orthogonal to the current winning architectural stack: if the export path helps, the same idea should transfer to stronger 10L/11L records as well.

## External research that informed it

- **AWQ** (Activation-aware Weight Quantization, arXiv:2306.00978) argues that a small subset of salient weights disproportionately determines quantization quality and that selectively protecting them is often sufficient.
- **GPTQ** (arXiv:2210.17323) shows that one-shot post-training quantization can stay accurate when the quantizer better fits weight geometry.
- **The Llama 3.1 quantization study** (arXiv:2411.02355) reports that well-tuned INT8 is often nearly lossless, which supports spending complexity on a better int8 path instead of introducing heavier training infrastructure.

The implementation here is intentionally simpler than AWQ or GPTQ proper: it uses weight-only rowwise reconstruction error, which fits this repository's single-file constraint and low-cost validation workflow.

## Base implementation and what changed

**Base implementation:** the repository root `train_gpt.py`.

### Changes versus the base

1. **Candidate-directory-safe paths.** Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root, so the script can be run directly from this candidate directory.
2. **GPTQ-lite row clip search for int8 export.** Matrix quantization now searches several clip percentiles per row instead of using a single fixed percentile.
3. **Salient embedding row rescue.** For `tok_emb.weight`, the quantizer keeps the worst reconstructed rows in fp16 (`TOKEN_EMBED_KEEP_ROWS`, default `64`) and stores only the remaining rows in sparse int8 per-row form.
4. **CPU smoke mode.** `SMOKE_TEST=1` runs a tiny synthetic forward/backward plus quantization roundtrip without CUDA or dataset shards.

## Records that most influenced this candidate

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600`
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3`

Those runs together suggest that compression quality, not just training loss, is often the binding constraint under this challenge's artifact limit.

## How to run

From this directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
TOKEN_EMBED_KEEP_ROWS=32 torchrun --standalone --nproc_per_node=8 train_gpt.py
TOKEN_EMBED_KEEP_ROWS=64 torchrun --standalone --nproc_per_node=8 train_gpt.py
TOKEN_EMBED_KEEP_ROWS=96 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

By default the script resolves:

- `DATA_PATH` to `../../data/datasets/fineweb10B_sp1024`
- `TOKENIZER_PATH` to `../../data/tokenizers/fineweb_1024_bpe.model`

relative to this candidate directory, so it can be launched without `cd`-ing back to the repo root.

## Validation

Commands run:

```bash
/tmp/gh-aw/agent/pg-venv/bin/python -m compileall candidates/202604032312_salient-embed-gptq/train_gpt.py
cd candidates/202604032312_salient-embed-gptq
SMOKE_TEST=1 /tmp/gh-aw/agent/pg-venv/bin/python train_gpt.py
```

Outcomes:

- `compileall`: passed
- CPU smoke: `smoke_test_ok loss:6.9566 roundtrip_loss:6.9574 token_embed_override_rows:64`

## Main risks and tradeoffs

- The row-selection heuristic is **weight-error based, not activation-aware**, so it may protect the wrong embedding rows compared with a true AWQ-style saliency signal.
- This is primarily an **export-path** candidate. If the training ceiling of the base model is already too low, quantization gains alone may not be enough.
- Larger `TOKEN_EMBED_KEEP_ROWS` values may improve roundtrip quality but can push the artifact back toward the 16 MB ceiling.
