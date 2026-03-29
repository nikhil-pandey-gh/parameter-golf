# Embed-Residual GPTQ-lite

## Hypothesis

The strongest non-TTT record in this repo already uses an 11-layer GPTQ-lite + EMA stack, but it still pays a measurable post-quantization penalty and keeps the tied embedding fully quantized. The hypothesis here is that the tied embedding/output head is so sensitivity-dense that a **tiny fp16 low-rank residual** on top of the existing int8 GPTQ-lite base can recover a useful slice of the lost quality for only a small artifact cost.

## Why this is promising for this repository

Several records independently point to the same bottleneck:

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` shows the tied embedding is unusually sensitive to quantization.
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` frames quantization quality, not raw train loss, as the dominant post-training bottleneck.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` gets the best non-TTT static score by improving the quantizer itself, which suggests more export-time work is still a high-leverage surface.

This candidate spends a small number of extra bytes exactly where the repo evidence says they matter most: the tied embedding matrix that also serves as the output head.

## Prior records that informed this candidate

Primary base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Most relevant influences:

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`

There were no prior `candidates/` directories in the repository when this candidate was created.

## External research that informed it

This candidate is mainly inspired by the broader quantization literature's repeated message that a small number of sensitive directions dominate low-bit error, plus the low-rank literature's observation that useful corrections can often be represented compactly:

- **GPTQ** (`arXiv:2210.17323`) motivates smarter weight-only post-training quantization rather than treating all rows identically.
- **AWQ** (`arXiv:2306.00978`) argues that protecting a small set of salient directions can preserve much more accuracy than uniform treatment.
- **SmoothQuant** (`arXiv:2211.10438`) shows that moving or concentrating quantization difficulty offline can materially improve low-bit behavior.
- **QuaRot** (`arXiv:2404.00456`) reinforces the same idea from another angle: outlier structure, not just average error, is the core quantization problem.
- **LoRA** (`arXiv:2106.09685`) provides the low-rank prior: a small-rank correction can carry surprisingly large functional impact.

This implementation does **not** reproduce any one of those papers directly. Instead, it adapts their shared intuition to this repository's artifact-constrained export path: keep the existing GPTQ-lite int8 embedding base, then add a compact low-rank fp16 residual only for `tok_emb.weight`.

## What changed vs the chosen base implementation

Relative to `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. The candidate adds `EMBED_RESIDUAL_RANK` (default `32`).
2. During export, `tok_emb.weight` is still quantized with the existing GPTQ-lite int8 path.
3. After quantization, the script computes the embedding reconstruction error and stores a rank-`r` fp16 residual factorization (`resid_l`, `resid_r`).
4. During roundtrip load, the dequantized embedding is reconstructed as `int8_base + resid_l @ resid_r`.
5. The default data/tokenizer paths are resolved relative to the repository root so `train_gpt.py` can be run directly from this candidate directory.
6. Attention now falls back to PyTorch SDPA if FlashAttention is unavailable, which enables a local CPU-only smoke test of the model code without changing the intended GPU execution path.

## How to run / evaluate

From this candidate directory:

```bash
SEED=1337 \
EMBED_RESIDUAL_RANK=32 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Important notes:

- The script defaults `DATA_PATH` and `TOKENIZER_PATH` to the repository's shared `data/` directory, so it can be launched directly from `candidates/202603291915_embed-residual-gptq/`.
- On GPU with FlashAttention installed, the script keeps the same fast attention path as the base record.
- On machines without FlashAttention, the fallback is intended for smoke testing and functional validation, not for competitive throughput.

## Main risks and tradeoffs

- The extra residual might buy back too little BPB relative to its added bytes.
- A rank-32 correction is heuristic; the best rank may differ by hardware budget or final artifact margin.
- Correcting only `tok_emb.weight` may leave larger downstream quantization errors elsewhere untouched.
- The extra SVD at export time is small compared with training, but it is still added complexity versus the base record.

## Validation

- `python -m compileall /home/runner/work/parameter-golf/parameter-golf/candidates/202603291915_embed-residual-gptq/train_gpt.py`
  Passed in this workflow runner (`exit 0`).

- Planned CPU-only smoke test: import the module, instantiate a tiny GPT, run a forward pass through the SDPA fallback, then exercise `mixed_quantize_int6(..., embed_residual_rank=4)` and `dequantize_mixed_int6(...)`.
  **Not feasible in this runner** because both `python` and `python3` report `torch` as unavailable (`importlib.util.find_spec("torch") -> None`), so the repository's runtime dependency needed for any model import or forward check is missing.

Because of that environment limitation, local validation here is limited to syntax compilation plus static inspection of the candidate code paths.
