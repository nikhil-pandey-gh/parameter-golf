# Salient-Row Rescue

## Hypothesis

The current 11-layer EMA + GPTQ-lite stack is already strong on training-side quality, but its export path still spends most of the artifact budget on quantized weights. Prior records show that the tied embedding/head and late attention keys are unusually quantization-sensitive. This candidate keeps the base training recipe unchanged and instead rescues a tiny set of the highest-error rows from those tensors in fp16, so the model pays a small byte premium only where quantization hurts most.

## Why this is promising here

- `2026-03-18_FP16Embed_WD3600` showed that fully preserving `tok_emb.weight` in fp16 nearly eliminated the quantization gap, but it needed a narrower MLP to stay under 16MB.
- `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` kept the last-layer key projection in fp16 because late attention keys were especially fragile.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` already pushed post-training quantization with per-row clip search, yet still left room in the artifact budget (~15.55MB total) for a more selective mixed-precision export.

Those three runs suggest the next cheap win is not a broader architecture rewrite, but a more surgical export format: keep almost everything quantized, and only pay fp16 bytes on the worst rows.

## Prior experiments that influenced this candidate

| Prior run | Influence |
|---|---|
| `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` | Chosen base stack: 11L, EMA, GPTQ-lite clip search, partial RoPE, XSA, VE, warmdown=3500 |
| `2026-03-18_FP16Embed_WD3600` | Evidence that the tied embedding/output head is the single most quantization-sensitive tensor |
| `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` | Evidence that selective fp16 on late key projections can be worth paying for |
| `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` | Current SOTA, but too close to the artifact cap to add much extra fp16 payload safely |
| `2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090` | Negative result on naive layer recurrence/depth reuse, which made architecture-sharing ideas less attractive for a first pass |

## External research

- **LLM.int8()**: Dettmers et al. show that a small set of emergent outlier features dominate transformer behavior and motivate mixed-precision handling for those outliers instead of uniform low-bit quantization. <https://arxiv.org/abs/2208.07339>
- **AWQ**: Lin et al. show that protecting only about 1% of salient weights can sharply reduce weight-only quantization error. <https://arxiv.org/abs/2306.00978>
- **SpQR**: Egiazarian et al. show that explicitly isolating outlier weights in higher precision can make 3-4 bit compression near-lossless, especially where smaller models are more accuracy-sensitive. <https://arxiv.org/abs/2306.03078>

These papers all point in the same direction: sparse high-precision exceptions can beat uniform quantization when bytes are tight.

## What changed vs. the base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes:

1. Added export knobs for sparse fp16 rescue:
   - `OUTLIER_FP16_PAYLOAD_BUDGET_BYTES` (default `196608`, with `OUTLIER_FP16_BUDGET_BYTES` accepted as a legacy alias)
   - `OUTLIER_FP16_MAX_ROWS_PER_TENSOR` (default `64`)
   - `OUTLIER_FP16_EMBED_ENABLED` (default `1`)
   - `OUTLIER_FP16_LATE_K_ENABLED` (default `1`)
   - `OUTLIER_FP16_LATE_K_LAYERS` (default `2`)
2. After GPTQ-lite/int8 quantization is chosen for a tensor, the script computes row-wise reconstruction error.
3. It gathers candidates from:
   - `tok_emb.weight` / `lm_head.weight`
   - the last `OUTLIER_FP16_LATE_K_LAYERS` attention key matrices
4. It keeps only the highest-error rows that fit under a raw fp16 override payload budget, stores those rows in fp16, and overwrites the dequantized rows with the preserved fp16 versions at eval time.
5. Training, EMA, sliding-window eval, GPTQ-lite clip search, and the 11-layer architecture are otherwise unchanged.

## Why this candidate, not grouped weight sharing

ALBERT-style sharing is a reasonable research direction, but the repo evidence is mixed for depth reuse under the strict wall-clock cap. The 1x5090 sweep already found naive recurrence to be net negative, while most successful repository gains came from quantization/export improvements. Sparse fp16 rescue is the lower-risk idea that best matches both the literature above and what has actually worked in this codebase.

## How to run

From this directory:

```bash
RUN_ID=salient_row_rescue \
OUTLIER_FP16_PAYLOAD_BUDGET_BYTES=196608 \
OUTLIER_FP16_MAX_ROWS_PER_TENSOR=64 \
OUTLIER_FP16_LATE_K_LAYERS=2 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The remaining defaults intentionally match the 2026-03-22 base stack.

## How to evaluate

The script preserves the base flow:

1. train under the 600s wall-clock cap,
2. apply EMA weights,
3. export the sparse-fp16 mixed-precision artifact,
4. dequantize it back,
5. run standard eval and sliding-window eval.

Watch for the new export log line:

```text
outlier_fp16 rows:<N> payload_bytes:<B> tensors:<name:count,...>
```

That line reports how much fp16 override payload the run actually used.
Use the existing `Serialized model int6+...` / `Total submission size ...` lines to confirm the real compressed artifact size; the payload budget is only a pre-serialization selector for the rescued rows.

## Risks and tradeoffs

- The row ranking is based on weight-domain reconstruction error, not activation-calibrated saliency, so it is a lighter-weight proxy for AWQ rather than a full activation-aware method.
- The payload budget is only a proxy for the final artifact size: too little fp16 rescue may leave quantization error on the table, while too much rescue can still erase the byte savings once `torch.save` metadata and compression are included.
- The best row budget may depend on whether the compressor is `zstd` or `zlib` in the runtime environment.
- This does not change training dynamics at all, so the upside is bounded by how much of the remaining gap is truly export-side.

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604061727_salient-row-rescue/train_gpt.py` | Passed |
| `python -m py_compile candidates/202604061727_salient-row-rescue/train_gpt.py` | Passed |
| Minimal CPU smoke test | Not feasible in this environment without changing the candidate further: this script inherits the CUDA + FlashAttention 3 runtime path from the 2026-03-22 base and requires the challenge dataset/tokenizer layout to enter `main()` |
