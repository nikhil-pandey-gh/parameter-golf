# Candidate: Signed-Hadamard GPTQ-lite on the LeakyReLU2 + Legal TTT stack

## Hypothesis

The current best line in this repository already does most of the obvious training-side things well: 11 layers, MLP3x, Partial RoPE, XSA, EMA/SWA-style averaging, parameter banking, and legal score-first TTT. The remaining gap looks increasingly compression-bound.

This candidate tests whether a `SpinQuant`/`QuaRot`-style **rotation-aware weight-only export** can improve the final int6 artifact without disturbing the training stack. The specific hypothesis is that applying a deterministic signed Hadamard rotation to large 2D attention/MLP weights **before** row-wise int6 quantization, then inverting that rotation **after** dequantization, will reduce outlier-driven quantization error enough to improve roundtrip and sliding-window `val_bpb`.

## Why this is promising for this repository

Recent repository history says the biggest durable gains came from compression-aware changes rather than broad rewrites:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` showed that a pure export-side GPTQ-lite clip search still bought another measurable gain on top of the strong 11-layer stack.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` pushed the frontier further with LeakyReLU2, parameter banking, and legal TTT, but still kept the same artifact-sensitive int6 export path.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` is a strong reminder that extra training alone does not eliminate the post-quantization gap.

That makes export quality one of the highest-leverage remaining knobs.

## Prior repository work that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - chosen base implementation because it is the strongest current line and already contains the latest repo-proven architecture/training choices.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - important precedent that small, careful export improvements can still move the metric.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - reinforces the repo trend toward small, composable late-stage improvements rather than wholesale rewrites.
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`
  - early clear statement that this challenge is often won by training/export choices that make weights easier to compress.

There were no prior `candidates/` directories in the repo at the time this candidate was created.

## External research that informed it

- **QuaRot** — Croci et al., arXiv:2404.00456
  Shows that orthogonal rotations can remove outliers and make even aggressive low-bit quantization much easier; notably, the paper reports essentially lossless 6- and 8-bit behavior for LLaMA2 with simple round-to-nearest quantization.
- **SpinQuant** — Liu et al., arXiv:2405.16406
  Shows that some rotations are materially better than others, and that choosing better rotations can noticeably shrink the quantization gap.

This candidate adapts those ideas in the lightest-weight way that fits this repo:

- no calibration dataset,
- no learned rotation parameters,
- no runtime graph surgery,
- no changes to the training path,
- only a deterministic export/dequantization change.

## What changed versus the chosen base implementation

Base file:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

New file:

- `candidates/202603300805_signed-hadamard-gptq/train_gpt.py`

Changes:

1. Added a **SpinQuant-lite** int6 path for large 2D attention/MLP weights after the state dict is unbanked for export.
2. For each eligible tensor, the exporter now compares:
   - the existing identity-space GPTQ-lite row-wise int6 quantizer, and
   - a **deterministic signed Hadamard rotated** version.
3. The rotated path is only kept if it produces lower reconstruction MSE than the baseline quantizer for that tensor.
4. The chosen rotation metadata is stored in the quantization metadata and inverted during dequantization before loading the evaluation model.
5. Added two lightweight knobs:
   - `QUANT_ROTATE_ENABLED=1` to enable/disable the new path
   - `QUANT_ROTATE_MIN_DIM=128` to skip tiny matrices
6. Added a small export log line showing how many int6 tensors selected the rotated path.

Everything else stays aligned with the base record: LeakyReLU2, parameter banking, legal score-first TTT, Partial RoPE, XSA, VE, and the existing lzma-compressed int6 artifact.

## How to run or evaluate

For a fair comparison against the March 23 base record, start from its launch recipe and add the rotation toggle. The most important override relative to this copied script's defaults is `BIGRAM_VOCAB_SIZE=1536` (the base record used 1536, while this script inherits a default of 2048).

Base-aligned candidate command:

```bash
BIGRAM_VOCAB_SIZE=1536 \
QUANT_ROTATE_ENABLED=1 \
TTT_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Minimal ablation against the identity-space exporter:

```bash
BIGRAM_VOCAB_SIZE=1536 \
QUANT_ROTATE_ENABLED=0 \
TTT_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

These snippets only show the candidate-specific toggle plus the base-record BigramHash override. For closer parity, carry over the rest of the `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` launch settings as well.

## Main expected risks and tradeoffs

- Lower per-tensor weight MSE may not translate into lower final `val_bpb`; the safeguard is local, not metric-aware.
- Rotating weights can improve quantization error while making the compressed byte stream slightly less lzma-friendly.
- Export time increases because eligible int6 tensors now try two quantizers instead of one.
- This is only a weight-space adaptation of rotation-based PTQ; it does not capture the fuller activation/KV-cache treatment from QuaRot/SpinQuant.

## Validation

Executed validation:

- `python -m compileall candidates/202603300805_signed-hadamard-gptq/train_gpt.py`
  - **Result:** passed.
- `python - <<'PY' ... importlib.util.find_spec('torch') ... find_spec('sentencepiece') ... PY`
  - **Result:** both runtime dependencies were missing in this workflow container (`torch_spec=None`, `sentencepiece_spec=None`).
- Intended CPU-only smoke test: import the candidate with a stubbed `flash_attn_interface`, run the new int6 quantizer on synthetic matrices, and verify that rotation metadata round-trips through dequantization correctly.
  - **Result:** not feasible in this environment because the candidate cannot be imported without `torch`, and `torch` is not installed here.

Because of that dependency gap, the lightweight validation here is limited to syntax compilation plus dependency detection. A real runtime smoke should be rerun in the normal repository training environment where `torch`, `sentencepiece`, and `flash_attn_interface` are available.
