# Cubic MLP Equalization before GPTQ-lite Export

## Hypothesis

The current 11-layer leaderboard stack looks more constrained by **post-training int6 export damage** than by raw pre-quant model quality. In this model family, each MLP hidden channel can be rescaled exactly because the nonlinearity is `LeakyReLU(x, 0.5)^2`: if `fc` rows are multiplied by `s`, the paired `proj` columns can be divided by `s^2` with no change to the exact floating-point function.

The candidate uses that invariance to flatten MLP outliers **before** GPTQ-lite int6 export. The expectation is a smaller quantization penalty on the largest weight blocks in the model, with no added train-time cost and no architectural change.

## Why this is promising here

Repository history points at the same bottleneck repeatedly:

- `2026-03-18_FP16Embed_WD3600` showed that reducing export damage on the tied embedding was worth almost the entire improvement.
- `2026-03-19_WarmdownQuantization` explicitly framed the challenge as "training for compression".
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` got another gain from better clip selection during int6 export.
- `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` and `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` both suggest that the MLP stack is one of the biggest byte and quality levers in this regime.

That makes a **data-free, function-preserving pre-quantization transform for the MLP pair** a good fit: it attacks the same failure mode as the winning quantization tweaks, but without adding a new training objective or broad new infrastructure.

## Records and prior experiments that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Quantization reference:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **MLP/artifact pressure:** `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
- **Compression-aware training evidence:** `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`

## External research that informed it

- **SmoothQuant** (arXiv:2211.10438): mathematically equivalent offline scaling can move quantization difficulty without changing model function.
- **QuaRot** (arXiv:2404.00456): function-preserving rotations remove outliers and improve PTQ.
- **SpinQuant** (arXiv:2405.16406): learned rotations materially improve quantized accuracy over simpler transforms.
- **FlatQuant** (arXiv:2410.09426): flattening weight/activation distributions before quantization remains a strong PTQ direction.
- **KurTail** (arXiv:2503.01483) and **OptRot** (arXiv:2512.24124): recent work keeps pushing cheap outlier-suppression transforms for better PTQ.

This candidate intentionally takes the cheapest repo-compatible slice of that idea family: **MLP-only, data-free, export-time scaling**, rather than learned rotations or activation calibration.

## What changed versus the chosen base

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate adds one new export-time step:

1. Unbank the MLP weights as usual.
2. For each `blocks.{i}.mlp.fc.weight` / `blocks.{i}.mlp.proj.weight` pair, compute a channel scale from the cube-root ratio of:
   - `proj` column outlier scale, and
   - `fc` row outlier scale.
3. Normalize and clamp the scale, then try strengths `{0.25, 0.5, 0.75, 1.0}`.
4. Pick the strength that minimizes the same row-wise int6 proxy MSE already used by the export path.
5. Quantize the transformed weights with the existing GPTQ-lite-style per-row clip search.

The transform is exact in floating point for positive scales:

```python
fc'   = diag(s) @ fc
proj' = proj @ diag(s^-2)
```

because `LeakyReLU(sx, 0.5)^2 = s^2 * LeakyReLU(x, 0.5)^2` for `s > 0`.

New knobs:

- `MLP_EQ_ENABLED=1` (default)
- `MLP_EQ_MAX_SCALE=4.0`

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202604091029_cubic-mlp-eq
RUN_ID=cubic_mlp_eq \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For an ablation against the same script, disable the new export transform:

```bash
cd candidates/202604091029_cubic-mlp-eq
RUN_ID=cubic_mlp_eq_noeq \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MLP_EQ_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604091029_cubic-mlp-eq/train_gpt.py` | Passed in this environment |
| Pure-Python algebra check of `fc *= s`, `proj /= s^2` | Passed with `mlp_eq_pure_python_max_err=1.421085e-14` |

### CPU-only smoke test

Not feasible in this environment without adding extra infrastructure:

- the container does not have `torch`,
- it does not have `flash_attn_interface`,
- and this training stack explicitly requires CUDA at runtime.

## Main risks and tradeoffs

- The proxy MSE used to choose the equalization strength may not correlate perfectly with validation BPB.
- The transform currently targets **MLP weights only**; attention outliers may still dominate the remaining quant gap.
- Export-time CPU work increases modestly because each MLP layer tries a small scale grid before quantization.
- Because the base is the current best TTT stack, gains from the new quantization path may be small in absolute BPB even if the idea is correct.
