# SliderQuant-lite on the 11L EMA + GPTQ-lite stack

## Hypothesis

The current 11-layer EMA + GPTQ-lite record is already strong on training-side quality, so the next cheap win is more likely to come from **better post-training quantization** than from a large architectural rewrite.

This candidate tests a lightweight version of the recent **SliderQuant** intuition: quantization sensitivity is not uniform across depth, and the **first / last layers plus the tied token table** deserve different treatment than the middle of the network. The hypothesis is that replacing the current one-size-fits-all symmetric row-wise export with a **layer-aware affine search on the most sensitive tensors** can reduce roundtrip error at essentially zero training-time cost.

## Why this is promising for this repository

The repo history is unusually clear that export quality matters a lot here:

- `2026-03-18_FP16Embed_WD3600` showed that the tied embedding is disproportionately sensitive to quantization.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved the best clean non-TTT stack by adding a simple clip search in PTQ.
- Longer or more exotic training alone has not been enough when the post-quantization gap remains large.

That makes this repository a good fit for a PTQ-focused candidate: if the export gets better, the challenge metric gets better without paying extra training wall-clock.

## Prior records and experiments that informed this candidate

Primary local influences:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the base because it is the strongest clean PTQ-oriented stack in the repo.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - strong evidence that the tied embedding / output interface is unusually fragile under quantization.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - confirms that later gains mostly came from activation + TTT, so a quantization-only experiment is easiest to interpret if it starts from the cleaner 2026-03-22 base.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - helpful negative result showing that naive layer recurrence is a bad fit for the 10-minute wall-clock budget.

## External research that informed it

- **SliderQuant: Accurate Post-Training Quantization for LLMs** (`arXiv:2603.25284`)
  - motivates the main idea: shallow/deep layers, especially the first and last, are more quantization-sensitive than middle layers, so PTQ should not treat all layers identically.
- **Rethinking Weight Tying: Pseudo-Inverse Tying for Stable LM Training and Updates** (`arXiv:2602.04556`)
  - not implemented directly here, but it reinforced the repo’s own observation that the tied token embedding / unembedding interface is a special case and worth handling more carefully during export.

## What changed versus the chosen base implementation

This candidate starts from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py` and keeps the training stack intact:

- 11 layers, 512d, 8 heads / 4 KV heads
- 3x MLP
- XSA on the last 4 layers
- Partial RoPE (16 / 64)
- LN scaling
- SmearGate + BigramHash + VE128
- EMA + warmdown3500

The new changes are deliberately narrow:

1. **Layer-aware PTQ sensitivity detection**
   - `tok_emb.weight`, `lm_head.weight`, and the first / last `SLIDERQUANT_EDGE_LAYERS` transformer blocks are treated as export-sensitive.

2. **Affine row-wise search for sensitive tensors**
   - For sensitive tensors only, export now compares the base symmetric quantizer against a small **affine quantization** search over low/high row quantiles.
   - If affine wins on reconstruction MSE, the export stores per-row zero-points in addition to scales.
   - Middle layers keep the original cheaper symmetric GPTQ-lite-style path.

3. **Runner-friendly defaults and fallback paths**
   - Default data/tokenizer paths resolve relative to the repository root so the script can be run from this candidate directory directly.
   - If `flash_attn_interface` is unavailable, attention falls back to `torch.nn.functional.scaled_dot_product_attention`.
   - `SMOKE_TEST=1` adds a synthetic no-dataset validation path that builds the model, runs forward/backward, quantizes, dequantizes, reloads, and checks a roundtrip loss.

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603291552_sliderquant-lite
```

Standard training / eval run (same overall recipe as the 2026-03-22 base):

```bash
RUN_ID=sliderquant_lite \
SEED=1337 \
SLIDERQUANT_EDGE_LAYERS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Local synthetic smoke test:

```bash
SMOKE_TEST=1 RUN_ID=smoke python train_gpt.py
```

Notes:

- `SLIDERQUANT_EDGE_LAYERS=1` means only the first and last transformer blocks get the more expensive affine-search comparison. That is the current default.
- The candidate is intentionally **export-focused**. It does not try to stack yet another training-side architectural change on top of the base.

## Validation

Commands run for this candidate in the workflow:

```bash
python -m compileall train_gpt.py
```

Outcome: passed.

```bash
SMOKE_TEST=1 RUN_ID=smoke python train_gpt.py
```

Outcome: passed in a temporary virtualenv with repo dependencies installed for validation.

Observed output:

```text
smoke_test_ok device:cpu loss:6.9469 roundtrip_loss:6.9506
```

This is only a startup / export sanity check on synthetic tokens, not a real FineWeb score.

## Main expected risks and tradeoffs

- **Artifact size risk**: affine-sensitive tensors now store per-row zero-points, so the export may gain quality but lose some compression headroom.
- **Export-time cost**: the affine candidate search adds some CPU-side export overhead.
- **Unverified on 8xH100**: this workflow only ran syntax + smoke validation, not a real training run.
- **May be too conservative**: limiting the affine path to edge layers might leave gains on the table, while expanding it further might hurt artifact size.

## Suggested next experiments

1. Sweep `SLIDERQUANT_EDGE_LAYERS` over `1` vs `2`.
2. Log per-tensor quantization choices to see whether affine wins mostly on `tok_emb.weight`, early attention, or late MLPs.
3. If the roundtrip gap improves without breaking the size budget, stack this export on the later LeakyReLU² / TTT codebase.
