# Packed low-bit export + gated attention + BigramHash(3072)

## Hypothesis

The strongest recent stack in this repo is already highly optimized for training throughput and evaluation quality, so the next useful lever is **artifact efficiency** rather than another large architectural rewrite. This candidate keeps the 2026-03-23 record recipe, but tries to reclaim bytes with a packed mixed-bit export and log-domain scale double-quantization, then spends that headroom on a larger BigramHash table while also enabling gated attention by default to reduce attention outliers before quantization.

## Why this is promising here

The repo's record history is very consistent:

- lower-bit export repeatedly funded better models under the 16MB cap,
- larger lexical/context helpers like BigramHash kept paying off,
- and the top run is still close enough to the artifact ceiling that export efficiency is likely still a bottleneck.

This candidate follows that pattern directly: **compress harder, then buy back modeling capacity** instead of adding a new subsystem.

## Record influences

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest overall stack: LeakyReLU(0.5)^2, parameter banking + Parallel Muon, partial RoPE, XSA4, VE, EMA/SWA, legal score-first TTT.
- **Quantization inspiration:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - GPTQ-lite percentile search showed that post-training quantization quality still matters on top of the 11L/XSA stack.
- **Mixed-bit precedent:** `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - showed that MLP weights can tolerate lower precision than attention, and that saved bytes can be profitably reallocated to a larger BigramHash.
- **Bigram + SmearGate precedent:** `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - reinforced that lightweight lexical-memory features remain worthwhile in this regime.

There were **no prior `candidates/` directories** in this repository when this candidate was created.

## External research that informed this candidate

| Source | Relevant takeaway |
|---|---|
| [GPTQ (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323) | Low-bit post-training quantization can preserve LM quality when quantization is done carefully. |
| [AWQ (Lin et al., 2023/2024)](https://arxiv.org/abs/2306.00978) | Not all weights are equally important; protecting the most sensitive structures is better than uniform shrinking. |
| [Quantizable Transformers / gated attention (Bondarenko et al., 2023)](https://arxiv.org/abs/2306.12929) | Gated attention can suppress the outlier behavior that makes transformer layers harder to quantize. |
| [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) | Double-quantizing quantization metadata is a practical way to reduce storage overhead. |
| [Williams et al., 2024 calibration study](https://arxiv.org/abs/2311.09755) | Calibration choices matter; avoid adding a fragile calibration-heavy pipeline unless the repo already supports it. |

This candidate intentionally stays on the repo's existing percentile-search quantizer path instead of adding a full AWQ/GPTQ calibration stack, but it borrows the same **"preserve sensitive parts, compress cheap parts harder"** principle.

## What changed vs. the chosen base

Compared with `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

1. **Turns gated attention on by default** (`GATED_ATTENTION=1`).
2. **Raises the default BigramHash size to 3072** (`BIGRAM_VOCAB_SIZE=3072`).
3. **Makes the script runnable from the candidate directory** by resolving default dataset/tokenizer paths relative to the repository root instead of the current working directory.
4. **Replaces the export path with packed mixed-bit serialization**
   - MLP weights: **int5**
   - attention/projection weights: **int6**
   - remaining large tensors: existing per-row/per-tensor **int8**
   - low-bit payloads are **bit-packed**
   - row scales use **log-domain uint8 double-quantization** when large enough
   - final artifact stays on **lzma** compression
5. **Sets TTT defaults to the strongest published 2026-03-23 recipe**
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`

Everything else stays intentionally close to the top record: LeakyReLU(0.5)^2, partial RoPE, XSA on the last 4 layers, VE on late layers, EMA/SWA, and legal score-first TTT.

## How to run

From this candidate directory:

```bash
cd candidates/202604032116_packed-gated-bigram
RUN_ID=packed_gated_bigram \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The script now resolves `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root by default, so it can be launched directly from this directory.
- Override `BIGRAM_VOCAB_SIZE`, `GATED_ATTENTION`, or `TTT_ENABLED` with env vars if you want ablations or faster iteration. `BIGRAM_VOCAB_SIZE` should be `0` to disable bigrams entirely or `>= 2` when enabled.

## Validation

Commands run in this workspace:

```bash
cd candidates/202604032116_packed-gated-bigram
python -m compileall train_gpt.py
```

Outcome:

- `python -m compileall train_gpt.py` **passed**
- A minimal smoke launch was **not feasible here** because the expected repo-local FineWeb shards and tokenizer files were missing from `data/datasets/fineweb10B_sp1024/` and `data/tokenizers/`, so the candidate could not be started meaningfully in this container without extra data setup. The candidate also keeps the record branch's CUDA/FlashAttention execution path.

## Main risks and tradeoffs

- **Roundtrip risk:** int5 MLP packing plus scale double-quantization may save enough bytes to help artifact pressure, but the extra compression can widen the post-quantization gap if the scale encoding is too aggressive.
- **Gated-attention risk:** the outlier-reduction benefit may mostly help quantization rather than pre-quant loss, so the net gain depends on the export path actually reclaiming useful headroom.
- **Capacity-allocation risk:** if `BIGRAM_VOCAB_SIZE=3072` still leaves the artifact too close to the limit after a real run, the next ablation should be `2560` or `2048` while keeping the packed export.
- **Complexity risk:** this is a more complex serialization path than the prior records, so correctness of de/packing is the first thing to verify on a real GPU run.
