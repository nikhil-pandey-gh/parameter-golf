# AWQ-lite Activation-Aware Int6 Clip Search

## Hypothesis

The strongest pre-TTT stack in this repo already proved that **weight-only GPTQ-lite clip search** buys a small but real gain on top of the 11-layer XSA/VE/bigram architecture. The next likely win is to make that clip search **activation-aware**: choose each int6 row clip percentile using the channels the model actually uses most, not plain weight MSE.

Concretely, this candidate runs a few extra **late-training probe forwards on the current training batches while training is still in progress**, measures per-input-channel RMS activations for each `CastedLinear`, and uses those statistics to weight the per-row reconstruction error during int6 clip selection. The expectation is a smaller post-quantization gap than the current weight-only GPTQ-lite export.

## Why this looks promising here

Repository history says the frontier is now bottlenecked by export quality almost as much as by training quality:

- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` shows a 4-hour run reaching much better pre-quant quality than its final post-quant score.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` already got another `-0.0006` BPB from **weight-only** GPTQ-lite clip search with zero training cost.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` documents that the repo's late-QAT toggle was dead-code-eliminated by `torch.compile`, so a clean PTQ improvement path is especially attractive.

That makes activation-aware PTQ a good fit: it attacks a known bottleneck, costs almost nothing at training time, and does not require the complexity jump of the 2026-03-23 TTT + parameter-banking stack.

## Prior runs that informed this candidate

- **Chosen base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
  - 11 layers, MLP3x, XSA on last 4 layers, partial RoPE, LN scale, VE128, EMA, tight SWA, BigramHash, SmearGate.
- **Negative guidance:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - late QAT looked promising on paper but was not actually active in the compiled path.
- **Why not just clone the current record:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - the latest record's final edge comes from LeakyReLU² plus legal TTT and parallel-banked Muon; this candidate instead targets the cleaner non-TTT export path.

## External research

The implementation is mainly inspired by the common thread across these quantization papers:

| Paper | Main takeaway | Relevance here |
|---|---|---|
| **GPTQ** — Frantar et al., 2022, arXiv:2210.17323 | One-shot weight-only PTQ matters a lot for autoregressive transformers. | The 2026-03-22 record already validated a tiny GPTQ-lite version of this idea in this repo. |
| **AWQ** — Tang et al., 2024, arXiv:2306.00978 | Activation statistics identify salient channels better than weights alone. | This candidate ports that intuition into the existing per-row clip search with minimal code. |
| **SmoothQuant** — Xiao et al., 2023/2024, arXiv:2211.10438 | Equivalent transforms can move quantization difficulty using activation information. | Supports the broader idea that activation-aware calibration is a good next lever for export quality. |
| **QuaRot / SpinQuant** — Croci et al., 2024, arXiv:2404.00456; Liu et al., 2025, arXiv:2405.16406 | Outlier suppression via rotations can further improve quantization. | Promising, but too invasive for a surgical single-script candidate. This candidate keeps the lighter AWQ-style step instead. |

## What changed vs the base implementation

1. **Activation-aware int6 clip search**
   - `quantize_int6_per_row(...)` now accepts optional per-channel activation RMS.
   - Clip-percentile search still tries the same GPTQ-lite style candidates, but the score is now the activation-weighted row reconstruction error instead of plain weight MSE.

2. **Training-time calibration path**
   - During late warmdown, the script runs a small number of extra probe forwards on the current training batches while the run is still inside the training loop.
   - Those probe forwards record per-input-channel RMS for every `CastedLinear`.
   - After training, only the aggregated activation statistics are reused for export; no training batches are replayed or reread.

3. **Disable broken late-QAT by default**
   - `LATE_QAT_THRESHOLD` now defaults to `0.0`.
   - This candidate is intentionally a PTQ-first experiment rather than another latent-QAT variant.

Everything else stays aligned with the 2026-03-22 base stack.

## How to run

From this candidate directory:

```bash
cd candidates/202604021440_awq-lite-int6

SEED=1337 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
AWQ_ENABLED=1 AWQ_CALIBRATION_BATCHES=8 AWQ_CAPTURE_SCALE=0.15 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still exports an int6+zstd artifact and prints the roundtrip/sliding-window validation metrics exactly like the recent record scripts.

## Main risks and tradeoffs

- The cached late-training batches may not perfectly match the final EMA model's activation distribution.
- Using only a handful of calibration batches could add seed-to-seed noise.
- Activation-aware clipping may preserve important channels better while also keeping slightly more outlier mass, which could offset some compression gains.
- This is deliberately a light-touch AWQ-style adaptation, not a full equivalent-transform or rotation-based quantizer.

## Validation

Commands run in this repo:

```bash
python -m compileall candidates/202604021440_awq-lite-int6/train_gpt.py
python - <<'PY'
import importlib.util
for name in ('torch', 'flash_attn_interface', 'sentencepiece', 'numpy', 'zstandard'):
    print(name, bool(importlib.util.find_spec(name)))
PY
```

Observed outcome:

- `compileall` succeeded for `train_gpt.py`.
- A real smoke run was **not feasible in this runner** because `torch`, `flash_attn_interface`, `sentencepiece`, and `numpy` are not installed here, while this candidate inherits the CUDA/FlashAttention execution path from the record stack.
