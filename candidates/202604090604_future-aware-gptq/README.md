# Future-Aware GPTQ-lite on the 11L XSA/EMA Base

## Hypothesis

The late-March record stack looks close to saturated on training-side tweaks, but still leaves measurable loss on post-training int6 export. A lightweight **future-aware clip search** for the most sensitive late-layer matrices should beat pure weight-MSE GPTQ-lite calibration, especially when paired with the already-proven **LeakyReLU(0.5)^2** MLP on the stable 11-layer XSA/EMA/VE base.

## Why this is promising here

- The repo history shows that **quantization is still the main bottleneck** once the model reaches the stronger 10L/11L recipes.
- Extra compute alone was not enough: the non-record 4-hour baseline still lagged because export quality stayed weak.
- The best late records improved mostly via **better averaging and better export heuristics**, not another large architectural jump.
- Recent PTQ work keeps finding gains from **downstream-aware calibration** rather than plain local reconstruction error.

## Prior records that shaped this candidate

- **`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`** is the chosen base because it is the strongest stable pre-TTT stack in the repo: 11 layers, XSA on late layers, partial RoPE, LN scaling, shared value embeddings, EMA, and GPTQ-lite int6 export.
- **`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`** contributed the LeakyReLU(0.5)^2 MLP, which was one of the cleanest late gains.
- Early records such as **`2026-03-19_MixedQuant_Int6Int8_SlidingWindow`**, **`2026-03-19_MLP3x_QAT_Int6_SlidingWindow`**, and **`2026-03-18_FP16Embed_WD3600`** reinforced the same lesson: **export precision choices matter disproportionately**.
- The repo has **no prior `candidates/` directory**, so this is the first candidate iteration branch.

## External research that informed it

- **SpinQuant** (arXiv:2405.16406) — learned rotations show that downstream quantization quality depends strongly on how calibration interacts with later layers, not just local weight error. <https://arxiv.org/abs/2405.16406>
- **FlatQuant** (arXiv:2410.09426) — argues that flatter transformed distributions improve PTQ beyond simple clipping heuristics. <https://arxiv.org/abs/2410.09426>
- **Future-Aware Quantization / FAQ** (arXiv:2602.02538) — explicitly uses future-layer activations to choose quantization hyperparameters with low overhead. <https://arxiv.org/abs/2602.02538>
- **SERQ** (arXiv:2603.08185) — further supports saliency-aware, error-aware post-training correction as an active frontier after basic GPTQ/AWQ-style methods saturate. <https://arxiv.org/abs/2603.08185>

## What changed vs the chosen base

1. **LeakyReLU(0.5)^2 MLP**
   - Replaces `relu^2` with the March 23 record's `leaky_relu(0.5)^2`.

2. **Future-aware GPTQ-lite clip search**
   - The base record's GPTQ-lite export picked clip percentiles by minimizing **weight reconstruction MSE**.
   - This candidate adds a lightweight **train-split calibration batch** and searches clip percentiles for the **last 4 transformer blocks** by minimizing **downstream token loss** instead.
   - The scope is intentionally limited to late layers to keep export-time cost reasonable.

3. **Pragmatic runtime fallbacks**
   - Falls back from FlashAttention 3 to PyTorch SDPA when `flash_attn_interface` is unavailable.
   - Supports CPU startup/smoke paths.
   - Auto-picks the smallest **portable** compressor among `lzma` and `zlib`, with self-describing artifact headers. `zstd` remains available only when explicitly requested.

4. **Late QAT de-emphasized**
   - The base family's late-QAT path was a fragile lever in prior repo history, so this candidate defaults it off and focuses on the export calibration change instead.

## How to run

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
FAQ_ENABLED=1 FAQ_LAST_N=4 FAQ_CALIB_SEQS=4 \
MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_SLOPE=0.5 \
COMPRESSION_MODE=auto \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If FlashAttention is not installed, the script will fall back automatically. `COMPRESSION_MODE=auto` stays on stdlib compressors; `COMPRESSION_MODE=zstd` is optional and requires the package explicitly.

## How to evaluate

- Standard roundtrip eval is still produced at the end of training.
- Sliding-window eval remains available through `EVAL_STRIDE` (default `64`).
- The candidate logs the chosen compressor and every future-aware clip decision.

## Main risks / tradeoffs

- **Export-time overhead:** future-aware clip search runs extra forwards after training.
- **Modest upside:** late-layer clip selection may only recover a small amount of the remaining quant gap.
- **Calibration sensitivity:** using too few calibration sequences could make clip choices noisy; using too many makes export slower.
- **No training-speed win:** unlike parameter banking or recurrence ideas, this candidate aims to improve artifact quality, not raw throughput.

## Validation

Commands run here:

```bash
python -m compileall candidates/202604090604_future-aware-gptq/train_gpt.py
```

Outcome:

- `compileall` **succeeded**.
- A real CPU smoke run was **not feasible in this environment** because the runtime image does not include `torch`, the repository does not ship local tokenizer/shard assets, and an attempted temporary virtualenv install of the CPU wheel was blocked by the network proxy. The script still includes CPU / no-FlashAttention fallbacks so that smoke validation is possible in a normal Python environment with the declared dependencies installed.
