# 11L EMA + GPTQ-lite + activation-aware export scaling

## Hypothesis

The current record line is already strong on pre-quant quality, so the most promising next gain is to further shrink the **post-quantization gap** rather than change the 11-layer architecture again. This candidate adds a lightweight **activation-aware export transform** on top of the current best `11L + EMA + GPTQ-lite` stack: calibrate each large block linear on a small sample of **training tokens**, choose a per-layer channel scale that minimizes activation-weighted int6 reconstruction error, fold that scale into the weight columns, and store the inverse scale as a tiny runtime buffer.

In other words, this keeps the function of the full-precision model almost unchanged while making the exported int6 weights easier to quantize.

## Why this is promising for this repository

Repository history strongly suggests that the leaderboard is now dominated by techniques that reduce the quantization penalty rather than by large architectural jumps:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` shows that **GPTQ-lite clip search** and **EMA** each delivered about `-0.0006 BPB` with little or no extra training cost.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` shows that protecting quantization-sensitive tensors can collapse the quant gap.
- The 4-hour non-record run `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` improved pre-quant quality a lot but still left a large post-quant penalty, which is a strong sign that export-time compression remains a central bottleneck.

I explicitly did **not** choose shared-depth / recurrent blocks even though they are interesting externally, because this repo already has a negative result for layer recurrence under a fixed wallclock budget: `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` reports recurrence as the worst experiment in that sweep.

## Records and prior experiments that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Key inherited motifs:

- `11L`, `512d`, `8 heads / 4 KV heads`, `MLP 3x`
- `SmearGate + BigramHash`
- `XSA` on the last 4 layers
- `Partial RoPE (16/64)`
- `LN scale`
- `EMA + tight SWA`
- `GPTQ-lite` int6 export with per-row clip-percentile search
- `ValueEmbedding` on layers `9,10`

Additional record context:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` confirmed that **Partial RoPE + LN scale** mattered, while the late-QAT path was fragile under `torch.compile`.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md` established **EMA + XSA4** as a reliable improvement.
- `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/README.md` and `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md` showed that cheap token-pair inductive bias plus compression-funded capacity was the winning structural line.

There were no prior runs under `candidates/` when this candidate was created.

## External research that informed it

The implementation is deliberately modest, but it is motivated by the same basic idea that drives several strong PTQ papers:

- **AWQ** — *Activation-aware Weight Quantization for LLM Compression and Acceleration* (`arXiv:2306.00978`): activation statistics identify salient channels, and equivalent rescaling can protect them during weight-only quantization.
- **SmoothQuant** — *Accurate and Efficient Post-Training Quantization for Large Language Models* (`arXiv:2211.10438`): quantization difficulty can be migrated between activations and weights by a mathematically equivalent channel transform.
- **RPTQ** — *Reorder-based Post-training Quantization for Large Language Models* (`arXiv:2304.01089`): channel-wise range imbalance is a major quantization problem, not just isolated outliers.
- **QuaRot** (`arXiv:2404.00456`) and **SpinQuant** (`arXiv:2405.16406`) reinforce the broader point that equivalent transforms which reduce outlier severity can materially improve low-bit export.
- *Post Training Quantization of Large Language Models with Microscaling Formats* (`arXiv:2405.07135`) is also relevant because it explicitly studies combinations of GPTQ-, AWQ-, and SmoothQuant-style ideas instead of treating them as mutually exclusive.

## What changed versus the chosen base implementation

This candidate only changes the new candidate-local `train_gpt.py`; the repository root and all historical records are left untouched.

The main code changes are:

1. `CastedLinear` now owns a small `input_scale` buffer.
2. After EMA is applied, the script runs a short **training-token calibration pass** over the already-trained model.
3. Forward pre-hooks collect per-channel activation maxima for the large block linears that are exported at int6.
4. For each such linear, the exporter searches a small set of `alpha` values (`0.0, 0.35, 0.65` by default) and chooses the scale that minimizes an **activation-weighted int6 reconstruction proxy**.
5. The chosen scale is folded into the weight columns before export, and the inverse scale is stored in `input_scale` so inference remains approximately function-preserving.
6. In distributed runs, the activation stats are `all_reduce(MAX)`'d so every rank derives the same export transform.
7. The AWQ activation flag is recomputed automatically after `load_state_dict()`, and the `.ptz` blob carries an explicit compression-format header so reload does not depend on the current runtime guessing `zstd` vs `zlib`.

The actual quantizer is still the repo's familiar **GPTQ-lite style per-row int6 clip search**, so this is meant to **stack with** the current record rather than replace it.

## How to run / evaluate

From the candidate directory:

```bash
cd candidates/202603241504_activation-aware-export

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
AWQ_EXPORT_ENABLED=1 AWQ_CALIB_STEPS=4 AWQ_CALIB_SEQS=8 \
AWQ_ALPHA_CANDIDATES=0.0,0.35,0.65 AWQ_MAX_SCALE=4.0 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablation:

```bash
AWQ_EXPORT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This uses the same dataset and tokenizer environment variables as the record it forks from.

## Main expected risks and tradeoffs

- The added `input_scale` buffers consume some extra artifact bytes, so a win on quantization quality still has to survive the 16MB budget.
- The transform is exactly equivalent in real arithmetic, but only approximately equivalent in bf16 practice.
- Calibration adds a little post-training wallclock overhead.
- Because this is still a weight-only int6 export, gains may be modest if the remaining bottleneck is elsewhere.
- If the calibration sample is unrepresentative, some layers may choose unhelpful scales; the per-layer alpha search is meant to reduce that risk.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202603241504_activation-aware-export/train_gpt.py
python -m pip show torch
```

Outcomes:

- `python -m compileall ...` passed.
- `python -m pip show torch` reported `WARNING: Package(s) not found: torch` in this runner.
- Because the local Python environment here does not have PyTorch installed, I could not run a meaningful import/forward smoke test of the candidate script without adding heavyweight dependencies. In a normal repo training environment with `requirements.txt` installed, the natural next smoke check is a tiny `torchrun` startup test or a CPU-side import test with a stubbed attention kernel.
