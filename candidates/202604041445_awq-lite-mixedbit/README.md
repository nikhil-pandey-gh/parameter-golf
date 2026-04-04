# AWQ-lite Mixed-Bit Export on the 11L EMA + GPTQ-lite Base

## Hypothesis

The current pure train/export stack is already very strong on the modeling side, so the next clean win is to reduce **post-training quantization damage** instead of making training slower or the architecture broader. The hypothesis here is that an **AWQ-style activation-aware export** can protect the most important input channels before per-row quantization, and that a **small number of conservative int8 rescues** can spend the remaining artifact headroom on the tensors that calibration says are most sensitive.

## Why this is promising here

Recent records suggest the repo is in a mature regime where incremental gains mostly come from better export and evaluation, not from another large architectural rewrite:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` showed that a stronger post-training quantizer alone was worth another improvement on top of the mature 11-layer stack.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/` and `records/track_10min_16mb/2026-03-19_WarmdownQuantization/` both showed that the challenge is heavily constrained by export quality and byte allocation, not only pre-quant training loss.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` suggests the modeling stack is already very optimized; the main untapped room in the pure train/export path is a better quantization budget.

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed this candidate

- **AWQ** ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)) argues that activation-aware channel protection can substantially reduce low-bit weight-only quantization error by scaling salient channels before quantization.
- **HAWQ-V2** ([arXiv:1911.03852](https://arxiv.org/abs/1911.03852)) motivates allocating higher precision only to the most sensitive layers/tensors instead of treating every matrix the same.
- **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) and **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)) reinforce the same general lesson: outliers and channel sensitivity are a major limiter for low-bit PTQ. I did **not** implement rotation-based quantization here because it would require broader graph surgery than this candidate aims to add.

## Base implementation

This candidate forks the pure train/export record:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

That base already contains the strongest non-TTT recipe in the repo:

- 11 layers, 512 width, 8H/4KV GQA
- 3x MLP
- XSA on late layers
- Partial RoPE + LN scaling
- VE128
- EMA
- GPTQ-lite-style per-row clip search

## What changed versus the base

1. **Candidate-local default paths**
   - The default dataset and tokenizer paths now resolve relative to the repository root, so `train_gpt.py` can be launched directly from this candidate directory.

2. **Activation calibration pass**
   - After EMA weights are applied, the script runs a short calibration pass over a few train batches and records RMS input activations for each large `CastedLinear` weight tensor.

3. **AWQ-lite per-column protection**
   - For attention and MLP matrices, calibration RMS values are converted into bounded per-column scales.
   - The exporter quantizes `W / s` and stores `s`, then reconstructs `W_hat = dequantize(W / s) * s` during round-trip evaluation.
   - This keeps the runtime graph simple while still using activation-aware channel protection.

4. **Conservative mixed-bit rescue**
   - The exporter computes both int6 and int8 candidates for large attention/MLP matrices and compares their activation-weighted reconstruction error.
   - Only the top few tensors with the largest relative gain are promoted to int8 (`AWQ_INT8_RESCUE_TOPK`, default `4`).
   - Everything else remains on the cheaper int6 path, and embeddings/non-block tensors keep the existing behavior from the base record.

5. **AWQ-specific artifact + logging**
   - The saved export artifact is now `final_model.awq.ptz`.
   - Export logs report calibration settings, how many tensors received AWQ column scales, and which tensors were rescued to int8.

## How to run

From this candidate directory:

```bash
cd candidates/202604041445_awq-lite-mixedbit
RUN_ID=awq_lite_mixedbit \
SEED=1337 \
AWQ_ENABLED=1 \
AWQ_CALIB_BATCHES=4 \
AWQ_INT8_RESCUE_TOPK=4 \
EXPORT_RESERVE_SECONDS=20 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

- `AWQ_CALIB_BATCHES` — more calibration data, more export-time overhead
- `AWQ_ALPHA` — strength of activation-aware scaling
- `AWQ_SCALE_MIN` / `AWQ_SCALE_MAX` — clamp range for protected channels
- `AWQ_INT8_RESCUE_TOPK` — number of tensors allowed to escape to int8
- `AWQ_INT8_RESCUE_MIN_GAIN` — minimum relative error reduction required for rescue
- `EXPORT_RESERVE_SECONDS` — reduces the **timed training-loop** budget to leave room for AWQ export work

## Expected risks / tradeoffs

- **Calibration overhead:** this adds export-time work after training.
- **Heuristic budgeter:** the int8 rescue policy is intentionally conservative and tensor-count-based, not a full byte-accurate optimizer.
- **Rank-local calibration:** the current calibration stats come from the local rank stream used during export, which is simple but not globally aggregated.
- **Possible no-op outcome:** the current GPTQ-lite exporter is already strong, so this candidate may only help if the activation-aware scaling meaningfully targets the remaining quantization outliers.
- **Inherited timing semantics:** like the base record, `MAX_WALLCLOCK_SECONDS` governs the timed main training loop rather than total process wallclock; `EXPORT_RESERVE_SECONDS` only shortens that timed region.

## Validation

Commands run:

```bash
python -m compileall candidates/202604041445_awq-lite-mixedbit/train_gpt.py
```

Outcome:

- `compileall` succeeded.
- A CPU helper smoke test for the new quantization path was **not feasible on this runner** because the repository runtime dependencies are not installed here (`torch` import failed), and the full trainer also hard-requires CUDA + FlashAttention for end-to-end startup.
- A code review pass found and I fixed two concrete issues: stale legacy metric keys in AWQ sliding-window logs, and missing explicit export-reserve handling/documentation for the timed training-loop budget.
