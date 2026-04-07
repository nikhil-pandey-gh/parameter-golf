# Bank-Aware LSQ-lite Late QAT on the LeakyReLU² + Legal TTT Stack

## Hypothesis

The current best stack already squeezes a lot out of architecture, optimizer overlap, and evaluation-time adaptation, but its biggest weight tensors still train in full precision and only see int6 distortion at export time. A compile-safe **late bank-aware fake-quant path** plus **learned per-matrix clip multipliers** should shrink the post-quantization gap without paying the cost of a broad architecture rewrite.

In short: keep the best model mostly unchanged, but finally train the banked transformer weights through something close to the quantizer they are evaluated with.

## Why this is promising here

Repository history points to three facts:

1. **Compression-aware training keeps paying off.** Mixed int6/int5, fp16 embeddings, GPTQ-lite clip search, and warmdown tuning are all part of the winning line.
2. **The best architecture already exists.** The 2026-03-23 record has the strongest overall stack, so a fresh candidate should attack a remaining bottleneck rather than restart architecture search.
3. **Late QAT on the 11-layer record family was never truly tested.** The 2026-03-21 Partial RoPE + LN Scale record explicitly notes that its late-QAT path was dead-code-eliminated by `torch.compile`, so the idea was underexplored by implementation failure rather than a clean negative result.

That makes a compile-safe late-QAT variant a better bet than revisiting layer recurrence, generic weight decay sweeps, or another large architectural fork.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best current overall stack: LeakyReLU(0.5)^2, legal score-first TTT, parameter banking, parallel Muon
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - GPTQ-lite per-row clip search and warmdown3500
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - strong note that late QAT did **not** activate under `torch.compile`
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`
  - long training still left a substantial post-quant gap, reinforcing that quantization itself remains a first-class bottleneck

## External research that informed it

- **Learned Step Size Quantization (LSQ)** — Esser et al., 2019 / ICLR 2020  
  <https://arxiv.org/abs/1902.08153>
  - motivates learning quantizer parameters jointly with the model instead of relying only on fixed heuristics
- **SmoothQuant** — Xiao et al., 2022 / ICML 2023  
  <https://arxiv.org/abs/2211.10438>
  - reinforces the value of moving quantization difficulty with lightweight scaling rather than changing the full architecture
- **SpinQuant** — Liu et al., 2024 / ICLR 2025  
  <https://arxiv.org/abs/2405.16406>
  - motivates making low-bit export more distribution-aware and learnable, especially when outliers dominate quantization error
- **BRECQ** — Li et al., 2021  
  <https://arxiv.org/abs/2102.05426>
  - additional evidence that post-training quantization quality depends strongly on local sensitivity and reconstruction-aware calibration

This candidate does **not** implement full LSQ, SmoothQuant, or SpinQuant. It borrows the shared lesson that quantization parameters should adapt to the weight distribution instead of being treated as a fixed afterthought.

## What changed vs. the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Bank-aware late fake quantization**
   - the large transformer bank tensors (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`) now get int6-style fake quantization during late training
   - this is where most of the model bytes live, so this is the part that matters most for export quality

2. **Compile-safe activation of late QAT**
   - instead of relying on a Python-side boolean that can be constant-folded out of the graph, the script recompiles once when late QAT begins
   - after that, a tensor-valued `bank_qat_mix` ramps the quantization strength smoothly from 0 to 1 over the late-training window

3. **LSQ-lite learned clip multipliers**
   - each bank slice gets a tiny learned clip multiplier (`bank_qat_logclip`)
   - these multipliers are reused during export quantization so train-time distortion better matches the final int6 roundtrip

4. **Candidate-directory-friendly defaults**
   - default `DATA_PATH` and `TOKENIZER_PATH` resolve relative to the repository root, so the script can be run directly from this candidate directory without patching paths

Everything else, including the base stack's fixed EMA export path and legal TTT evaluation flow, is intentionally left close to the 2026-03-23 record.

## How to run

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.20 \
BANK_QAT_ENABLED=1 BANK_QAT_BITS=6 BANK_QAT_CLIP_MIN=0.75 BANK_QAT_CLIP_MAX=1.25 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Defaults already point at the repository-root dataset and tokenizer:

- `../../data/datasets/fineweb10B_sp1024/`
- `../../data/tokenizers/fineweb_1024_bpe.model`

Override `DATA_PATH` / `TOKENIZER_PATH` if your layout differs.

If you also want the legacy fake-quant path on auxiliary `CastedLinear` modules (for example the bigram/value projection helpers), add `QAT_ENABLED=1`. The default candidate behavior keeps that path off and focuses the late-QAT budget on the banked transformer weights.

## Expected tradeoffs and risks

- **Late-stage throughput risk:** bank fake-quantization touches the biggest tensors in the model, so the late-training region may run slower than the current best stack.
- **Recompile overhead:** the script deliberately recompiles once when late QAT turns on; that avoids dead-code elimination but costs some wall-clock budget.
- **Quantizer mismatch risk:** the training path uses LSQ-lite clip multipliers plus per-row fake quant, while export still uses GPTQ-lite percentile search; the match is closer than before but not perfect.
- **Tuning sensitivity:** `LATE_QAT_THRESHOLD`, clip bounds, and ramp power may all need retuning if the quantized score regresses or if the model slows down too much.

## Validation

Commands run in this repository state:

```bash
python -m compileall candidates/202604071127_bank-qat-lsq/train_gpt.py
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604071127_bank-qat-lsq/train_gpt.py
```

Outcome:

- **Passed** syntax compilation for the candidate script.
- **Passed** the broader low-cost repository compile pass (`train_gpt.py`, `train_gpt_mlx.py`, `data/`, and this candidate script).

Additional smoke note:

- A runtime import smoke was attempted, but this container does not have the repository's Python dependencies installed (`numpy`, `sentencepiece`, `torch`) and also lacks `flash_attn_interface`. A full runtime smoke would additionally require CUDA, FlashAttention 3, and the FineWeb shards, so only compile-level validation was safe and feasible here.
