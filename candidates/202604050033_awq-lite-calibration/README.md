# AWQ-lite calibration on the 11L GPTQ-lite stack

## Hypothesis

The recent 11-layer records suggest this repo is now limited more by **post-training quantization loss** than by pre-quantized model quality. A short activation-only calibration pass should make the existing GPTQ-lite int6 export more sensitive to high-usage channels and cut the roundtrip loss without slowing training.

## Why this is promising here

- The strongest clean pre-TTT stack is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which already won with **EMA + GPTQ-lite clip search** but still scores clip candidates by plain weight MSE.
- Earlier records repeatedly showed that export quality is a first-order lever: `2026-03-19_WarmdownQuantization`, `2026-03-18_FP16Embed_WD3600`, and the later int6/GPTQ-lite records all improved mostly by shrinking the quantization penalty.
- The repo has already explored XSA, partial RoPE, LN scaling, SmearGate, BigramHash, VE, SWA/EMA, sliding eval, and TTT. It has **not** explored activation-aware PTQ/AWQ-style calibration.
- There were **no prior `candidates/` directories** in this checkout, so this idea is not duplicating an earlier candidate iteration.

## Prior repo influence

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Architecture stack carried forward:** `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` and `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
- **Quantization-first framing:** `2026-03-19_WarmdownQuantization` and `2026-03-18_FP16Embed_WD3600`
- **What I explicitly avoided repeating:** naive layer recurrence (negative result in `track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`) and broader TTT/Parallel-Muon changes from `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`

## External research that informed this candidate

- **AWQ** — Tang et al., *Activation-aware Weight Quantization for LLM Compression and Acceleration*, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978).  
  Main takeaway used here: protect salient channels using activation statistics instead of weight magnitude alone.
- **SmoothQuant** — Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, [arXiv:2211.10438](https://arxiv.org/abs/2211.10438).  
  Main takeaway used here: a tiny offline calibration pass can move quantization difficulty into a form the weight quantizer handles better.
- **QuaRot** — Ashkboos et al., *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*, [arXiv:2404.00456](https://arxiv.org/abs/2404.00456).  
  Main takeaway: outlier suppression matters, but the full rotation route is too invasive for this repo’s minimal-diff workflow.
- **SpinQuant** — Liu et al., *SpinQuant: LLM Quantization with Learned Rotations*, [arXiv:2405.16406](https://arxiv.org/abs/2405.16406).  
  Main takeaway: learned rotations can beat simpler PTQ heuristics, but again require much broader infrastructure than this candidate.
- **LLM-QAT** — Liu et al., *LLM-QAT: Data-Free Quantization Aware Training for Large Language Models*, [arXiv:2305.17888](https://arxiv.org/abs/2305.17888).  
  Main takeaway: training-aware low-bit methods can help, but a calibration-only step is a better fit than adding more training-time overhead here.

## What changed vs the chosen base implementation

Base file: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

1. **Made the script runnable from this candidate directory** by resolving default dataset and tokenizer paths relative to the script location instead of the current working directory.
2. **Added a short post-EMA calibration pass** over training tokens that collects per-input-channel RMS activations for the large `CastedLinear` layers that are exported to int6.
3. **Replaced plain weight-MSE clip scoring** in `quantize_int6_per_row()` with activation-weighted reconstruction error, so the clip percentile search cares more about channels the model actually uses.
4. **Added an AWQ-lite equivalent input scaling search** over a small alpha grid before per-row int6 quantization. If scaling helps a matrix, the candidate stores a tiny fp16 `input_scale` vector and inverts it on load.
5. **Made the exported artifact self-describing** by prefixing a `ZSTD` or `ZLIB` header before the compressed payload and choosing the decompressor from that header at load time.
6. **Left the training stack unchanged otherwise**: same 11-layer 512d model, XSA on the last 4 layers, partial RoPE, LN scaling, VE, SmearGate, BigramHash, EMA, warmdown3500, and the same sliding-window evaluation/export path.

## How to run or evaluate

From this directory:

```bash
cd candidates/202604050033_awq-lite-calibration
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
AWQ_ENABLED=1 AWQ_CALIBRATION_STEPS=8 AWQ_CALIBRATION_BATCH_TOKENS=131072 \
AWQ_CALIBRATION_SEQ_LEN=1024 AWQ_ALPHA_VALUES=0.0,0.25,0.5,0.75,1.0 \
AWQ_MAX_INPUT_SCALE=4.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you launch `train_gpt.py` from this candidate directory, the default `DATA_PATH` and `TOKENIZER_PATH` now resolve back to the repo’s root `data/` directory automatically.

## Main expected risks and tradeoffs

- The calibration pass adds a small post-training runtime cost.
- The artifact now stores a few extra fp16 per-matrix scale vectors; the overhead should stay small, but it is not free.
- If `zstandard` is unavailable, the candidate falls back to zlib automatically. That keeps the script runnable, but the artifact may compress worse than the zstd path.
- A tiny calibration slice can be noisy. If the sampled activation statistics are not representative, the activation-aware weighting could help some matrices and hurt others.
- This is still **weight-only PTQ**. It does not tackle activation/KV quantization the way SmoothQuant, QuaRot, or SpinQuant do.
- A real end-to-end smoke run still needs the repo’s runtime dependencies plus a CUDA/FlashAttention environment.

## Validation

| Command | Outcome |
| --- | --- |
| `python -m compileall train_gpt.py train_gpt_mlx.py data` | Passed in the repository root before candidate edits. |
| `python -m compileall candidates/202604050033_awq-lite-calibration/train_gpt.py` | Passed. |
| Stubbed no-training module load via `python - <<'PY' ...` | Attempted, but this container does not have the repo runtime dependencies installed (`numpy`, `torch`, and `sentencepiece` were all missing), so the smoke load failed before execution reached the candidate code. |
