# Candidate: Late Bank-QAT Recompile

## Hypothesis

The strongest current script trains almost all of its parameter mass inside four bank tensors (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`), but its late-QAT path only affects `CastedLinear` modules. This candidate tests whether **late-stage STE fake quantization on the bank weights themselves** can reduce the train/export mismatch that still exists in the March 23 stack.

The key bet is simple: if the model is going to be exported through per-row int6 quantization anyway, then the dominant bank weights should see that constraint during the last part of training instead of only at export time.

## Why this is promising for this repository

This repo's best records repeatedly improved by reducing the quantization gap rather than by adding broad new infrastructure:

- `2026-03-19_WarmdownQuantization` argued that post-training quantization quality was the main bottleneck.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` documented that its late-QAT flag was effectively dead because `torch.compile` constant-folded the toggle.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved the frontier again with better export-time quantization.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` moved the frontier to `1.1194`, but still exports the bank tensors only after training.

That combination makes a late bank-QAT path with a one-time recompile a good next experiment: it directly targets a known mismatch in the current best stack without changing the core architecture, tokenizer, or evaluation protocol.

## Prior records and experiments that influenced this candidate

1. **`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`**  
   Chosen as the base because it is the current best script and already contains the banked parameterization, legal score-first TTT, and LeakyReLU² activation.

2. **`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`**  
   Reinforced that export-time quantization details still move the metric at the frontier.

3. **`2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`**  
   Explicitly reported that its late-QAT path never really activated under `torch.compile`, which motivated the recompile-based switch in this candidate.

4. **`2026-03-19_WarmdownQuantization`**  
   Helped frame quantization as the main remaining source of avoidable loss in these small-model runs.

## External research that informed it

- **Learned Step Size Quantization** — Esser et al. (2019/2020), <https://arxiv.org/abs/1902.08153>  
  Motivated using a late-stage STE quantization path rather than relying only on post-training export.

- **PACT: Parameterized Clipping Activation for Quantized Neural Networks** — Choi et al. (2018), <https://arxiv.org/abs/1805.06085>  
  Reinforced the idea that quantizer behavior should be part of training rather than only an afterthought at export.

- **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits** — Ma et al. (2024), <https://arxiv.org/abs/2402.17764>  
  Strengthened the broader low-bit-training motivation, even though this candidate stays far more conservative at int6.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate adds five focused changes:

1. **Bank-weight fake quantization**  
   Added an int6 STE fake-quant helper and applied it to the six banked weights used by each block (`q`, `k`, `v`, `out`, `mlp_up`, `mlp_down`) during the late-QAT phase.

2. **Late-QAT switch with one-time recompile**  
   When the warmdown scale crosses `LATE_QAT_THRESHOLD`, the script now enables both the old `CastedLinear` late-QAT path and the new bank-QAT path, then recompiles once so the late path is entered through a fresh graph instead of relying on the original compile.

3. **Disable fake quant before final eval/export**  
   The fake-quant paths are turned back off before the real post-training diagnostic, export, dequantized roundtrip, and legal TTT evaluation.

4. **Candidate-directory-relative defaults**  
   Default dataset and tokenizer paths now resolve from the repository root even when this script is run from inside the candidate directory.

5. **CPU smoke path and SDPA fallback**  
   Added a `SMOKE_TEST=1 DEVICE=cpu` path plus a `scaled_dot_product_attention` fallback so the candidate can be sanity-checked without CUDA or the `flash_attn_interface` import.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202604061334_bank-late-qat
```

### Full GPU run

This is intended to inherit the March 23 training recipe and add bank late-QAT:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
BANK_QAT_ENABLED=1 BANK_QAT_PERCENTILE=1.0 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Minimal CPU smoke check

```bash
DEVICE=cpu SMOKE_TEST=1 BANK_QAT_ENABLED=1 \
BIGRAM_VOCAB_SIZE=0 VE_ENABLED=0 XSA_LAST_N=0 \
NUM_LAYERS=2 MODEL_DIM=64 NUM_HEADS=4 NUM_KV_HEADS=2 \
MLP_MULT=2.0 TRAIN_SEQ_LEN=16 SMOKE_SEQ_LEN=16 SMOKE_BATCH_SIZE=1 \
python train_gpt.py
```

## Validation run in this workflow

1. `python -m compileall candidates/202604061334_bank-late-qat/train_gpt.py`  
   **Outcome:** passed.

2. After installing `numpy`, `torch`, and `sentencepiece` into a temporary venv, ran:

```bash
cd candidates/202604061334_bank-late-qat
DEVICE=cpu SMOKE_TEST=1 BANK_QAT_ENABLED=1 \
BIGRAM_VOCAB_SIZE=0 VE_ENABLED=0 XSA_LAST_N=0 \
NUM_LAYERS=2 MODEL_DIM=64 NUM_HEADS=4 NUM_KV_HEADS=2 \
MLP_MULT=2.0 TRAIN_SEQ_LEN=16 SMOKE_SEQ_LEN=16 SMOKE_BATCH_SIZE=1 \
python train_gpt.py
```

   **Outcome:** completed successfully and printed  
   `smoke_ok device=cpu seq_len=16 loss=6.943973 bank_qat_loss=6.943973 logits_shape=(1, 8, 1024)`.

## Main risks and tradeoffs

- **Throughput risk:** enabling fake quant on every banked matmul in the late phase may cost some steps.
- **Approximation risk:** the training-time fake quant uses a simple STE path with a row-max / optional percentile clip, not the full export-time GPTQ-lite search.
- **Tight artifact budget:** the March 23 stack is already very close to 16 MB, so any future extension that adds persistent state will be expensive.
- **Still unproven on 8xH100:** this workflow only validated syntax and a tiny CPU smoke path; it did not exercise the `torch.compile` recompile path on GPU.
