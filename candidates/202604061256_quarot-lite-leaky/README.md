# QuaRot-lite Leaky GPTQ candidate

## Hypothesis

The strongest clean training-only stack in this repo still appears quantization-limited, so a deterministic rotation-aided PTQ pass can improve post-quantization `val_bpb` without paying extra training time. This candidate keeps the 11L EMA + GPTQ-lite recipe from the best non-TTT record, adds the repo-proven `LeakyReLU(0.5)^2` activation from the current SOTA line, and replaces plain rowwise PTQ with a blockwise sign-Hadamard "QuaRot-lite" rotation that is undone after dequantization.

## Why this is promising here

- Repo evidence says the recent wins are mostly coming from **better quantization and small low-risk refinements**, not from large architecture swings.
- The best training-only record, [`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`](../../records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md), already shows that PTQ details like EMA and GPTQ-lite clip search still move the score.
- The latest main-track record, [`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`](../../records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md), shows that `LeakyReLU(0.5)^2` is a real repo-local gain.
- Earlier repo exploration and non-record work both suggest that heavier ideas like recurrence or slower gated MLP variants are risky under the 10-minute budget, so a zero-training-cost PTQ improvement is a better next bet.

## Prior repo experiments that influenced this candidate

1. **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
   - best clean training-only stack in the repo,
   - already tuned around EMA, warmdown, partial RoPE, LN scaling, VE, and GPTQ-lite clip search.
2. **Activation import:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
   - strongest evidence in-repo that `LeakyReLU(0.5)^2` is a low-risk quality win.
3. **Negative guidance:** recurrence and slower architectural changes in
   - `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`
   - `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600`
   both argue against spending the candidate budget on deeper reuse or heavier gated MLPs.

There were no prior `candidates/` entries in the repository when this candidate was created.

## External research that informed it

1. **QuaRot** — *Outlier-Free 4-Bit Inference in Rotated LLMs* ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456))
   - motivates using orthogonal rotations to make low-bit quantization easier,
   - specifically notes that 6-bit and 8-bit quantization can become much more forgiving after rotation.
2. **SpinQuant** — *LLM Quantization with Learned Rotations* ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406))
   - shows that some rotations are materially better than others,
   - motivates using a deterministic sign-Hadamard transform as a lightweight approximation to the broader rotation idea.
3. **SmoothQuant** — *Accurate and Efficient Post-Training Quantization for Large Language Models* ([arXiv:2211.10438](https://arxiv.org/abs/2211.10438))
   - reinforces the core strategy of improving quantization with offline equivalent transformations rather than longer training.

## What changed versus the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. switches the MLP from `ReLU^2` to **`LeakyReLU(0.5)^2`**,
2. adds **repo-root-relative dataset/tokenizer defaults** so the script works when run from the candidate directory,
3. adds a **deterministic blockwise sign-Hadamard rotation** before rowwise PTQ and the matching inverse after dequantization,
4. exposes the quantization transform as env-tunable knobs:
   - `QUANT_ROTATION_ENABLED=1`
   - `QUANT_ROTATION_BLOCK_SIZE=128`
5. adds a **FlashAttention fallback to PyTorch SDPA** for environments without `flash_attn_interface`,
6. adds a **`CPU_SMOKE_TEST=1` mode** that instantiates a small CPU model, runs a forward pass, exercises the quantization round-trip, and exits.

The rest of the architecture is intentionally unchanged: 11 layers, partial RoPE, LN scaling, XSA on the deepest layers, bigram hash embedding, smear gate, shared value embeddings, EMA export, and mixed int6/int8 quantization.

## How to run

Run from the candidate directory so the script uses its candidate-local defaults correctly:

```bash
cd candidates/202604061256_quarot-lite-leaky

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 ADAM_WD=0.04 WARMDOWN_ITERS=3500 ITERATIONS=9000 \
MLP_LEAKY_SLOPE=0.5 QUANT_ROTATION_ENABLED=1 QUANT_ROTATION_BLOCK_SIZE=128 \
MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lightweight local validation

Commands run for this candidate in this workflow:

1. `python -m compileall candidates/202604061256_quarot-lite-leaky/train_gpt.py`
   - **Outcome:** passed.
2. `CPU_SMOKE_TEST=1 python train_gpt.py`
   - **Outcome:** passed inside an isolated temp venv after installing `numpy`, `sentencepiece`, and `torch`.
   - The smoke config is sized to exercise the rotated PTQ path on attention, MLP, and int8 embedding tensors.
   - Printed: `cpu_smoke_ok loss:5.5797 roundtrip_loss:5.5919`

## Main risks and tradeoffs

- The rotation is a **small deterministic approximation** to the stronger learned-rotation ideas in SpinQuant, so the gain may be smaller than the paper-level results.
- The extra code size must still pay for itself under the 16 MB artifact cap.
- `LeakyReLU(0.5)^2` and the rotated PTQ path may interact nonlinearly; if this candidate regresses, the first follow-up ablations should separate the activation change from the PTQ change.
- The full GPU path still benefits from `flash_attn_interface`; the SDPA fallback is primarily there to keep smoke validation and portability practical.
