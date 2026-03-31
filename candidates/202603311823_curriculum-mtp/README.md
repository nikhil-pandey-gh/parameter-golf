# Curriculum MTP on the 11L EMA + GPTQ-lite base

## Hypothesis

A tiny-model-friendly **forward curriculum for multi-token prediction (MTP)** can improve sample efficiency on this repository's strong 11-layer non-TTT stack without adding artifact bytes at submission time.

The core idea is to train with a **single auxiliary future-token head** but **ramp its loss weight in gradually**. This is meant to preserve the small-model stability of next-token training early on while still giving the trunk a richer future-prediction signal once the main language model has started to organize its representations.

## Why this looks promising for this repository

The current record history shows a few clear patterns:

- The best non-TTT base is the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack, which already combines the strongest durable ingredients in this repo: 11 layers, MLP3x, XSA-last4, partial RoPE, LN scale, EMA, and GPTQ-lite export.
- Most of the biggest wins in this repo are **artifact-neutral or nearly artifact-neutral** improvements: sliding-window eval, XSA, partial RoPE, LN scale, EMA, and GPTQ-lite clip search.
- Several recent 11-layer scripts already contain optional `mtp_heads` code paths and explicitly exclude them from export, but the documented runs keep MTP disabled. That makes MTP a rare case where there is already wiring for a potentially useful training-only objective that has not actually been explored in the recorded results.
- A negative-result non-record run found that naive depth recurrence was not worth the fixed-wallclock tradeoff on small hardware, so this candidate deliberately avoids large compute-for-capacity swaps and instead tests a lighter auxiliary objective.

In short: this candidate tries to add **training signal without paying submission bytes**, while staying on the strongest pre-TTT scaffold in the repo.

## Prior repository runs that influenced this candidate

### Main base implementation

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

This is the direct base because it is the strongest non-TTT result in the repo and already includes the mature 11-layer stack.

### Nearby runs that shaped the choice

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`

This clarified that the real gains in the late 11L line came from partial RoPE and LN scaling, not from a naive late-QAT toggle.

- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`

This README is useful because it already exposes `MTP_NUM_HEADS=0` in the run command, which helped confirm that MTP support existed in code but was not part of the recorded recipe.

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`

This run reports a negative result for layer recurrence, which pushed this candidate away from the cross-layer-sharing / recurrent-depth direction and toward a lighter training-objective change.

There were no prior `candidates/` directories in the repository when this candidate was created.

## External research that informed the idea

### 1. Better & Faster Large Language Models via Multi-token Prediction

- Fabian Gloeckle et al., 2024
- arXiv: <https://arxiv.org/abs/2404.19737>

This paper argues that predicting multiple future tokens with independent heads on a shared trunk improves sample efficiency and downstream capability. For this repo, the important part is not speculative decoding itself, but the claim that MTP can improve the underlying model's learned representations while keeping the architecture change local and simple.

### 2. Pre-Training Curriculum for Multi-Token Prediction in Language Models

- Ansar Aynetdinov and Alan Akbik, 2025
- arXiv: <https://arxiv.org/abs/2505.22757>

This is the key paper for the current challenge setting. It reports that **smaller language models struggle with naive MTP**, and that a **forward curriculum** helps small models get better next-token performance while retaining the useful MTP signal. That maps well onto Parameter Golf, where training time is short and models are much smaller than the LLMs often used in MTP papers.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate makes the following focused changes:

1. **Enable one MTP head by default**
   - `MTP_NUM_HEADS=1`
   - The auxiliary head remains excluded from the exported artifact.

2. **Add a forward curriculum for the MTP loss**
   - `MTP_LOSS_WEIGHT=0.15`
   - `MTP_WARMUP_STEPS=1500`
   - The auxiliary loss ramps from `0.0` to the target weight across the warmup window.

3. **Make the MTP weight runtime-mutable in a compile-safe way**
   - The active MTP weight is stored in a tensor buffer instead of relying on a Python-side constant in the forward path.
   - This avoids the same class of compile-time constant-folding problem that earlier repo history surfaced for a late-QAT toggle.

4. **Make the script runnable from the candidate directory**
   - Default `DATA_PATH` and `TOKENIZER_PATH` resolve relative to the repository root instead of assuming the current working directory is the repo root.

5. **Add a non-FlashAttention fallback**
   - The attention helper falls back to PyTorch scaled dot-product attention when `flash_attn_interface` is unavailable.
   - In practice this is mainly exercised by the local smoke-validation path; the main training/evaluation path still requires CUDA.

6. **Add a lightweight CPU smoke mode**
   - `SMOKE_TEST=1 python train_gpt.py`
   - This path instantiates the model, runs a forward/backward pass on random tokens, quantizes the exportable state, reloads it, and verifies logits can be produced.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603311823_curriculum-mtp
```

Main training / evaluation command:

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

A more explicit form with the candidate defaults spelled out:

```bash
SEED=1337 \
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.15 \
MTP_WARMUP_STEPS=1500 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Local smoke check:

```bash
SMOKE_TEST=1 python train_gpt.py
```

The main training path still requires CUDA; `SMOKE_TEST=1` is the CPU-friendly validation path.

If you want to override data locations explicitly from the candidate directory:

```bash
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- **Extra training compute:** even one extra auxiliary head is not free. If the throughput drop is large enough, the sample-efficiency gain may not offset the reduced step count inside the 600-second budget.
- **Curriculum tuning risk:** small models may want a different `MTP_LOSS_WEIGHT` or `MTP_WARMUP_STEPS` than the conservative defaults here.
- **Objective mismatch risk:** the benchmark still scores next-token compression, so a future-token auxiliary loss can help or hurt depending on how much it regularizes versus distracts.
- **Fallback attention is only for portability:** serious benchmark runs should still use the FlashAttention path when available.

## Validation

Commands run during candidate creation:

```bash
python -m compileall candidates/202603311823_curriculum-mtp/train_gpt.py
```

Outcome:
- Passed syntax compilation.

```bash
cd candidates/202603311823_curriculum-mtp && SMOKE_TEST=1 python train_gpt.py
```

Outcome:
- Passed in a temporary virtual environment containing the repo dependencies.
- Printed:

```text
smoke_test:ok attention_backend:torch_sdp loss:7.9611 logits_shape:(2, 64, 1024) mtp_loss_weight:0.1500
```
