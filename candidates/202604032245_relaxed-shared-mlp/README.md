# Relaxed Shared MLP

## Hypothesis

The largest trainable block in the strong 11-layer record family is the MLP stack, not attention. Sharing a single early-stack MLP across multiple layers, then relaxing that tie with tiny per-layer low-rank adapters, should reclaim artifact bytes without paying the full optimization cost of naive whole-block recurrence. In this candidate, that recovered budget is spent on a larger BigramHash table while keeping the rest of the proven 11-layer GPTQ-lite / EMA / XSA / partial-RoPE stack intact.

## Why it is promising for this repository

The repo history says two things clearly:

1. The best static runs come from the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` family: 11 layers, 3x MLP, BigramHash, SmearGate, XSA, partial RoPE, LN scaling, EMA/SWA, and strong post-training quantization.
2. Naive layer recurrence was already a negative result in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`: reusing whole layers under a fixed wall-clock budget cut step count too much and hurt badly.

This candidate tries a narrower version of parameter sharing that targets the heaviest parameter block only:

- share **MLPs**, not full blocks;
- share only the **early encoder-side layers** by default (`0,1,2`);
- keep **attention, norms, residual scales, skips, XSA, and VE** unique;
- add tiny **depth-specific low-rank adapters** so the tied layers are not forced to behave identically.

That makes the idea much closer to "artifact-efficient parameter allocation" than to the repo's previously negative full recurrence experiment.

## Prior records and experiments that influenced it

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest simpler static stack in the repo;
  - already has the right quantization, eval, and architectural scaffolding.
- **Activation carry-over:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the cheap `LeakyReLU(0.5)^2` activation win.
- **Capacity/compression trend:** `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - reinforced that 3x MLP and BigramHash are worth protecting.
- **Negative control:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - showed that naive repeated layers are the wrong form of recurrence for this challenge.

## External research that informed it

- **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** (Bae et al., arXiv:2410.20672, ICLR 2025)
  - argues that layer tying becomes much stronger when relaxed with small depth-wise low-rank adapters.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (Nguyen and Lin, arXiv:2505.01855, 2025)
  - suggests selective recurrence is better than indiscriminate whole-block reuse, with earlier layers especially promising.
- **Subformer: Exploring Weight Sharing for Parameter Efficiency in Generative Transformers** (Reid et al., arXiv:2101.00234, Findings of EMNLP 2021)
  - provides a generative-language-model precedent for sandwich-style parameter sharing beating naive cross-layer tying.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. **Shares one MLP across layers `0,1,2` by default** via `SHARED_MLP_LAYERS=0,1,2`.
2. **Adds per-layer low-rank adapters** (`SHARED_MLP_ADAPTER_RANK=16`) on the shared-MLP path.
3. **Switches the MLP activation to `LeakyReLU(0.5)^2`** via `MLP_NEGATIVE_SLOPE=0.5`.
4. **Raises BigramHash buckets from 2048 to 3072** to spend some of the recovered parameter budget.
5. **Adds a FlashAttention fallback** to `scaled_dot_product_attention`, so the module can at least import and run a CPU smoke forward when PyTorch is available but FlashAttention is not.
6. **Leaves late QAT disabled by default** (`LATE_QAT_THRESHOLD=0.0`) so this candidate isolates the sharing hypothesis instead of conflating it with the known late-QAT ambiguity in this record family.

## How to run or evaluate it

From this directory:

```bash
SEED=1337 \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=3072 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
SHARED_MLP_LAYERS=0,1,2 \
SHARED_MLP_ADAPTER_RANK=16 \
MLP_NEGATIVE_SLOPE=0.5 \
WARMDOWN_ITERS=3500 \
LATE_QAT_THRESHOLD=0.0 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To ablate the new idea cleanly:

- disable sharing entirely with `SHARED_MLP_LAYERS=`;
- keep sharing but remove the relaxation with `SHARED_MLP_ADAPTER_RANK=0`;
- revert to ReLU² with `MLP_NEGATIVE_SLOPE=0.0`;
- return to the base hash table with `BIGRAM_VOCAB_SIZE=2048`.

## Validation

- `python -m compileall train_gpt.py train_gpt_mlx.py candidates/202604032245_relaxed-shared-mlp/train_gpt.py`
  - passed in this workflow environment.
- Minimal CPU-only import/forward smoke in a temporary venv:
  - `python3 -m venv /tmp/gh-aw/agent/paramgolf-venv`
  - `/tmp/gh-aw/agent/paramgolf-venv/bin/pip install --quiet --upgrade pip`
  - `/tmp/gh-aw/agent/paramgolf-venv/bin/pip install --quiet torch sentencepiece numpy`
  - `/tmp/gh-aw/agent/paramgolf-venv/bin/python - <<'PY' ... PY`
  - outcome: the candidate imported successfully, instantiated a 4-layer CPU model with shared MLP layers `0,1`, and ran both `model(x, y)` and `model.forward_logits(x)` successfully.
  - observed smoke output: `{'loss': 4.842615604400635, 'logits_shape': (2, 16, 128), 'params': 110801}`
- Quantization/dequantization roundtrip smoke in the same temporary venv:
  - `python -m compileall candidates/202604032245_relaxed-shared-mlp/train_gpt.py`
  - `/tmp/gh-aw/agent/paramgolf-venv/bin/python - <<'PY' ... PY`
  - outcome: `mixed_quantize_int6(...)` and `dequantize_mixed_int6(...)` produced a state dict that loaded back into a reconstructed shared-MLP `GPT` instance with `strict=True`.
  - observed smoke output: `{'roundtrip_keys': 48, 'shared_mlp_layers': [0, 1]}`

## Main expected risks and tradeoffs

- If the shared early MLP becomes a bottleneck, the recovered bytes may not compensate for the lost layer-specific capacity.
- The adapter rank might be too small to relax the tie enough, or too large to keep the compression trade favorable.
- Sharing the early stack is research-motivated, but this repo's current best runs concentrate other tricks in deeper layers, so the best sharing location may still need retuning.
- Increasing BigramHash to 3072 may help use the saved budget, but it could also waste bytes if the shared-MLP change is not strong enough.
