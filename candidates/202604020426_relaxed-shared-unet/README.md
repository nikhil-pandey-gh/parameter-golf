# Relaxed Shared U-Net

## Hypothesis

The strongest non-TTT stack in this repo already looks close to saturated on quantization, evaluation, and token-feature tricks. The remaining open lever is **fixed-compute parameter sharing**: keep the same 11 logical passes, but reuse a smaller bank of block weights so the model spends fewer bytes on repeated matrices and more of the budget on the parts that seem to matter most.

This candidate tests a **mirror-shared 11-layer U-Net**:

- 11 logical layers
- 6 unique shared blocks
- mirror pattern `0,1,2,3,4,5,4,3,2,1,0`
- rank-8 per-depth attention LoRA adapters to relax the tying

The goal is to get the parameter-efficiency upside of recursive/shared transformers **without** repeating extra passes and paying the throughput penalty that hurt prior recurrence experiments.

## Why it is promising for this repository

Recent records converged on a strong common core:

- 11 layers, 512 width, 3x MLP
- SmearGate + BigramHash
- XSA on the deepest layers
- partial RoPE + LN scale
- VE128
- EMA + GPTQ-lite style int6 export

What is missing is a serious attempt at **modern layer tying** on that stack. The only explicit recurrence negative result in this repo came from a low-throughput non-record run that **increased effective depth** and therefore lost too many optimizer steps in a hard wall-clock budget. This candidate avoids that failure mode by keeping the total logical depth fixed at 11.

## Records and prior experiments that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
   - Chosen base implementation.
   - Keeps the best non-TTT 11L stack: XSA4, partial RoPE, LN scale, VE128, EMA, GPTQ-lite export.

2. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
   - Contributed the **LeakyReLU(0.5)^2** MLP activation.
   - Also motivated raising the default BigramHash table from 2048 toward the 3072 regime they found useful.

3. `records/track_10min_16mb/2026-03-19_SlidingWindowEval`
   - Its script contains dormant loop/LoRA code paths that show prior art for lightweight low-rank adaptation around reused transformer layers.

4. `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`
   - Important negative result: naive layer recurrence hurt badly when it reduced total steps.
   - This candidate responds by using **shared weights at fixed depth**, not extra recurrent passes.

## External research that informed it

1. **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** ([arXiv:2410.20672](https://arxiv.org/abs/2410.20672))
   - The closest direct inspiration.
   - Motivates combining layer tying with small depth-wise LoRA modules instead of strict sharing alone.

2. **Understanding Parameter Sharing in Transformers** ([arXiv:2306.09380](https://arxiv.org/abs/2306.09380))
   - Suggests parameter sharing helps low-budget models partly through convergence behavior, not just nominal depth.
   - That argues for a conservative sharing pattern plus separate optimizer treatment for the LoRA path.

3. **ALBERT** ([arXiv:1909.11942](https://arxiv.org/abs/1909.11942)) and **Universal Transformer** ([arXiv:1807.03819](https://arxiv.org/abs/1807.03819))
   - Older but still relevant evidence that cross-layer sharing can substantially improve parameter efficiency when depth is preserved.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. Replaces 11 unique transformer blocks with **6 shared blocks** and a mirror reuse schedule across the 11 logical layers.
2. Adds **per-logical-layer attention LoRA adapters** (`Q/K/V/out`) with a separate AdamW optimizer group (`DEPTH_LORA_RANK`, `DEPTH_LORA_LR`).
3. Switches the MLP nonlinearity to **LeakyReLU(0.5)^2**.
4. Increases the default **BigramHash** table from `2048` to `3072`.
5. Leaves **late QAT opt-in** by default (`LATE_QAT_THRESHOLD=0.0`) because a prior record showed the compiled class-flag path can silently no-op.
6. Fixes the default `DATA_PATH` and `TOKENIZER_PATH` so the script can be launched directly from this candidate directory.

Everything else stays intentionally close to the late-March 11L stack: partial RoPE, LN scale, XSA4, VE128, EMA, GPTQ-lite export, and stride-64 sliding evaluation.

## How to run / evaluate

From the repository root:

```bash
cd candidates/202604020426_relaxed-shared-unet
RUN_ID=relaxed_shared_unet \
NUM_SHARED_BLOCKS=6 \
SHARED_BLOCK_SCHEME=mirror \
DEPTH_LORA_RANK=8 \
DEPTH_LORA_LR=0.01 \
BIGRAM_VOCAB_SIZE=3072 \
LEAKY_RELU_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- If you keep the repository data cache in the standard location, the default paths should work when run from this directory.
- Otherwise override `DATA_PATH` and `TOKENIZER_PATH`.
- The script still emits the same main evaluation outputs as the base stack, including the int6 roundtrip metrics and sliding-window metrics.

## Validation run in this workflow

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604020426_relaxed-shared-unet/train_gpt.py` | Passed |
| Minimal CPU-only import/forward smoke test | Not feasible in this runner because the repo runtime dependencies (`torch`, `sentencepiece`) from `requirements.txt` were not installed |

## Main expected risks and tradeoffs

1. **Oversharing risk**: this shares full transformer blocks, so some per-layer specialization is now carried only by depth position, skip structure, VE placement, and the LoRA adapters.
2. **Throughput risk**: the LoRA path is small, but it still adds extra matmuls on every logical layer.
3. **Interaction risk**: XSA, VE, and mirror sharing may want a different placement schedule than the original 11 unique layers.
4. **Next experiment if this is close but not enough**: keep the same mirror sharing pattern and combine it with a compile-safe late-QAT implementation or a narrower subset-TTT evaluation pass.
