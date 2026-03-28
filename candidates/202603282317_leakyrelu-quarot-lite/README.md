# LeakyReLU^2 + QuaRot-lite Attention Rotation

## Hypothesis

The strongest clean local base in this repo is the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack: 11 layers, 3x MLP, XSA on the deepest layers, Partial RoPE, LN scale, EMA, GPTQ-lite clip search, and mixed int6/int8 export.

This candidate keeps that stack intact, ports the later `LeakyReLU(0.5)^2` MLP activation from the current top record, and adds a new export-time idea:

- apply a **function-preserving orthogonal rotation** to paired attention/value weights before mixed int6 GPTQ-lite quantization;
- leave the training graph unchanged;
- quantize the rotated weights with the existing per-row clip search.

The goal is to reduce the **post-quantization gap**, which has repeatedly been the main bottleneck in this repository.

## Why this is promising for this repository

Repository review strongly suggests that:

- training quality alone is not enough; the 4-hour non-record baseline improved pre-quant quality much more than post-quant quality;
- the best recent runs are all compression-aware and export-sensitive;
- the cleanest high-end base is the 2026-03-22 stack, because it is already near-SOTA without depending on parameter banking or legal TTT;
- the 2026-03-23 top record showed that `LeakyReLU(0.5)^2` is a real, low-cost gain worth porting.

Rotation-based PTQ is therefore a good fit: it is an **artifact-side** improvement, costs no extra training time, and can be folded into the existing paired attention/value matrices without changing the model function before quantization noise is introduced.

## Which records influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the direct base because it is the strongest non-TTT, non-parameter-banking stack in the repo.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the `LeakyReLU(0.5)^2` MLP activation.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - relevant because the rotation implementation explicitly preserves the RoPE-applied prefix and only mixes the non-RoPE attention subspace.

## External research that informed it

This candidate is inspired by the rotation-based PTQ line for LLMs:

- **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs** — fixed Hadamard-style rotations can smooth outliers before quantization.
- **SpinQuant: LLM quantization with learned rotations** — learned orthogonal rotations improve PTQ further, but require extra calibration/optimization.
- **OptRot: Mitigating Weight Outliers via Data-Free Rotations for Post-Training Quantization** (`arXiv:2512.24124`) — argues that even data-free rotations can outperform plain Hadamard rotations for weight PTQ.
- **BASE-Q: Bias and Asymmetric Scaling Enhanced Rotational Quantization for Large Language Models** (`arXiv:2506.15689`) — treats rotations as a core ingredient of strong modern PTQ pipelines.

Given this repository's constraints, I implemented the simplest low-risk subset of that literature:

- no new training loop,
- no calibration data dependency,
- no extra runtime graph,
- only a fusible export-time rotation on already-paired attention/value weights.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **LeakyReLU^2 MLP**
   - Replaced `relu(x)^2` with `leaky_relu(x, negative_slope=0.5)^2`.
   - Exposed as `MLP_NEGATIVE_SLOPE` (default `0.5`).

2. **QuaRot-lite export path**
   - Added `QROT_ENABLED=1` by default.
   - Before mixed int6 quantization, rotate:
     - `c_q` and `c_k` outputs with a block Hadamard transform that preserves the RoPE prefix;
     - `c_v` outputs and `proj` inputs with a full head-wise Hadamard transform;
     - `ve_shared.proj` outputs to keep value embeddings aligned with the rotated value space.
   - The rotations are paired so the pre-quantization floating-point function is unchanged.

3. **Script-relative defaults**
   - Updated `DATA_PATH` and `TOKENIZER_PATH` defaults to resolve from the repository root, so this script can be launched from the candidate directory directly.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603282317_leakyrelu-quarot-lite
RUN_ID=leakyrelu_quarot_lite \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# disable the export rotation
QROT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# revert to relu^2
MLP_NEGATIVE_SLOPE=0.0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`DATA_PATH` and `TOKENIZER_PATH` default to the repository's standard `data/` layout and can still be overridden explicitly.

## Validation

Ran:

```bash
python -m compileall candidates/202603282317_leakyrelu-quarot-lite/train_gpt.py
```

Outcome:

- **Passed** syntax compilation.

Attempted an additional CPU-only smoke test, but it was **not feasible in this runner**:

- the environment does not have `torch` installed, so even an import-level CPU smoke cannot run here without adding heavyweight dependencies;
- the real `main()` path also hard-requires CUDA plus `flash_attn_interface`, so a faithful end-to-end start check is not possible on this host.

## Main expected risks and tradeoffs

- The rotation may improve quantization error but still hurt final artifact bytes if it makes the compressed int6 payload less zstd-friendly.
- Only the attention/value path is rotated; the MLP remains the dominant parameter block, so gains may be modest if most error comes from MLP quantization.
- `LeakyReLU^2` was ported from the top record, but the 2026-03-22 base was not fully re-tuned around it; warmdown and QAT thresholds may want follow-up tuning.
