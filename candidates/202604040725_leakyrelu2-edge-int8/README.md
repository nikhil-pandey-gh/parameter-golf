# LeakyReLU² + edge-aware attention int8

## Hypothesis

The cleanest near-SOTA non-TTT stack in this repository still missed the most recent MLP activation win, and its exporter still uses uniform low-bit treatment for almost every attention block. This candidate tests whether **adding LeakyReLU(0.5)^2 to the 2026-03-22 11-layer EMA + GPTQ-lite stack**, while **keeping the first and last attention blocks at int8 and leaving the rest of attention/MLP weights on GPTQ-lite int6**, improves post-quantization `val_bpb` without breaking the 16MB artifact budget.

## Why this is promising here

- The strongest non-TTT base in the repo is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which already combines 11 layers, 3x MLP, XSA, partial RoPE, LN scale, EMA, GPTQ-lite, and sliding eval.
- The latest top record, `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`, reports **LeakyReLU(0.5)^2** as the biggest single remaining pre-TTT gain on its stack.
- Earlier records repeatedly showed that quantization error, not raw train loss alone, is often the limiting factor. `records/track_10min_16mb/2026-03-19_WarmdownQuantization/` and the later int5/int6 runs both emphasize that preserving a few sensitive tensors can buy disproportionate post-export quality.

## Prior repository runs that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - chosen as the base implementation
   - contributes EMA, GPTQ-lite clip search, warmdown3500, XSA4, partial RoPE, LN scale, VE128, and the strong 11-layer 3x-MLP stack
2. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - contributes the LeakyReLU(0.5)^2 MLP change
   - motivates trying that activation on the simpler non-TTT stack before adopting its heavier parameter-banking + TTT machinery
3. `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
   - showed that selective higher precision for quantization-sensitive tensors can pay off
4. No prior `candidates/` directory existed when this candidate was created.

## External research that informed the choice

- **SliderQuant: Accurate Post-Training Quantization for LLMs** (arXiv:2603.25284, 2026) argues that the first and last layers are unusually quantization-sensitive, so uniform bit-width allocation is suboptimal.
- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (arXiv:2306.00978, 2023) argues that only a small subset of salient weights/channels need extra protection to reduce quantization error.
- **SpinQuant: LLM quantization with learned rotations** (arXiv:2405.16406, 2024) and **QuaRot** (arXiv:2404.00456, 2024) suggest that outlier-aware PTQ can matter a lot, but both require broader architectural/export refactors than fit this minimal candidate.
- I also considered recurrent-depth / layer-sharing ideas from **Thinking Deeper, Not Longer** (arXiv:2603.21676, 2026), **Intra-Layer Recurrence in Transformers for Language Modeling** (arXiv:2505.01855, 2025), and **ALBERT** (arXiv:1909.11942, 2019), but the record review suggested recurrence has been a weak direction under this repo's 10-minute training budget.

## What changed versus the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. The MLP activation changed from `relu^2` to **LeakyReLU(0.5)^2**.
2. The mixed exporter now keeps **attention weights in the first and last transformer blocks at int8**, while the rest of the block attention/MLP path stays on GPTQ-lite int6. Auxiliary projection weights outside `blocks.*` keep the base script's quantization behavior.
3. The default late-QAT threshold is set to `0.0` so this candidate does not rely on the previously fragile runtime QAT toggle path.

Everything else stays intentionally close to the base stack.

## How to run

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

The script resolves its default dataset and tokenizer paths relative to the repository root, so the command above works when launched from the candidate directory. Override paths only if your local checkout differs.

The script keeps the base defaults: 11 layers, `seq_len=2048`, EMA, GPTQ-lite mixed int6 export, sliding evaluation at stride 64, and zstd compression when available.

## Validation

Commands run for this candidate in the workflow environment:

```bash
python -m compileall candidates/202604040725_leakyrelu2-edge-int8/train_gpt.py
```

Outcome:

- `compileall` succeeded.
- A minimal CPU smoke run was **not feasible** in this workflow environment because the runtime dependencies needed by the training script (`torch`, `numpy`, `sentencepiece`, `flash_attn_interface`) were not installed, and the script also hard-requires CUDA.

## Expected risks and tradeoffs

- The extra int8 precision on edge attention blocks may help the roundtripped model, but the gain could be very small if GPTQ-lite already removed most of the relevant error.
- LeakyReLU(0.5)^2 transferred well in the newer record stack, but it has not yet been validated on this exact EMA + GPTQ-lite base.
- Keeping the candidate close to the base stack reduces implementation risk, but it also means this is an incremental step rather than a large architectural jump.
