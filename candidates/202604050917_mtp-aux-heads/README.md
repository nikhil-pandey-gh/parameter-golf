# 11L Banked MTP Auxiliary Head + Legal TTT

## Hypothesis

Add a **single training-only multi-token prediction (MTP) head** to the current best 11-layer banked stack so the model learns a slightly richer future-prediction objective during the same 600-second budget, while keeping the exported artifact unchanged because the auxiliary head is dropped before serialization.

The repository's best records repeatedly show that:

1. **step-time regressions are dangerous**, so the next idea should be low-overhead,
2. **artifact size is precious**, so improvements that disappear at export are attractive, and
3. **quantization/export quality must stay intact**, so the mature 03-23 training/eval/quantization stack should be preserved.

MTP fits that profile better than a broader architecture rewrite.

## Why this is promising for this repository

- The latest record (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`) already provides the strongest full stack: LeakyReLU(0.5)^2, parameter banking + Parallel Muon, partial RoPE, XSA on the deepest layers, VE, EMA/SWA, GPTQ-lite-style int6 export, and legal score-first TTT.
- Earlier record-family scripts already contained **MTP scaffolding**, but every recorded run left it disabled (`mtp_num_heads:0` in the logs), so the repository never actually evaluated MTP as part of the winning stack.
- MTP is especially attractive under the 16 MB cap because its auxiliary heads are **training-only**; this candidate keeps the existing export path that excludes `mtp_heads` from the serialized artifact.

## Prior repository influences

- **Root baseline:** `train_gpt.py` in the repository root established the Muon/Adam split, tokenizer-agnostic BPB evaluation, and compressed export conventions that later records kept.
- **03-21 / 03-22 records:** partial RoPE, layer-scale damping, EMA, and GPTQ-lite clip search are the most successful pre-TTT ideas and remain untouched here.
- **03-23 record:** this candidate directly forks `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` because it is the strongest documented stack in this checkout (`1.1194` mean post-TTT BPB).
- **Prior candidates:** there was no `candidates/` directory in this checkout when this candidate was created.

## External research that informed the choice

1. **Fabian Gloeckle et al., _Better & Faster Large Language Models via Multi-token Prediction_**  
   arXiv:2404.19737 — argues that predicting multiple future tokens from a shared trunk improves sample efficiency and helps induction-style behavior without requiring the extra heads at deployment time.  
   <https://arxiv.org/abs/2404.19737>
2. **DeepSeek-AI, _DeepSeek-V3 Technical Report_**  
   arXiv:2412.19437 — a recent frontier training report that explicitly calls out a multi-token prediction objective as part of its stronger training recipe.  
   <https://arxiv.org/abs/2412.19437>

Other research directions considered in the external survey were:

- **real LSQ/LSQ+ late QAT plus PACT-style clipping**,
- **AWQ/SmoothQuant-lite export rescaling**,
- **QuaRot/SpinQuant-style rotations**,
- **BitNet-style low-bit training**, and
- **factorized/adaptive embeddings**.

Those ideas are plausible, but on top of the current 03-23 banked implementation they would require a broader rewrite of either the banked matmul path or the export/calibration path. MTP was chosen for this candidate because it is the cleanest low-overhead, no-artifact-bytes intervention that still has recent primary-source support.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate makes four targeted changes:

1. **Enable a 1-step MTP auxiliary head by default** with `MTP_NUM_HEADS=1` and the existing conservative `MTP_LOSS_WEIGHT=0.2`.
2. **Fix MTP optimizer wiring.** In the 03-23 fork, `mtp_heads` existed in the model and loss but were not added to any optimizer param group. This candidate routes them through the replicated AdamW path alongside other small non-banked matrices.
3. **Bake the legal TTT configuration into defaults** so running the script from the candidate directory actually evaluates the intended candidate (`TTT_ENABLED=1`, `TTT_FREEZE_BLOCKS=0`, `ITERATIONS=9000`).
4. **Resolve dataset/tokenizer defaults from the repository root** instead of the current working directory, so `cd candidates/202604050917_mtp-aux-heads && torchrun ... train_gpt.py` works without rewriting paths.

Everything else deliberately stays aligned with the 03-23 record:

- 11 layers, 512 width, 8 heads / 4 KV heads
- LeakyReLU(0.5)^2 MLP
- Partial RoPE (16 dims), LN scale, XSA on the last 4 layers
- VE on late layers, smear gate, parameter banks, Parallel Muon
- EMA/SWA averaging
- int6 + lzma export and legal score-first TTT evaluation

## How to run or evaluate

Train + export + built-in evaluation from the candidate directory:

```bash
cd candidates/202604050917_mtp-aux-heads
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
cd candidates/202604050917_mtp-aux-heads

# Turn off MTP while keeping the rest of the stack
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Measure the pre-TTT path only
TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604050917_mtp-aux-heads/train_gpt.py`  
  **Outcome:** passed.
- `python -c "import importlib.util; print(importlib.util.find_spec('torch'))"`  
  **Outcome:** returned `None`, so a real module-import or CPU forward smoke test was not feasible in this container.
- Full end-to-end `main()` startup is also **not CPU-safe by default** here because the script hard-requires CUDA plus the FlashAttention dependency path used by the record-family code.

## Main expected risks or tradeoffs

- **Step-time risk:** even one extra auxiliary head increases training compute; if the step slowdown outweighs the sample-efficiency gain, BPB could regress despite the better objective.
- **Objective interference:** the MTP head is discarded at export, so any benefit must transfer into the shared trunk rather than depending on the auxiliary head itself.
- **TTT interaction risk:** legal TTT already adapts the post-quant model; MTP may improve the base model, have no effect, or reduce how much TTT can still recover.
- **Quantization neutrality is assumed, not guaranteed:** the head itself is not exported, but the trunk weights it shapes still need to quantize as cleanly as the 03-23 stack.
