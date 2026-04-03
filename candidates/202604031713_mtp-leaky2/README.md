# MTP auxiliary heads on the GPTQ-lite 11L stack

## Hypothesis

Add a small **multi-token prediction (MTP)** auxiliary objective to the strongest clean pre-TTT stack so the shared trunk learns faster within the same 10-minute budget, while keeping the exported artifact essentially unchanged because the extra MTP heads are already excluded from export.

I also carry over the proven **LeakyReLU(0.5)^2** MLP activation from the current best overall record, because it is a low-risk trunk-quality improvement that should stack naturally with MTP.

## Why this is promising for this repository

The record history suggests that the repo has already harvested most of the easy gains from architecture shape, eval stride, and quantization plumbing:

- sliding-window eval is already standard,
- the 11-layer XSA/Partial-RoPE/LN-scale stack is mature,
- GPTQ-lite + EMA + warmdown3500 is the strongest clean export path before expensive TTT,
- legal TTT still helps, but only by a small margin for a large eval-time cost.

That makes a **training-only objective improvement** attractive. MTP is a good fit here because:

- it improves supervision density without changing inference-time trunk structure,
- the extra heads are dropped from export, so artifact bytes stay focused on the main model,
- the codebase already has dormant MTP support, which keeps this candidate precise and low-infrastructure.

## Prior records that influenced this candidate

- **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best clean pre-TTT recipe: 11L, XSA4, Partial RoPE, LN scale, EMA, warmdown3500, GPTQ-lite.
- **Activation carried forward from:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - LeakyReLU(0.5)^2 was the clearest training-side gain in the current top record.
- **Earlier enabling trends:** `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` and `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - confirm that the strongest stack is already centered on 11L + XSA + Partial RoPE + LN scaling.

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

- **Gloeckle et al., 2024 — "Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`)
  - independent future-token heads on a shared trunk improved sample efficiency and downstream quality.
- **DeepSeek-V3 Technical Report** (`arXiv:2412.19437`)
  - explicitly uses a multi-token prediction training objective for stronger performance.
- **Gerontopoulos et al., 2025 — "Multi-Token Prediction Needs Registers"** (`arXiv:2505.10518`)
  - reinforces that MTP can add negligible parameter overhead while remaining close to the next-token objective.
- **Mehra et al., 2025 — "On multi-token prediction for efficient LLM inference"** (`arXiv:2502.09419`)
  - argues that MTP heads work best when trained jointly with the backbone instead of bolting them onto a frozen next-token model.

## What changed versus the chosen base implementation

This directory forks `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py` and makes only three intentional changes:

1. **Enable MTP by default** with `MTP_NUM_HEADS=2`.
2. **Keep the existing export exclusion for `mtp_heads`**, so the auxiliary heads do not count toward the final serialized model artifact.
3. **Replace ReLU^2 with LeakyReLU(0.5)^2** in the MLP, following the later 2026-03-23 record.

Everything else stays aligned with the strong 11-layer GPTQ-lite stack.

## How to run or evaluate

From this directory:

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate resolves its default `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so the command above works when launched from `candidates/202604031713_mtp-leaky2/`.

Useful ablations:

```bash
# Turn off the new objective.
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Reduce auxiliary pressure.
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still exports and evaluates the main model exactly as before; MTP heads are training-only.

## Main expected risks and tradeoffs

- **Step-time tax:** extra vocab heads add training compute, so MTP only wins if the sample-efficiency gain beats the lost steps.
- **Small-model horizon mismatch:** a tiny 11-layer model may benefit from predicting one future token but not necessarily two.
- **Interaction risk:** LeakyReLU^2 and MTP are each plausible on this stack, but they have not been validated together here yet.
- **No CPU smoke run:** the script is CUDA-only and depends on the existing GPU attention path, so lightweight validation is limited to static checks in this environment.

## Validation

- `python -m compileall candidates/202604031713_mtp-leaky2/train_gpt.py`
- `python -m compileall train_gpt.py train_gpt_mlx.py data`
- `cd candidates/202604031713_mtp-leaky2 && timeout 20s python train_gpt.py`

Outcomes in this workflow:

- Both `compileall` commands succeeded.
- The bounded startup probe failed immediately in this container with `ModuleNotFoundError: No module named 'numpy'`, so no true runtime smoke test was possible here without changing the environment.
- Separately, this script is also not a real CPU-smoke candidate because the runtime path imports `flash_attn_interface` and later hard-requires CUDA.
