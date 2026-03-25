# Factorized Tail-Only MTP

## Hypothesis

The next underexplored lever in this repository is not a bigger exported model, but a cheaper **train-time-only multi-token prediction (MTP)** auxiliary loss that improves sample efficiency inside the 600-second budget while adding **zero exported artifact bytes**.

The main bet is that the existing 11-layer EMA + GPTQ-lite stack can benefit from seeing a small amount of farther-ahead supervision if the auxiliary head is made cheap enough:

- predict only a small set of farther-ahead targets,
- predict them only on the tail of each sequence,
- reuse the tied token embedding matrix for logits, and
- strip all MTP-only parameters before export.

## Why it is promising for this repository

Repository review suggests three strong trends:

1. The best recent gains came from **artifact-aware training/export improvements** rather than wholesale architecture swaps.
2. This codebase already has working MTP plumbing, but the recent winners kept it disabled, likely because the naive version used full extra vocab heads and added too much training cost.
3. The repo already embraces techniques that improve the trained model without increasing exported size, such as EMA/SWA and evaluation-side improvements.

That makes MTP a particularly natural next candidate here: it can change the training signal without making the artifact any larger.

## Prior records that influenced this candidate

- `train_gpt.py` in the repository root: baseline 9-layer GQA GPT with tied embeddings and int8 export.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - established the strong 11-layer XSA + EMA direction.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - reinforced partial RoPE + LN scale and also showed how fragile compile-time feature toggles can be.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen base implementation for this candidate because it already had the strongest clean 11-layer training/export stack and existing MTP hooks.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - confirmed that train/eval tricks that do not consume export bytes can still move the score materially.

The main dead ends I specifically avoided repeating are naive depth recurrence and heavyweight architectural swaps that would likely cost too many steps inside the 10-minute budget.

## External research that informed the idea

- Gloeckle et al., *Better & Faster Large Language Models via Multi-Token Prediction* (2024): <https://arxiv.org/abs/2404.19737>
  - motivates MTP as a training-efficiency tool rather than an inference-time architecture change.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023): <https://arxiv.org/abs/2305.13245>
  - relevant because this repository's strongest stacks already rely on GQA, so the candidate should preserve that attention layout.
- DeepSeek-AI et al., *DeepSeek-V2 Technical Report* (2024): <https://arxiv.org/abs/2405.04434>
  - I considered MLA-style latent KV compression as an alternative next step, but chose MTP first because it fits the existing code more directly and stays export-free.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Replace full extra MTP vocab heads with tiny factorized adapters**
   - The base script had optional extra `d_model -> vocab` heads.
   - This candidate replaces them with `MTPAdapter`, a small residual low-rank adapter that transforms hidden states and then reuses the tied embedding matrix for logits.
   - This keeps MTP train-time-only while drastically shrinking auxiliary parameters.

2. **Predict only farther-ahead targets**
   - New default: `MTP_TARGET_STEPS=2,4`
   - This makes the auxiliary task distinct from the main next-token objective.

3. **Restrict MTP to the tail of each sequence**
   - New default: `MTP_TAIL_TOKENS=256`
   - The auxiliary loss is only applied to the last 256 valid positions for each target offset, which cuts overhead substantially versus scoring the entire sequence.

4. **Keep the export path artifact-neutral**
   - All `mtp_heads` parameters are still excluded before serialization and quantization.

5. **Add a non-FlashAttention fallback**
   - If `flash_attn_interface` is unavailable, the script now falls back to causal `scaled_dot_product_attention`.
   - This does not target leaderboard speed; it exists to keep the module import/forward path debuggable and to let CUDA hosts without `flash_attn_interface` use PyTorch SDPA instead.
   - The actual `main()` training entrypoint remains CUDA-only, just like the recent record scripts.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603251149_factorized-tail-mtp
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This candidate resolves its default `DATA_PATH` and `TOKENIZER_PATH` from the repository root via `__file__`, so it can be launched directly from the candidate directory without extra path overrides.

Useful ablations:

```bash
# Disable MTP entirely
MTP_TARGET_STEPS= MTP_LOSS_WEIGHT=0.0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Narrower auxiliary head
MTP_ADAPTER_RANK=8 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Shorter auxiliary tail
MTP_TAIL_TOKENS=128 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Single farther-ahead target
MTP_TARGET_STEPS=2 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603251149_factorized-tail-mtp/train_gpt.py
```

Outcome:

- `compileall` passed for the root training scripts, `data/`, and the new candidate script.

Attempted smoke check:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path("candidates/202603251149_factorized-tail-mtp/train_gpt.py")
spec = importlib.util.spec_from_file_location("candidate_train_gpt", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
...
PY
```

Outcome:

- A true CPU forward-pass smoke test was **not feasible in this workflow container** because both `/usr/bin/python` and `/usr/bin/python3` raise `ModuleNotFoundError: No module named 'torch'`.
- The candidate still includes the SDPA fallback so the same import-level smoke path should work in an environment where repository dependencies from `requirements.txt` are installed, even if `flash_attn_interface` is absent.

## Main expected risks and tradeoffs

- Even a cheaper MTP path may still reduce training throughput enough to cancel its sample-efficiency gains.
- The best `MTP_TARGET_STEPS`, `MTP_TAIL_TOKENS`, `MTP_ADAPTER_RANK`, and `MTP_LOSS_WEIGHT` are uncertain.
- I intentionally kept this candidate **compile-safe and simple** rather than adding runtime on/off schedules, because this repository already hit a `torch.compile` constant-folding bug around late-QAT toggles.
- Because the MTP adapters are train-time-only, this candidate's success depends entirely on whether the auxiliary loss improves the base model rather than simply fitting the extra objective.
