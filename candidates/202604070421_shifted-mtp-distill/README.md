# Shifted MTP Distill on the 1.1194 Legal-TTT Stack

## Hypothesis

A **single export-free multi-token prediction (MTP) head** should improve sample efficiency on the current best stack without increasing final artifact size, and a **shifted self-distillation target** from the future main-head logits should keep that auxiliary task aligned with the inference-time objective instead of pulling the trunk away from next-token quality.

## Why this is promising here

- The strongest lineage in this repo already converged on the same 11-layer recipe: XSA on late layers, partial RoPE, VE128, EMA/SWA, aggressive low-bit export, and legal score-first TTT.
- Multiple recent record scripts still carry MTP support in code, but the reviewed runs from the March 20-23 record line all log `mtp_num_heads:0`, so this looks like a real untested gap rather than a repeated idea.
- MTP is especially attractive for Parameter Golf because the auxiliary heads can be **excluded from export**, so the candidate spends extra parameters only during training.

## Prior repo work that influenced this candidate

- **Base implementation**: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Best current result in this repo review: **1.1194 mean val_bpb**
  - Supplies the strongest current stack: LeakyReLU(0.5)^2, parameter banking + parallel Muon, partial RoPE, XSA4, VE128, EMA + tight SWA, GPTQ-lite-style int6+lzma export, and legal score-first TTT.
- **Supporting lineage**
  - `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: quantization-aware clip search / EMA trend
  - `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: partial RoPE and LN-scale gains
  - `2026-03-20_11L_EfficientPartialXSA_FA3_SWA120`: efficient late-layer XSA
- **Dead ends avoided**
  - The repo’s non-record single-GPU exploration explicitly reports **layer recurrence as a regression**, so this candidate does not chase shared-depth recurrence.

At implementation time there were **no prior experiments under `candidates/`**.

## External research that informed it

1. **Better & Faster Large Language Models via Multi-token Prediction** (Gloeckle et al., arXiv:2404.19737)  
   Motivates MTP as an auxiliary objective that can improve sample efficiency while keeping inference overhead optional.
2. **Self-Distillation for Multi-Token Prediction** (Zhao et al., arXiv:2603.23911)  
   Highlights that MTP heads are easier to use when they are regularized to preserve main-head quality.
3. **Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing** (Goel et al., arXiv:2603.17942)  
   Reinforces the idea that pretrained decoder stacks already contain latent future-token structure worth exposing.
4. **Thinking into the Future: Latent Lookahead Training for Transformers** (Noci et al., arXiv:2603.20219)  
   Supports the broader thesis that modest future-token supervision can improve planning/lookahead behavior.

## What changed vs. the chosen base implementation

1. **Enabled one MTP head by default**
   - `MTP_NUM_HEADS=1`
   - The extra head predicts the token two steps ahead and is excluded from export.
2. **Added shifted self-distillation**
   - The MTP head is trained on the hard future-token target **plus** an MSE match against the main head’s logits at the corresponding future position.
   - This is a cheap approximation to MTP self-distillation that uses tensors already produced in the forward pass.
3. **Fixed MTP optimizer wiring in the parameter-banked stack**
   - The copied March 23 script logged MTP settings but did not actually attach `mtp_heads` to any optimizer in this branch.
   - This candidate adds an explicit AdamW optimizer for MTP heads plus replicated-gradient all-reduce coverage.
4. **Made the script runnable from the candidate directory**
   - Default dataset/tokenizer paths are now resolved from the candidate file location instead of assuming the repo root as the working directory.
5. **Added a CPU smoke path**
   - `CPU_SMOKE_TEST=1` runs a no-data forward/backward sanity check on CPU.
   - Attention falls back to PyTorch SDPA when FlashAttention is unavailable; competitive GPU runs still use FlashAttention when present.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202604070421_shifted-mtp-distill

SEED=1337 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 MTP_DISTILL_WEIGHT=0.25 MTP_LR=0.025 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The remaining defaults intentionally stay close to the March 23 record stack. The script now resolves the standard repo data/tokenizer paths automatically, so `DATA_PATH` and `TOKENIZER_PATH` are optional when the repo layout is unchanged.

For a cheap local sanity check without CUDA or data:

```bash
cd candidates/202604070421_shifted-mtp-distill
CPU_SMOKE_TEST=1 python train_gpt.py
```

## Validation

Commands run during implementation:

1. `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604070421_shifted-mtp-distill/train_gpt.py`
   - **Passed**
2. `cd candidates/202604070421_shifted-mtp-distill && CPU_SMOKE_TEST=1 python train_gpt.py`
   - **Passed**
   - Output: `cpu_smoke_ok seq_len:64 loss:7.9668 flash_attn_fallback:True`

For the workflow run that produced the smoke-test output above, the command was executed inside a temporary virtualenv populated from the repo's existing `requirements.txt`, because the runner's system Python was externally managed.

## Main expected risks / tradeoffs

- **Training throughput risk**: even a single auxiliary head adds extra projection work and may cost some steps within the 600s budget.
- **Objective-mismatch risk**: future-token auxiliary losses can help sample efficiency, but they can also hurt the main next-token objective if weighted too aggressively.
- **TTT interaction risk**: this candidate changes the pretrained model, not the legal TTT recipe, so gains could either amplify or wash out after score-first adaptation.
- **Distillation simplification risk**: the self-distillation term here is a lightweight logit-matching approximation rather than a full MTP-D reproduction.
