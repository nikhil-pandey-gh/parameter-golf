# Candidate: Bank-Aware LSQ-Style Late QAT

## Hypothesis

The current best stack already exports its dominant attention and MLP banks with GPTQ-lite int6, but its late-QAT path only reaches small `CastedLinear` modules. A late-phase, bank-aware fake-quant path with tiny learned clip multipliers should reduce the train/export mismatch on the tensors that matter most for both artifact size and post-quant BPB.

## Why it is promising for this repository

- The strongest recent records kept winning by shaving down the quantization/export gap rather than by changing the whole model family.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly notes that a previous late-QAT path was compiled away and had no effect.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` banks the large q/k/v/out and MLP matrices, so its inherited late-QAT hook never touched the weights that dominate the final int6 artifact.
- This candidate stays inside the proven 11-layer, XSA, partial-RoPE, LeakyReLU^2, EMA/SWA, GPTQ-lite, legal-TTT stack instead of betting on a slower recurrent or shared-layer architecture.

## Prior records or candidates that influenced it

- **Primary base:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best overall stack in the repo
  - already includes LeakyReLU(0.5)^2, parameter banking, Parallel Muon, GPTQ-lite export, and legal score-first TTT
- **Quantization/export influence:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - showed that better clip selection and EMA could still squeeze out meaningful gains late in the repo’s progression
- **Failure mode influence:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - documented the `torch.compile` late-QAT dead-branch issue that this candidate explicitly avoids
- **Prior candidates:** none were present in the repository when this candidate was created

## External research that informed it

- **LSQ — Learned Step Size Quantization** (Esser et al., 2019): <https://arxiv.org/abs/1902.08153>
  - motivates learning quantizer scale/clipping parameters jointly with model weights
- **PACT — Parameterized Clipping Activation** (Choi et al., 2018): <https://arxiv.org/abs/1805.06085>
  - motivates learnable clipping rather than fixed hand-tuned thresholds in low-bit training

This candidate does not implement full LSQ/PACT machinery. Instead, it takes the minimal version that fits this codebase: one learned clip scalar per exported bank matrix, activated only in the late-QAT phase.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`:

1. Added learned clip logits for each banked attention/MLP matrix (`qo`, `kv`, `mlp_up`, `mlp_down`).
2. Added bank-aware fake int6 quantization in the attention and MLP forward paths, but only after late QAT activates.
3. Reused the learned clip logits during export by feeding them into GPTQ-lite as an extra clip candidate.
4. Switched late-QAT activation to drop from `torch.compile(...)` back to eager mode when the late-QAT threshold is crossed, so the QAT branch cannot be constant-folded away.
5. Kept the rest of the winning stack intact: LeakyReLU(0.5)^2, parameter banking, Parallel Muon, EMA + tight SWA, partial RoPE, XSA4, VE128, GPTQ-lite export, and legal score-first TTT.

## How to run or evaluate it

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 BANK_QAT_ENABLED=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

EMA stays always-on in this inherited base; the command above only needs to configure the optional pieces that are actually exposed as environment variables.

## Validation

- From repo root: `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604012350_bank-aware-lsq-qat/train_gpt.py` -> passed
- `python - <<'PY' ...  # import the candidate with a CPU FlashAttention stub and run a toy forward/export roundtrip` -> failed in this runner with `ModuleNotFoundError: No module named 'torch'`
- A true CPU-only start check is still not feasible without changing the candidate’s intended stack, because the inherited record code expects CUDA + FlashAttention-3 at runtime.

## Main expected risks or tradeoffs

- Switching to eager mode when late QAT begins may cost some late-phase throughput and slightly reduce final step count.
- Learned clip logits are deliberately tiny and simple; they may still be too weak to matter or may over-clip some layers.
- Training uses row-max-scaled fake quantization while export still includes percentile search, so alignment is improved but not mathematically exact.
- Legal TTT remains unchanged, so any gain from this candidate depends on the base stack still leaving room after post-quant export.
