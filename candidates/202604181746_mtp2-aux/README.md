# LeakyReLU² + Parallel Muon + 2-Head MTP Aux

## Hypothesis

The strongest next step is to turn on **multi-token prediction (MTP) as an auxiliary training loss** on top of the current fastest pre-TTT stack, without paying any artifact-size cost at export time.

This repository's record history already converged on an 11-layer LeakyReLU² / XSA / partial-RoPE / EMA / GPTQ-lite family, but the later scripts still carry a dormant MTP path with `MTP_NUM_HEADS=0` in submitted configs. The hypothesis here is that **predicting a small number of future tokens during training improves sample efficiency enough to beat the small extra training cost inside the fixed 600-second budget**.

## Why this is promising here

1. The repo review showed that the biggest recent wins came from **small, composable training/eval improvements** on the same core 11-layer architecture rather than from radically new model families.
2. The 2026-03-23 record showed the current pre-TTT stack is already strong and fast: **LeakyReLU(0.5)^2** helped materially, while **Parallel Muon** mainly bought throughput headroom.
3. Prior records and candidates did **not** report an MTP-based submission, even though the later record scripts already contain the machinery.
4. The MTP heads are **training-only** in this codepath and are excluded from export, so the candidate can chase training-signal gains without inflating the final model bytes.

## Prior repository work that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current pre-TTT stack,
  - LeakyReLU²,
  - parameter banking + Parallel Muon,
  - 11L / XSA4 / partial RoPE / VE128 / EMA + tight SWA.
- **Direct architectural parent:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - 11L XSA4 + EMA + GPTQ-lite + warmdown3500,
  - same family as the current best non-TTT training stack.
- **Negative guidance:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - layer recurrence hurt badly under the fixed wallclock,
  - so this candidate stays inside the proven 11-layer family instead of revisiting recurrence.
- **Evidence that MTP is still open here:** `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md`
  - submitted config explicitly keeps `MTP_NUM_HEADS=0`.

## External research behind the idea

- **Better & Faster Large Language Models via Multi-token Prediction** (arXiv:2404.19737) argues that predicting multiple future tokens improves sample efficiency and generative capability.
- **Speculative Streaming** (arXiv:2402.11131) shows future n-gram prediction can be made parameter-efficient and useful in resource-constrained settings.
- **MuToR** (arXiv:2505.10518) reinforces the idea that lightweight MTP variants can work with negligible added parameter cost.
- **On multi-token prediction for efficient LLM inference** (arXiv:2502.09419) notes that hidden states are heavily specialized for next-token prediction, which is why this candidate keeps the standard horizon-specific auxiliary heads instead of collapsing everything to one shared head.

## What changed vs the chosen base

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

1. **Turns on MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
2. **Matches the pre-TTT bigram setting directly in defaults**
   - `BIGRAM_VOCAB_SIZE=1536`
3. **Keeps TTT off by default**
   - the goal here is to isolate whether MTP improves the training-time backbone before any legal TTT layer is added back in.
4. **Adds a FlashAttention fallback**
   - if `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA.
5. **Adds `CPU_SMOKE=1`**
   - this runs a tiny synthetic forward/backward plus export-filter/quantization sanity check on CPU without dataset or CUDA dependencies.

## How to run

From the repository root:

```bash
cd candidates/202604181746_mtp2-aux
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate defaults already encode the intended stack, including `MTP_NUM_HEADS=2`, `MTP_LOSS_WEIGHT=0.15`, and `BIGRAM_VOCAB_SIZE=1536`.

## Minimal CPU smoke command

```bash
cd candidates/202604181746_mtp2-aux
CPU_SMOKE=1 \
VOCAB_SIZE=128 \
NUM_LAYERS=2 \
MODEL_DIM=64 \
NUM_HEADS=4 \
NUM_KV_HEADS=2 \
MLP_MULT=2 \
TRAIN_SEQ_LEN=32 \
EVAL_SEQ_LEN=32 \
BIGRAM_VOCAB_SIZE=128 \
VE_ENABLED=0 \
XSA_LAST_N=1 \
python train_gpt.py
```

## Main risks / tradeoffs

1. **Auxiliary-loss overhead:** even export-free MTP still adds training FLOPs and may reduce step count enough to wash out the gain.
2. **Transfer risk:** the MTP heads are not exported, so the win must be absorbed into the trunk weights during training.
3. **Tiny-model uncertainty:** MTP papers are encouraging, but gains are often larger at bigger model scales than this challenge allows.
4. **Sweep sensitivity:** the best setting may be `MTP_NUM_HEADS=1` or a different `MTP_LOSS_WEIGHT`; this candidate chooses a conservative middle ground instead of pretending the optimum is known.

## Validation

Planned lightweight validation for this candidate:

1. `python -m compileall candidates/202604181746_mtp2-aux/train_gpt.py`
2. the CPU smoke command above

The exact commands and outcomes are recorded below after local validation runs.

### Validation outcomes

- `python -m compileall candidates/202604181746_mtp2-aux/train_gpt.py` — **passed**
- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604181746_mtp2-aux/train_gpt.py` — **passed**
- CPU smoke command (run in a temporary venv with the repo's minimal runtime deps installed) — **passed**
  - output: `cpu_smoke:ok loss=5.5791 seq_len=32 excluded_mtp_params=16384`
- Standard non-smoke launch in this container still requires CUDA, so a full end-to-end training start was **not** feasible here beyond the synthetic CPU smoke path.
