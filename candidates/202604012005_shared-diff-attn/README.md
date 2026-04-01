# Shared Differential Attention on the Banked 11L Base

## Hypothesis

The strongest current stack already wins by removing noisy or redundant signal in deep attention layers: XSA removes self-value leakage, partial RoPE reduces over-positionalization, and LeakyReLU squared improves gradient flow. A **shared differential attention** variant should extend that trend by letting the deepest layers compare a base attention map against a second, low-rank-adjusted map and subtract the noisy component before the output projection.

The candidate keeps the current banked 11-layer base intact and only adds this mechanism to the last 2 layers, where the records already show attention-side interventions pay off most.

## Why it is promising here

- The repository's best results come from **small, targeted deep-layer changes** rather than wholesale rewrites: XSA, partial RoPE, LN scaling, EMA/SWA, and LeakyReLU squared all stacked cleanly.
- The artifact budget is already tight, so a **shared-base + low-rank** attention tweak is more realistic than adding full second projection banks or a new recurrent backbone.
- The current banked base buys back some step time, which helps offset the extra FlashAttention call in only a couple of layers.

## Prior repository runs that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`  
  Current best overall stack. This candidate reuses its banked core, LeakyReLU squared MLP, XSA, partial RoPE, VE, and optimizer path, but leaves `TTT_ENABLED=0` by default to isolate the architectural change.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`  
  Best recent training-only family and the clearest proof that the 11L/XSA/partial-RoPE regime is the right starting point.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`  
  Motivates keeping deep-layer attention edits and partial RoPE focused on the last layers.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`  
  Shows that selective attention surgery in the deepest layers can win without adding many parameters.

There were no prior `candidates/` directories in the repo at the time this candidate was created.

## External research that informed it

- **Diff Transformer** ([arXiv:2410.05258](https://arxiv.org/abs/2410.05258)) argues that subtracting one attention map from another reduces irrelevant-context noise and improves language modeling plus long-context robustness.
- **DINT Transformer** ([arXiv:2501.17486](https://arxiv.org/abs/2501.17486)) extends the same line of thought, emphasizing robustness to noisy context and better global-token selection.
- **Shared DIFF Transformer** ([arXiv:2501.17900](https://arxiv.org/abs/2501.17900)) is the closest direct inspiration: keep a shared base attention path and add lightweight low-rank flexibility instead of fully duplicating projections.
- I also considered **Relaxed Recursive Transformers** ([arXiv:2410.20672](https://arxiv.org/abs/2410.20672)) and **Mixture-of-Recursions** ([arXiv:2507.10524](https://arxiv.org/abs/2507.10524)), but the repo's own recurrence results were negative under the 10-minute constraint, so I did not pursue that path here.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes:

1. **Added shared differential attention controls**
   - `DIFF_ATTN_LAST_N` (default `2`)
   - `DIFF_RANK` (default `16`)
   - `DIFF_LAMBDA_INIT` (default `0.0`)

2. **Added low-rank differential Q/K updates in selected deep layers**
   - The main banked Q/K weights stay unchanged.
   - Selected layers instantiate small `diff_q_*` and `diff_k_*` adapters.
   - Attention becomes:
     - base path: `A(q, k, v)`
     - differential path: `A(q + dq, k + dk, v)`
     - output: `A(q, k, v) - lambda * A(q + dq, k + dk, v)`

3. **Kept the diff parameters in the control/passthrough path**
   - They are tiny relative to the banked core.
   - Keeping them out of aggressive export quantization is the safer first test.

4. **Made default dataset/tokenizer paths repo-relative**
   - The script now resolves `data/` relative to the repository root inferred from `__file__`.
   - This lets it be run directly from the candidate directory without manually overriding paths.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202604012005_shared-diff-attn
RUN_ID=shared_diff_attn \
BIGRAM_VOCAB_SIZE=1536 \
XSA_LAST_N=4 \
DIFF_ATTN_LAST_N=2 \
DIFF_RANK=16 \
DIFF_LAMBDA_INIT=0.0 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 SWA_ENABLED=1 SWA_EVERY=50 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If the training-only result is promising, the obvious follow-up is to re-run with `TTT_ENABLED=1` and compare whether differential attention still helps after score-first adaptation.

## Main expected risks / tradeoffs

- **Step-time risk:** each diff-attn layer adds a second FlashAttention call, so too many layers could erase any quality gain by reducing steps.
- **Oversubtraction risk:** XSA and differential attention both suppress attention-side signal. Using only the last 2 layers is intended to keep this from becoming too aggressive.
- **Artifact risk:** the low-rank diff weights are small, but they are currently kept in the passthrough path for stability, so they still consume some extra bytes.
- **Unverified runtime risk:** this environment did not have the full CUDA/PyTorch runtime needed for an actual local launch.

## Validation

- `python -m compileall candidates/202604012005_shared-diff-attn/train_gpt.py`  
  **Passed**
- Minimal runtime smoke test  
  **Not feasible here**: the local Python environment does not have `torch` installed, and this script also requires CUDA plus `flash_attn_interface` to start a real run.
