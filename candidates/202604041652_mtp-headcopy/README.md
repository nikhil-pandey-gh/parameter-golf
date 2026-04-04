# MTP Head-Copy Auxiliary on the 11L GPTQ-lite Base

## Hypothesis

The best non-TTT stack in this repo already has dormant multi-token prediction (MTP) support, but every published run leaves it off (`mtp_num_heads:0`). Enabling a small training-only MTP auxiliary should improve sample efficiency in the 10-minute budget, and initializing those future-token heads from the main tied embedding projection should make that auxiliary useful immediately instead of spending early steps climbing out of zero-init.

## Why it is promising here

- The repository's strongest non-TTT base is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, which already stacks the most reliable recent ideas: Partial RoPE, LayerNorm scaling, XSA on the deepest layers, EMA, and GPTQ-lite quantization.
- The full record review shows the recent gains are getting small and mostly come from eval-time TTT or tiny quantization refinements, while the MTP path is present but untested in the published 11-layer scripts.
- MTP heads are excluded from export in the base code, so this is a rare knob that can improve training while adding **zero bytes** to the final artifact.

## Prior repo work that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- **Architecture trendline:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` and `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
- **Why not start from TTT:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` is better overall, but its README shows the post-training TTT gain is only about `-0.0025 BPB` on top of a much heavier eval stack. This candidate instead isolates a lighter training-time change.
- **Untried opportunity:** the 2026-03-20 through 2026-03-23 11-layer scripts all log `mtp_num_heads:0`, so the mechanism exists but has not been exercised in the record history.

## External research that informed it

- **Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737).** Motivates MTP as an auxiliary objective that improves sample efficiency and downstream quality while keeping inference/export overhead optional.
- **Zhao et al., "Self-Distillation for Multi-Token Prediction" (arXiv:2603.23911).** Emphasizes that keeping MTP heads aligned with the main head matters; that inspired the lightweight head-copy initialization used here.
- **Sun et al., "LayerNorm Scaling" (arXiv:2502.05795).** Already reflected in the chosen base; retained unchanged because the repo's best non-TTT runs consistently benefit from it.

## What changed versus the chosen base

1. `MTP_NUM_HEADS` now defaults to `2` instead of `0`.
2. `MTP_LOSS_WEIGHT` now defaults to a conservative `0.15`.
3. Added `MTP_INIT_FROM_MAIN_HEAD=1` (default on), which copies the tied main projection into each MTP head at initialization instead of leaving future heads zero-initialized.
4. Added logging for the new MTP init mode.
5. Kept the base export path unchanged: `mtp_heads` are still stripped before serialization, so the candidate preserves the base artifact budget behavior.

## How to run

Run from this candidate directory. The script resolves the default dataset and tokenizer paths relative to the repository root, so it does not require `DATA_PATH` or `TOKENIZER_PATH` overrides for the standard challenge layout:

```bash
cd candidates/202604041652_mtp-headcopy
SEED=1337 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful MTP knobs:

```bash
MTP_NUM_HEADS=2
MTP_LOSS_WEIGHT=0.15
MTP_INIT_FROM_MAIN_HEAD=1
```

## Main expected risks and tradeoffs

- The auxiliary heads add training compute, so a bad head-count / weight choice could reduce total steps enough to wash out the sample-efficiency gain.
- Copy-initializing from the main head may over-bias future-token heads toward next-token behavior if the weight is too large.
- Because the repo's recent gains depend heavily on the quantization-friendly late-training regime, MTP could help early optimization but still hurt the endgame if it regularizes too strongly.
- No head-count or loss-weight sweep is included yet; `2` heads at `0.15` is a deliberately conservative first setting.

## Validation

- `python -m compileall candidates/202604041652_mtp-headcopy/train_gpt.py` — passed
- Minimal CPU runtime smoke test — not feasible in this environment because the candidate still hard-requires the CUDA/FlashAttention challenge stack for execution; only syntax-level validation was safe to run here.
