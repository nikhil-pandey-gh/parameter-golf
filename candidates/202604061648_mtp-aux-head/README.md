# 202604061648_mtp-aux-head

## Hypothesis

Turn on a **single training-only multi-token prediction (MTP) auxiliary head** on top of the current best `LeakyReLU² + Legal TTT + Parallel Muon` stack. The extra head predicts token *t+2* from the shared final hidden state, which should improve sample efficiency during the 10-minute train window without increasing submission artifact size because the MTP head is excluded from export.

## Why this is promising here

The repository's strongest runs already show that the main bottlenecks are **sample efficiency under a fixed 600s budget** and **keeping improved training dynamics within the 16MB artifact cap**. MTP is unusually well matched to this challenge because:

1. it is a **training-time-only** improvement,
2. the existing record stack already contains dormant MTP support,
3. the export path already strips `mtp_heads` from the final state dict.

That makes it one of the few modern ideas that can plausibly improve the best stack **without paying artifact bytes** and without needing broad new infrastructure.

## Prior repository evidence that informed this candidate

- **Base stack:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - strongest current mean score (`val_bpb: 1.1194`)
  - best overall combination of LeakyReLU², Partial RoPE, XSA, VE, EMA/SWA, GPTQ-lite int6, legal TTT, and Parallel Muon
- **Closest no-TTT base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - strongest pure train/export stack before legal TTT was added
- **Important dormant signal:** the 2026-03-20, 2026-03-21, 2026-03-22, and 2026-03-23 scripts all already include MTP code paths, but their logs still report `mtp_num_heads:0`, so this idea exists in the codebase but has not actually been exercised in a record run.

## External research that informed it

- **Better & Faster Large Language Models via Multi-Token Prediction** ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737))
  - argues that predicting multiple future tokens from a shared trunk improves sample efficiency and can strengthen induction-style behavior
  - especially attractive here because our candidate can treat MTP as an auxiliary loss and then discard the extra head at export time
- **LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding in Large Language Models** ([arXiv:2404.16710](https://arxiv.org/abs/2404.16710))
  - reinforces the broader idea that **auxiliary supervision on shared hidden states** can make a language model learn useful representations earlier
  - I did **not** choose LayerSkip directly because it requires a more invasive training recipe and inference framing than this repo needs
- **Diff Transformer** ([arXiv:2410.05258](https://arxiv.org/abs/2410.05258))
  - promising attention alternative, but materially more invasive than toggling the dormant MTP path
- **Hyper-Connections** ([arXiv:2409.19606](https://arxiv.org/abs/2409.19606))
  - interesting residual-path alternative for deep models, but again would require a much broader rewrite than this candidate

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS` default changed from `0` to `1`
   - keeps the existing `MTP_LOSS_WEIGHT=0.2`
   - this adds one auxiliary horizon (`t+2`) while keeping extra compute smaller than a multi-head MTP sweep
2. **Add a safe CPU smoke path**
   - `SMOKE_TEST=1` instantiates the model on CPU, runs a tiny forward/backward step, validates the same optimizer-group assembly used by the real training path, confirms MTP heads are excluded from the exported state dict, reloads that stripped state into a zero-MTP eval model with `strict=True`, and asserts that the logits match
   - this is only for local validation in environments like this workflow; it does not change the intended GPU training path
3. **Add an attention fallback for smoke validation**
   - `flash_attn_interface` remains required for full leaderboard-style runs
   - the `scaled_dot_product_attention` fallback is now restricted to `SMOKE_TEST=1`, so normal runs fail loudly instead of silently changing the performance profile

## How to run

### Full candidate run

From this candidate directory:

```bash
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TTT_ENABLED=1 \
MTP_NUM_HEADS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Important defaults inherited from the base stack:

- `NUM_LAYERS=11`
- `TRAIN_SEQ_LEN=2048`
- `TRAIN_BATCH_TOKENS=786432`
- `EVAL_STRIDE=64`
- `WARMDOWN_ITERS=3500`
- `XSA_LAST_N=4`
- `ROPE_DIMS=16`
- `VE_ENABLED=1 VE_LAYERS=9,10`
- `TTT_ENABLED=0` in code, so set it explicitly for the full legal-TTT evaluation recipe

### Lightweight smoke validation

```bash
SMOKE_TEST=1 \
VOCAB_SIZE=128 \
MODEL_DIM=64 \
NUM_LAYERS=4 \
NUM_HEADS=4 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TRAIN_SEQ_LEN=32 \
TRAIN_BATCH_TOKENS=128 \
BIGRAM_VOCAB_SIZE=128 \
BIGRAM_DIM=32 \
VE_ENABLED=0 \
XSA_LAST_N=0 \
TTT_ENABLED=0 \
python train_gpt.py
```

## Expected risks and tradeoffs

- **Step-time regression:** even one auxiliary vocab head adds compute; the gain must outweigh any lost training steps.
- **Horizon mismatch:** the simple existing implementation predicts future tokens from the final hidden state with independent heads, which is lighter than the full paper recipe but may also be weaker.
- **TTT interaction is uncertain:** MTP could improve the base model and help TTT, but it could also smooth the representation in a way that reduces downstream adaptation headroom.
- **No artifact penalty does not mean no optimization penalty:** the exported model stays the same size class, but training dynamics could still worsen if the auxiliary loss is overweighted.
- **The repo's optimizer split was easy to get wrong:** this candidate explicitly wires MTP heads into the AdamW small-matrix group and raises at runtime if they are omitted from optimizer groups.

## Validation

- `python -m compileall candidates/202604061648_mtp-aux-head/train_gpt.py`
  - **Outcome:** passed in this workflow
- CPU smoke test command above
  - **Outcome:** not completed in this workflow environment because the runner did not have `torch` installed, and an attempted temporary PyTorch install was blocked by the network/proxy path used for wheel resolution

So the candidate is syntax-validated here, and the smoke path is implemented and documented, but it still needs to be executed in a runner that already has PyTorch available.
