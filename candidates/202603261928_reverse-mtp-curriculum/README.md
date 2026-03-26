# Reverse-Curriculum MTP on the current 11L + Legal TTT stack

## Hypothesis

The current best line in this repo is already strong on architecture, export, and evaluation: 11 layers, LeakyReLU(0.5)^2, XSA, partial RoPE, EMA, GPTQ-lite/int6 export, and legal score-first TTT. A cleaner next bet is to improve **training signal density** instead of adding another export or eval trick.

This candidate turns on **training-only multi-token prediction (MTP)** and anneals it back toward pure next-token prediction during warmdown. The hypothesis is that:

1. early MTP improves sample efficiency under the 600s budget,
2. the reverse curriculum avoids leaving too much auxiliary-task bias in the final checkpoint, and
3. because the MTP heads are dropped before export, any gain comes at effectively zero final artifact cost.

## Why this is promising for this repository

The repo review suggests three important facts:

- the strongest scores now come from the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` stack rather than from another one-off architectural rewrite,
- recent gains after the big 11-layer jump have mostly been incremental and training-budget-sensitive, and
- the strongest current script already contains dormant MTP hooks, but no published record actually reports an MTP run.

That makes MTP a particularly good candidate here: it is adjacent to the winning implementation, cheap in final bytes, and directly targeted at the challenge's main bottleneck: **how much useful training signal we can squeeze into 10 minutes**.

## Prior repository work that influenced this candidate

### Primary base

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - strongest current record in this checkout
  - contributes the banked 11-layer stack, LeakyReLU^2 MLP, parallel Muon, legal TTT, and int6+lzmа export path

### Immediate lineage

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - shows the modern 11L stack is already export-efficient and quantization-sensitive
  - important because this candidate keeps that export discipline and spends novelty on training signal instead

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - confirms the current line benefits from small, targeted training/architecture tweaks rather than wholesale rewrites

### Why not an eval-only follow-up

- `records/track_10min_16mb/2026-03-17_LoRA_TTT`
  - showed that document-aware eval and TTT can help a lot, but this repository already has a mature legal TTT path in the strongest record
  - this candidate therefore focuses on **training-time efficiency** instead of piling on more evaluation complexity

## External research that informed the idea

### 1. Fabian Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (arXiv:2404.19737, 2024)

The key result is that predicting multiple future tokens with separate auxiliary heads improves sample efficiency while preserving the shared trunk. That is exactly the shape of intervention that fits this challenge: more learning signal during the same wallclock budget, without paying final artifact bytes if the auxiliary heads are excluded from export.

### 2. Ansar Aynetdinov and Alan Akbik, *Pre-Training Curriculum for Multi-Token Prediction in Language Models* (arXiv:2505.22757, ACL 2025)

This paper is particularly relevant because it studies **small language models**. Their results suggest that SLMs do better with a curriculum over the MTP objective, and that **reverse curriculum** (stronger MTP earlier, then decaying toward NTP) gives the best final next-token quality. Since this repository is evaluated on next-token BPB rather than speculative decoding speed, reverse curriculum is the better fit.

## What changed versus the chosen base implementation

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes five concrete changes:

1. **Enables MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`

2. **Adds reverse-curriculum scheduling for the MTP loss**
   - `MTP_CURRICULUM=reverse`
   - `MTP_WARMDOWN_SCALE=0.25`
   - `MTP_FINAL_LOSS_WEIGHT=0.0`
   - MTP stays fully active through most of training, then decays toward pure NTP late in warmdown.

3. **Actually wires the MTP heads into optimization**
   - the copied base script already had `mtp_heads`, but they were not placed into any optimizer group
   - this candidate adds a dedicated AdamW optimizer for them via `MTP_HEAD_LR`
   - the auxiliary heads are initialized from the token embedding matrix so they start providing useful trunk gradients immediately

4. **Makes the MTP weight compile-safe to change at runtime**
   - the weight is stored in a mutable buffer instead of as a constant Python float read inside the compiled graph
   - this mirrors the repo lesson from earlier QAT attempts: avoid runtime toggles that are easy for `torch.compile` to constant-fold away

5. **Adds candidate-local validation ergonomics**
   - repo-root-relative data/tokenizer defaults, so the script can be run from the candidate directory itself
   - `SMOKE_TEST=1` CPU path
   - FlashAttention import fallback to SDPA so smoke testing works without Hopper-specific kernels

## How to run

From the candidate directory:

```bash
cd candidates/202603261928_reverse-mtp-curriculum

torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To include the existing legal TTT path during evaluation, add:

```bash
TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate keeps the strong-stack defaults from the March 23 record, but changes the MTP-related defaults. Useful knobs:

```bash
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
MTP_FINAL_LOSS_WEIGHT=0.0 \
MTP_CURRICULUM=reverse \
MTP_WARMDOWN_SCALE=0.25 \
MTP_HEAD_LR=0.025
```

## Validation run for this candidate

I ran the following lightweight checks in this workflow environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603261928_reverse-mtp-curriculum/train_gpt.py
```

Outcome: **passed**.

Because the runner did not have the required Python packages preinstalled, I created a local virtual environment under `/tmp/gh-aw/agent/pgolf-venv` and ran a minimal CPU smoke test with tiny settings:

```bash
python -m venv --system-site-packages /tmp/gh-aw/agent/pgolf-venv
source /tmp/gh-aw/agent/pgolf-venv/bin/activate
pip install torch numpy sentencepiece
SMOKE_TEST=1 MODEL_DIM=64 NUM_LAYERS=2 NUM_HEADS=4 NUM_KV_HEADS=2 \
MLP_MULT=2 VOCAB_SIZE=128 TRAIN_SEQ_LEN=16 BIGRAM_VOCAB_SIZE=128 \
VE_ENABLED=0 XSA_LAST_N=1 MTP_NUM_HEADS=2 \
python train_gpt.py
```

Outcome:

- `smoke_test_loss:5.576912`
- `smoke_test_attention:sdpa_fallback`

That confirms the candidate script imports, builds the model, executes the MTP path, and completes a forward pass without immediately crashing.

## Main expected risks / tradeoffs

- **Training throughput risk.** Even with only two MTP heads, auxiliary logits still cost compute. The gain must outweigh any step-count loss.
- **Schedule sensitivity.** The best MTP weight and warmdown crossover are probably narrow; `0.15 / 0.25` is a principled starting point, not a proven optimum.
- **Compilation behavior on the full GPU run.** The mutable-buffer approach is designed to be safer than a Python-side constant toggle, but the true wallclock behavior still needs an 8xH100 run.
- **Head learning rate sensitivity.** The MTP heads are now actually optimized; `MTP_HEAD_LR` may need a quick sweep.
- **TTT interaction uncertainty.** Reverse-curriculum MTP may improve the pre-TTT checkpoint, but the final gain after legal TTT is not guaranteed to be additive.

## Suggested next experiments if this is promising

1. Sweep `MTP_NUM_HEADS` in `{1, 2, 4}`.
2. Sweep `MTP_LOSS_WEIGHT` in `{0.05, 0.10, 0.15, 0.20}`.
3. Sweep `MTP_WARMDOWN_SCALE` in `{0.15, 0.25, 0.35}`.
4. Compare `MTP_CURRICULUM=reverse` against `none` on the same base.
5. If pre-TTT improves, test whether legal TTT still gives the same marginal gain or needs retuning.
