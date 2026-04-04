# Primer-Style Depthwise QKV Mixing on the 11L XSA/TTT Stack

## Hypothesis

Add the missing **Primer** attention-side improvement - a small **causal depthwise convolution after the Q/K/V projections** - to the strongest local stack in this repository.

The current record line already adopted several pieces that are complementary to this idea:

- the **squared-activation family** (`LeakyReLU(0.5)^2`) from the same Primer paper,
- **query/key normalization** before attention,
- **late-layer specialization** via XSA on the last 4 blocks,
- aggressive quantization-aware export and parameter banking.

The hypothesis is that **late-layer local token mixing on Q/K/V** will improve short-range pattern formation without paying the full runtime cost of adding the convolution to all 11 layers.

## Why this looks promising here

Repository review points to a clear trend:

1. **Compression-aware training and export dominate**: warmdown tuning, fp16 embeddings, mixed int6/int8, EMA, GPTQ-lite, and low-overhead architectural tweaks consistently improved BPB.
2. **Cheap inductive biases help**: SmearGate, BigramHash, XSA, Partial RoPE, and LN scaling all paid off.
3. **The best overall run** (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`) already moved to the Primer activation family, but none of the prior records added Primer's **depthwise Q/K/V mixing**.
4. The non-record exploration explicitly reported that **layer recurrence was bad** under the 10-minute budget, which argues for a targeted architectural bias rather than heavier structural reuse.

So this candidate keeps the winning 11-layer banked/XSA/EMA/TTT path intact and adds a single new mechanism with a strong language-modeling prior.

## Prior work that influenced this candidate

### Root baseline

- `train_gpt.py` in the repository root: 9-layer 512d GQA baseline with Muon + Adam split, tied embeddings, and simple int8 export.

### Most relevant records

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
  - established the 11-layer XSA + EMA line.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - showed Partial RoPE + LN scaling mattered, while the late-QAT path in that snapshot was effectively inactive under `torch.compile`.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - tightened export quality with GPTQ-lite clip search and longer warmdown.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - best local result and direct implementation base for this candidate.

### Prior candidates

- No prior `candidates/` directory existed at review time.

## External research that informed the choice

- **Primer: Searching for Efficient Transformers for Language Modeling** (So et al., 2021, arXiv:2109.08668)
  - identified two main wins over a standard Transformer: **squared ReLU** and **depthwise convolution after each Q/K/V projection**.
  - This repository already absorbed the activation-side idea; this candidate tests the attention-side half.
- **Query-Key Normalization for Transformers** (Henry et al., 2020, arXiv:2010.04245)
  - motivated the current repository trend of normalizing Q and K before softmax saturation becomes unstable.
  - The chosen base already does this, so Primer-style local mixing slots into an attention stack that is already norm-stabilized.
- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** (Lan et al., 2019, arXiv:1909.11942)
  - was considered as an alternative direction because cross-layer sharing reduces bytes.
  - I did **not** choose it here because prior repository evidence says the 10-minute track is more sensitive to quality-per-step and quantization behavior than to raw parameter count alone.

## What changed versus the chosen base

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps:

- parameter banking + parallel Muon,
- 11 layers at 512d with 8 heads / 4 KV heads,
- XSA on the last 4 layers,
- Partial RoPE, LN scaling, VE layers, EMA/SWA plumbing,
- LeakyReLU(0.5)^2 MLP,
- legal score-first TTT machinery,
- mixed int6 export + lzma roundtrip validation.

This candidate adds:

1. **`CausalDepthwiseConv1d`**
   - a tiny learnable depthwise causal convolution initialized to identity.
2. **Selective Primer-style Q/K/V mixing**
   - Q, K, and V are locally mixed **after projection and before attention**.
3. **Late-layer targeting**
   - controlled by:
     - `PRIMER_DWCONV=1`
     - `PRIMER_DWCONV_KERNEL=3`
     - `PRIMER_DWCONV_LAST_N=4`
   - by default, only the last 4 blocks use the new mixers, matching the repo's existing late-layer XSA specialization pattern.
4. **Control/export integration**
   - the depthwise kernels are treated as small control tensors so they stay in the normal optimizer/export path without changing the wider banked-quantization flow.

## How to run

From the repository root:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
PRIMER_DWCONV=1 PRIMER_DWCONV_KERNEL=3 PRIMER_DWCONV_LAST_N=4 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 candidates/202604040304_primer-dwqkv-xsa/train_gpt.py
```

To ablate the new idea while keeping the rest of the stack fixed:

```bash
PRIMER_DWCONV=0 torchrun --standalone --nproc_per_node=8 candidates/202604040304_primer-dwqkv-xsa/train_gpt.py
```

## Validation run in this environment

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604040304_primer-dwqkv-xsa/train_gpt.py` | Passed |
| `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604040304_primer-dwqkv-xsa/train_gpt.py` | Passed |
| `python - <<'PY' ... import torch ...` | Failed: `ModuleNotFoundError: No module named 'torch'` |

### Notes

- A real startup smoke test was **not feasible** in this runner because the local Python environment does not have `torch` installed.
- Even with `torch`, this candidate inherits the chosen base's **CUDA + FlashAttention** runtime requirements, so a meaningful end-to-end smoke test belongs on a runner with those dependencies available.

## Main risks and tradeoffs

- **Step-time risk**: the added depthwise mixers may slightly slow late-layer attention, reducing total steps within the 600s cap.
- **Over-localization risk**: local Q/K/V smoothing could help n-gram structure but partially fight XSA or long-context abstractions if it is too aggressive.
- **Export-size risk**: the new kernels are small, but they are extra non-banked tensors and still consume some artifact budget.
- **Interaction risk with TTT**: the best local score currently relies on legal TTT; if the new attention bias changes adaptation dynamics, pre-TTT and post-TTT gains may move differently.

## Suggested next experiments

1. Sweep `PRIMER_DWCONV_LAST_N` over `2, 4, 6`.
2. Compare `PRIMER_DWCONV_KERNEL=3` vs `5`.
3. Measure the effect both **with and without TTT**, because the repo evidence suggests activation and evaluation gains can interact non-linearly.
4. If step-time stays acceptable, try the same mixer on the `2026-03-22` pre-TTT stack to isolate the architectural gain without TTT noise.
