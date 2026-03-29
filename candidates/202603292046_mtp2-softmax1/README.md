# MTP2 + Softmax-1 Null Slot on the 11L EMA/GPTQ-lite Stack

## Hypothesis

The strongest non-TTT branch in this repository is already very good at fitting more useful capacity under the 16MB cap, but it still depends on aggressive low-bit export and limited training time. External research points to **multi-token prediction (MTP)** as the highest-ROI training-only improvement for short-budget language-model training, because it increases sample efficiency without needing to keep the auxiliary heads in the exported artifact. This candidate therefore turns on **2-step MTP by default** on top of the strongest reusable 11-layer stack.

I pair that with a **softmax-1 reformulation** motivated by recent attention-sink/quantization work. The softmax-1 path gives every attention head an explicit "attend to nothing" option with **zero added parameters**.

Concretely:

- training uses **2 auxiliary next-token heads** (`MTP_NUM_HEADS=2`), whose parameters are dropped before export, and
- each attention layer prepends a zero query/key/value slot before causal attention and then drops the first output position afterward. Under ordinary causal softmax, that is equivalent to computing softmax over the original logits plus one extra logit fixed at zero, i.e. `softmax-1`.

## Why this is promising for this repository

This challenge is judged after compression, not just after dense bf16 training. That makes **both sample efficiency and quantization robustness** unusually valuable.

Repository review suggests the best recent gains came from:

- the 11-layer / 3x MLP / XSA / BigramHash / Partial-RoPE stack,
- EMA and better warmdown tuning,
- improved post-training quantization such as GPTQ-lite,
- and evaluation-aware tricks such as sliding-window scoring.

The best non-TTT reusable base is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`.

MTP is attractive here because it is:

- training-only, so the extra heads can be dropped before export,
- explicitly motivated by better sample efficiency per observed token,
- already wired into the selected base script but left disabled in the reviewed records,
- and therefore very cheap to test in this repository.

Softmax-1 remains attractive here because it is:

- parameter-free,
- cheap enough to fit the same 10-minute regime,
- directly motivated by quantization behavior rather than only pre-quant loss,
- and implementable as a precise attention-path change instead of new infrastructure.

I also carried over the **LeakyReLU(0.5)^2** MLP activation from `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, since that record reported a clear pre-TTT improvement from the activation alone and it composes naturally with the 2026-03-22 base.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Important local influences:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - established Partial RoPE + layerwise LN scaling on the 11-layer stack.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - established XSA on late layers and EMA as a strong default.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - showed that LeakyReLU(0.5)^2 is a meaningful local win even before TTT.

Also relevant:

- multiple strong record scripts already include dormant `MTP_NUM_HEADS` / `MTP_LOSS_WEIGHT` support but keep it at `0`, so this candidate is intentionally the first reviewed branch here to turn that training-only path on by default.

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

- **Gloeckle et al., 2024 — "Better & Faster Large Language Models via Multi-token Prediction"**  
  arXiv: `2404.19737` — <https://arxiv.org/abs/2404.19737>

  This was the strongest external recommendation for this repository: it promises better sample efficiency from a training-only auxiliary objective, which is especially attractive under a hard 10-minute budget and a strict artifact cap.

- **Kaul et al., 2024 — "Transformer Softmax Attacks the First Token: A Study on Attention Sinks and Quantization"**  
  arXiv: `2410.17174` — <https://arxiv.org/abs/2410.17174>

  This paper reports two effects that are directly relevant here: strong first-token attention dominance and poor low-bit quantization behavior. Their proposed **softmax-1** reformulation sharply reduces the first-token sink and preserves quality under aggressive weight quantization. That is the main motivation for the secondary softmax-1 change in this candidate.

- **Gemma 2 Technical Report (Riviere et al., 2024)**  
  arXiv: `2408.00118` — <https://arxiv.org/abs/2408.00118>

  Gemma 2 is not the direct implementation target here, but it is a useful reference point that small models often gain from narrow architecture changes that improve efficiency without requiring a wholesale systems rewrite. Compared with Gemma 2's local-global attention changes, softmax-1 is much easier to slot into this repository.

## What changed versus the chosen base implementation

Starting from `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes four focused changes:

1. **MTP enabled by default**
   - `MTP_NUM_HEADS` now defaults to `2` in this candidate.
   - `MTP_LOSS_WEIGHT` remains configurable (default `0.2`).
   - The script keeps MTP heads in the dense `final_model.pt` checkpoint for reload compatibility, but excludes them from the compressed submission artifact, so this primarily changes training.

2. **Softmax-1 attention via a prepended null slot**
   - New env flag: `SOFTMAX1_ENABLED` (default `1`).
   - Each attention layer prepends a zero query/key/value token, runs ordinary causal attention, then discards the first output position.
   - This preserves the rest of the attention stack and keeps the change local to the attention path.

3. **LeakyReLU(0.5)^2 MLP activation**
   - New env flag: `MLP_NEGATIVE_SLOPE` (default `0.5`).
   - Replaces `relu(x)^2` with `leaky_relu(x, 0.5)^2`.

4. **Non-FlashAttention fallback for import/smoke use**
   - If tensors are not on CUDA, the script falls back to a simple causal attention implementation for import/smoke use.
   - CUDA training/eval still **fails fast** if `flash_attn_interface` is unavailable, so the intended H100 path cannot silently degrade.

Everything else stays aligned with the selected base: 11 layers, 3x MLP, XSA on the last 4 layers, Partial RoPE, layerwise LN scale, VE on layers 9-10, EMA, GPTQ-lite int6 export, and sliding-window evaluation.

## How to run

From the repository root:

```bash
cd candidates/202603292046_mtp2-softmax1

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 \
SOFTMAX1_ENABLED=1 MLP_NEGATIVE_SLOPE=0.5 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For an ablation against the same script, disable the new idea with:

```bash
MTP_NUM_HEADS=0 SOFTMAX1_ENABLED=0 MLP_NEGATIVE_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate

The script follows the same evaluation flow as the base record:

- dense validation,
- EMA application,
- GPTQ-lite int6 export,
- round-trip validation,
- and sliding-window validation (`EVAL_STRIDE=64` by default).

The main metrics of interest remain the exact printed `final_int6_sliding_window_exact` / `final_int8_zlib_roundtrip_exact` values and the total submission size.

## Main expected risks and tradeoffs

- **MTP may trade off some one-step specialization.** The auxiliary loss is supposed to help sample efficiency, but it can also pull capacity toward short-horizon pattern prediction if the weight is too high.
- **Attention behavior may become too diffuse.** A null option can reduce sink behavior, but it may also weaken useful copy or recency behavior in early layers.
- **Compile/kernel interactions could matter.** The extra prefixed slot is small, but it slightly changes attention shapes and may interact with `torch.compile` or FlashAttention performance in ways that only show up on real GPU runs.
- **Quantization gain is still a hypothesis here.** The motivating paper shows strong quantization robustness results, but this repository already uses a different small-model stack, GPTQ-lite export, and XSA.
- **LeakyReLU(0.5)^2 may not transfer perfectly.** It helped in the current top record lineage, but that record also used different systems and TTT choices.

## Validation run for this candidate

Commands attempted in this workflow environment:

```bash
python -m compileall candidates/202603292046_mtp2-softmax1/train_gpt.py
```

Outcome: **passed**.

Attempted minimal synthetic CPU smoke check (import the file, instantiate `GPT`, run a tiny forward pass on random tokens) but the workflow environment did not have `torch` installed, so that runtime smoke test was **not feasible here** without adding new heavyweight dependencies.
