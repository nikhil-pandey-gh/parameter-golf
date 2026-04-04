# Candidate: Gated Attention + Value Residual on the 11L banked TTT stack

## Hypothesis

The current frontier in this repository keeps improving by stacking tiny, low-byte changes on top of the strong 11-layer banked backbone: XSA, partial RoPE, LN scaling, EMA/SWA, GPTQ-lite, and legal score-first TTT. This candidate extends that pattern by turning on two already-plumbed but unpublished mechanisms in the latest banked model:

1. **Head-wise gated attention** after SDPA.
2. **Value residual mixing** that keeps the first-layer value stream available deeper in the network.

The bet is that these two changes improve information routing and suppress noisy attention behavior without meaningfully increasing artifact size or engineering complexity.

## Why this looks promising here

- The best repo gains after the early quantization jump have mostly come from **small attention/eval/export changes**, not from broad rewrites.
- The historical review showed **layer recurrence** and similar compute-heavy reuse ideas were negative under the 10-minute cap, while **zero- or tiny-parameter attention tweaks** kept paying off.
- Both gated attention and value residuals are cheap in this codebase: they add only small control tensors, reuse the existing optimizer/export path, and fit naturally beside XSA, partial RoPE, VE, and legal TTT.

## Prior repository evidence that influenced this candidate

- **Root baseline**: 9L/512d GPT with Muon, GQA, U-Net skips, tokenizer-agnostic BPB eval, and exact quantized roundtrip scoring (`../../train_gpt.py`).
- **`records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`**: established 11 layers + XSA + EMA as the first strong 11L frontier.
- **`records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`**: showed that tiny attention-side changes like partial RoPE and LN scaling still move BPB.
- **`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`**: reinforced that export-aware refinements and VE can stack on top of the same backbone.
- **`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`**: supplied the chosen base implementation because it already contains the banked optimizer path, LeakyReLU^2 MLP, legal score-first TTT, and dormant gated-attention/value-residual hooks.
- There were **no existing `candidates/` directories** at review time, so this idea is not overlapping a prior candidate folder.

## External research that informed it

- **Qiu et al., _Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free_** ([arXiv:2505.06708](https://arxiv.org/abs/2505.06708)): reports that a simple **head-specific sigmoid gate after SDPA** consistently improved performance, stability, and long-context behavior.
- **Zhou et al., _Value Residual Learning_** ([arXiv:2410.17897](https://arxiv.org/abs/2410.17897)): argues that preserving token information through **value residual connections** improves deep information flow and can reach the same loss with fewer parameters/data.
- **He et al., _RealFormer: Transformer Likes Residual Attention_** ([arXiv:2012.11747](https://arxiv.org/abs/2012.11747)): shows residual attention can stabilize optimization and induce sparser attention, which makes a residualized attention/value path especially relevant for tiny models.

## What changed versus the chosen base implementation

Base: `../../records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps that script's architecture, export path, and training loop intact, but changes the default configuration so the new hypothesis is active by default:

- `GATED_ATTENTION=1`
- `VALUE_RESIDUAL=1`

It also makes two surgical usability changes:

- dataset/tokenizer defaults resolve from the repository root so the script can be launched directly from this candidate directory,
- startup logs now print the gated-attention/value-residual/TTT settings explicitly.

## How to run

From this directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

That default run keeps the new idea isolated. The main pre-TTT comparison metric is:

- `final_int6_sliding_window_exact` (with the default `EVAL_STRIDE=64`, this is the stride-64 score)

If you want to test compatibility with the current legal score-first TTT recipe as a follow-up, run:

```bash
TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

In that mode:

- `final_int6_sliding_window_exact` is still the **pre-TTT** score
- `legal_ttt_exact` is the **post-TTT** score

Useful ablations:

```bash
# Gate-only ablation
VALUE_RESIDUAL=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Value-residual-only ablation
GATED_ATTENTION=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Current-record-style eval on top of this candidate
TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

By default the script resolves `DATA_PATH` and `TOKENIZER_PATH` from the repo root. Override them explicitly if your data lives elsewhere.

## Main expected risks and tradeoffs

- **No direct 8xH100 ablation yet**: this folder packages the candidate cleanly, but the actual BPB delta still needs a real multi-GPU run.
- **Attention gating could over-prune useful heads** if the learned gate saturates too aggressively.
- **Value residual mixing can blur late-layer specialization** if the model leans too hard on the first-layer value stream.
- **Legal TTT remains expensive at eval time**, so it can mask small training-side improvements unless pre-TTT and post-TTT numbers are both tracked.

## Validation

- `cd candidates/202604042311_gated-value-residual && python -m compileall train_gpt.py` — **passed**
- repo-root path resolution check — **passed** for path derivation (`REPO_ROOT=/home/runner/work/parameter-golf/parameter-golf`), but this workspace does **not** include `data/datasets/fineweb10B_sp1024/` or `data/tokenizers/`
- minimal CPU-only start test — **not feasible in this workspace** because `torch` and `flash_attn_interface` are not installed here, and the candidate intentionally keeps the CUDA/FlashAttention frontier runtime path unchanged
