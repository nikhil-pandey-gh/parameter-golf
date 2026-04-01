# Candidate: Training-Only Multi-Token Prediction on the LeakyReLU² + Legal TTT stack

## Hypothesis

The repository's strongest published stack already contains dormant support for multi-token prediction (MTP): auxiliary heads that predict multiple future tokens during training and are explicitly excluded from export. Turning that path on should improve sample efficiency and representation quality without increasing the final artifact size, making it a strong fit for the challenge's 16MB budget.

Concretely, this candidate enables a conservative **2-head MTP auxiliary loss** on top of the current best public recipe, betting that extra future-token supervision is worth more than the modest training-time overhead.

## Why this is promising for this repository

The local record history says the frontier is already heavily mined along architecture, eval, and export axes:

- sliding-window eval, FP16-sensitive tensors, int6/int5 mixed quantization, EMA/SWA, XSA, Partial RoPE, LN scaling, and legal TTT have all already been exploited,
- while explicit recurrence / depth reuse looked weak under the repository's fixed 10-minute wallclock budget,
- and the latest best script already contains an unused MTP path plus export logic that strips MTP heads from the saved artifact.

That combination makes MTP unusually attractive here: it is **distinct from prior records**, requires **minimal code churn**, and adds **training-only capacity** rather than permanent artifact bytes.

## Which records influenced this candidate

Primary local influences:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current public stack,
  - already includes the MTP code path and already excludes `mtp_heads` from export.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - establishes the strong pre-TTT 11-layer EMA/GPTQ-lite base this candidate inherits indirectly.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - shows that by this stage improvements are increasingly about targeted, low-overhead deltas rather than wholesale rewrites.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful negative evidence that throughput-heavy ideas like SwiGLU or layer recurrence can lose under a fixed wallclock budget.

## External research that informed it

Primary sources:

- **Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"** (`https://arxiv.org/abs/2404.19737`)
  - argues that predicting multiple future tokens with independent heads on a shared trunk improves sample efficiency and downstream capability.
- **DeepSeek-AI, "DeepSeek-V3 Technical Report"** (`https://arxiv.org/abs/2412.19437`)
  - explicitly calls out a multi-token prediction training objective as part of a state-of-the-art large-model recipe.

Repository-context research that I considered but did not follow directly:

- **ALBERT** (`https://arxiv.org/abs/1909.11942`) and related parameter-sharing work make recurrence/share-heavy designs look attractive on paper,
  but this repo already has negative local evidence on recurrence under the 10-minute budget.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

- set `MTP_NUM_HEADS=2` by default,
- set `MTP_LOSS_WEIGHT=0.15` by default,
- wire `mtp_heads` into the auxiliary head optimizer and replicated-gradient sync so the MTP loss actually trains both the heads and the shared trunk,
- bake the base recipe's legal-TTT defaults into the copied script (`TTT_ENABLED=1`, `TTT_FREEZE_BLOCKS=0`) so running this directory uses the intended stack without extra flags,
- otherwise leave the best-known LeakyReLU² + legal TTT + Parallel Muon stack unchanged.

Important implementation note:

- the script already filters `mtp_heads` out of `export_sd` before quantization/export, so the auxiliary heads are **training-only** and do **not** count toward the final artifact.

## How to run or evaluate it

From the repository root:

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 \
  candidates/202604010943_multi-token-prediction/train_gpt.py
```

Key defaults inherited from the base recipe include:

- 11 layers, 512 dim, 8 heads / 4 KV heads,
- LeakyReLU(0.5)^2 MLP,
- BigramHash + XSA + Partial RoPE + VE,
- EMA + tight SWA,
- GPTQ-lite int6 export + lzma,
- legal score-first TTT enabled,
- and now **2-head training-only MTP**.

If you want to ablate the idea cleanly:

```bash
MTP_NUM_HEADS=0 SEED=1337 torchrun --standalone --nproc_per_node=8 \
  candidates/202604010943_multi-token-prediction/train_gpt.py
```

## Main expected risks or tradeoffs

- **Training-time overhead:** MTP adds extra output heads and cross-entropy terms, so step time may rise enough to offset some of the sample-efficiency gain.
- **Under-tuned auxiliary weight:** `0.15` is a conservative starting point, not a sweep result.
- **Interaction with TTT:** the current best stack already relies on legal TTT at eval time; MTP may help the base model, but the combined effect is uncertain.
- **Export mismatch remains possible:** because MTP heads are dropped before export, benefits must transfer into the shared trunk rather than the discarded auxiliary heads themselves.

## Validation

Validation run from the repository root:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604010943_multi-token-prediction/train_gpt.py
python - <<'PY'
from pathlib import Path
import ast

text = Path("candidates/202604010943_multi-token-prediction/train_gpt.py").read_text(encoding="utf-8")
assert 'MTP_NUM_HEADS", 2' in text
assert 'MTP_LOSS_WEIGHT", 0.15' in text
assert 'TTT_ENABLED", "1"' in text
assert 'TTT_FREEZE_BLOCKS", 0' in text
assert 'aux_head_params = [head.weight for head in base_model.mtp_heads]' in text
assert 'replicated_params.extend(aux_head_params)' in text
assert 'export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}' in text
ast.parse(text)
print("static_check_ok")
PY
```

Observed outcomes:

- `compileall`: passed.
- `static_check_ok`: passed, confirming the intended MTP defaults, the optimizer wiring for `mtp_heads`, and the export-time MTP stripping logic are present in the candidate script.

I did **not** run a real CUDA training smoke here because this environment does not provide the challenge's GPU/runtime stack (`torchrun`, CUDA, FlashAttention 3, and the cached dataset/tokenizer) in a way that would make the result meaningful. I also could not run a CPU forward-pass smoke because the container does not have `torch` installed.
